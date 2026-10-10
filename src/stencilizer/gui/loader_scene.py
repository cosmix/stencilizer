"""Geometry for the loading view: wordmark letters, cutter paths, layout and theme colours."""

import re
from bisect import bisect_right
from dataclasses import dataclass
from importlib.resources import files
from math import hypot

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QPainterPath, QPalette, QPolygonF, QTransform

from stencilizer.gui import assets
from stencilizer.gui.header import WORDMARK_FILE

_TOKEN = re.compile(r"[MLHVQCZ]|-?\d*\.?\d+(?:e-?\d+)?")
_PATH_DATA = re.compile(r'\sd="([^"]+)"')
_VIEW_BOX = re.compile(r'viewBox="([^"]+)"')

# The wordmark letters span 520 x 690 units at most (the i-dot sits at y=0); the plate inner
# area is sized for that frame so every letter keeps the same stroke width.
LETTER_FRAME = QRectF(0.0, 0.0, 520.0, 690.0)
PLATE_ASPECT = 1.08
PLATE_INSET = 0.09


class _PathReader:
    """Turn an absolute-command SVG path string into one ``QPainterPath`` per subpath."""

    def __init__(self, data: str) -> None:
        self._tokens = _TOKEN.findall(data)
        self._index = 0
        self._x = 0.0
        self._y = 0.0
        self._current = QPainterPath()
        self.subpaths: list[QPainterPath] = []

    def read(self) -> list[QPainterPath]:
        """Consume every command and return the subpaths in file order."""
        handlers = {
            "M": self._move,
            "L": self._line,
            "H": self._horizontal,
            "V": self._vertical,
            "Q": self._quad,
            "C": self._cubic,
            "Z": self._close,
        }
        command = "M"
        while self._index < len(self._tokens):
            token = self._tokens[self._index]
            if token in handlers:
                self._index += 1
                command = token
            handlers[command]()
            if command == "M":
                command = "L"
        self._close()
        return self.subpaths

    def _started(self) -> bool:
        return not self._current.isEmpty()

    def _number(self) -> float:
        value = float(self._tokens[self._index])
        self._index += 1
        return value

    def _point(self) -> QPointF:
        self._x = self._number()
        self._y = self._number()
        return QPointF(self._x, self._y)

    def _move(self) -> None:
        self._close()
        self._current.moveTo(self._point())

    def _line(self) -> None:
        self._current.lineTo(self._point())

    def _horizontal(self) -> None:
        self._x = self._number()
        self._current.lineTo(self._x, self._y)

    def _vertical(self) -> None:
        self._y = self._number()
        self._current.lineTo(self._x, self._y)

    def _quad(self) -> None:
        control = self._point()
        self._current.quadTo(control, self._point())

    def _cubic(self) -> None:
        first = self._point()
        second = self._point()
        self._current.cubicTo(first, second, self._point())

    def _close(self) -> None:
        if self._started():
            self._current.closeSubpath()
            self.subpaths.append(self._current)
        self._current = QPainterPath()


@dataclass(frozen=True)
class Letter:
    """One wordmark letter in SVG units: its filled path and the contours the cutter follows."""

    path: QPainterPath
    parts: tuple[QPainterPath, ...]
    bounds: QRectF


@dataclass(frozen=True)
class Wordmark:
    """The wordmark split into letters, left to right, plus its SVG view box."""

    letters: tuple[Letter, ...]
    view_box: QRectF


def _group_letter(parts: list[QPainterPath]) -> Letter:
    """Combine the subpaths sharing a horizontal span (an i and its dot) into one letter."""
    path = QPainterPath()
    for part in parts:
        path.addPath(part)
    return Letter(path=path, parts=tuple(parts), bounds=path.boundingRect())


def load_wordmark() -> Wordmark:
    """Read the packaged wordmark SVG and split its single path into letters."""
    svg = files(assets).joinpath(WORDMARK_FILE).read_text(encoding="utf-8")
    data_match = _PATH_DATA.search(svg)
    box_match = _VIEW_BOX.search(svg)
    if data_match is None or box_match is None:
        raise ValueError("wordmark SVG has no path data or view box")
    box = [float(value) for value in box_match.group(1).split()]
    subpaths = sorted(_PathReader(data_match.group(1)).read(), key=lambda p: p.boundingRect().x())
    groups: list[list[QPainterPath]] = []
    for subpath in subpaths:
        if groups and subpath.boundingRect().x() < max(
            p.boundingRect().right() for p in groups[-1]
        ):
            groups[-1].append(subpath)
        else:
            groups.append([subpath])
    letters = tuple(_group_letter(group) for group in groups)
    return Wordmark(letters=letters, view_box=QRectF(box[0], box[1], box[2], box[3]))


class CutPath:
    """A letter outline flattened into widget space, addressable by distance along the cut."""

    def __init__(self, letter: Letter, transform: QTransform) -> None:
        self.points: list[QPointF] = []
        self.lengths: list[float] = []
        self._starts: set[int] = set()
        for part in letter.parts:
            for polygon in part.toSubpathPolygons(transform):
                self._append_polygon(polygon.toList())
        self.total = self.lengths[-1] if self.lengths else 0.0

    def _append_polygon(self, points: list[QPointF]) -> None:
        if not points:
            return
        if points[0] != points[-1]:
            points.append(QPointF(points[0]))
        distance = self.lengths[-1] if self.lengths else 0.0
        self._starts.add(len(self.points))
        previous = points[0]
        for point in points:
            distance += hypot(point.x() - previous.x(), point.y() - previous.y())
            self.points.append(point)
            self.lengths.append(distance)
            previous = point

    def position(self, distance: float) -> QPointF:
        """Return the point ``distance`` along the cut, clamped to the outline."""
        return self._at(distance, bisect_right(self.lengths, distance))

    def direction(self, distance: float) -> QPointF:
        """Return the unit tangent of the cut at ``distance``."""
        index = min(max(bisect_right(self.lengths, distance), 1), len(self.points) - 1)
        step = self.points[index] - self.points[index - 1]
        length = hypot(step.x(), step.y())
        return step / length if length > 0.0 else QPointF(1.0, 0.0)

    def _at(self, distance: float, index: int) -> QPointF:
        if index <= 0:
            return self.points[0]
        if index >= len(self.points):
            return self.points[-1]
        start, end = self.points[index - 1], self.points[index]
        span = self.lengths[index] - self.lengths[index - 1]
        if span <= 0.0:
            return end
        return start + (end - start) * ((distance - self.lengths[index - 1]) / span)

    def segment(self, start: float, end: float) -> list[QPolygonF]:
        """Return the polylines covering the cut between two distances, split per contour."""
        start, end = max(start, 0.0), min(end, self.total)
        if end <= start or not self.points:
            return []
        first = bisect_right(self.lengths, start)
        last = bisect_right(self.lengths, end)
        pieces: list[QPolygonF] = []
        current = [self._at(start, first)]
        for index in range(first, last):
            if index in self._starts:
                pieces.append(QPolygonF(current))
                current = []
            current.append(self.points[index])
        current.append(self._at(end, last))
        pieces.append(QPolygonF(current))
        return pieces


@dataclass(frozen=True)
class StageLayout:
    """Where the plate, the wordmark strip and the caption sit for one widget size."""

    plate: QRectF
    strip: QRectF
    caption_baseline: float
    letter_scale: float
    strip_transform: QTransform

    def stage_transform(self, letter: Letter) -> QTransform:
        """Map a letter's SVG units onto the plate, centred, at the shared letter scale."""
        return centred_transform(letter.bounds.center(), self.plate.center(), self.letter_scale)


def centred_transform(source_center: QPointF, target_center: QPointF, scale: float) -> QTransform:
    """Return the uniform transform placing ``source_center`` at ``target_center``."""
    transform = QTransform()
    transform.translate(target_center.x(), target_center.y())
    transform.scale(scale, scale)
    transform.translate(-source_center.x(), -source_center.y())
    return transform


def compute_layout(width: float, bottom: float, view_box: QRectF) -> StageLayout:
    """Lay the scene out in a ``width`` wide area whose usable height ends at ``bottom``."""
    margin = min(max(width * 0.06, 16.0), 48.0)
    ratio = view_box.width() / view_box.height()
    strip_height = min(max(width * 0.08, 22.0), 48.0)
    strip_width = min(width - 2.0 * margin, strip_height * ratio)
    strip_height = strip_width / ratio
    strip = QRectF(
        (width - strip_width) / 2.0, bottom - strip_height - 12.0, strip_width, strip_height
    )
    caption_baseline = strip.top() - 14.0
    plate_bottom = caption_baseline - 24.0
    available = max(plate_bottom - margin, 40.0)
    plate_width = min(width - 2.0 * margin, available / PLATE_ASPECT)
    plate_height = plate_width * PLATE_ASPECT
    plate = QRectF(
        (width - plate_width) / 2.0,
        margin + (available - plate_height) / 2.0,
        plate_width,
        plate_height,
    )
    inner = plate.adjusted(
        plate_width * PLATE_INSET,
        plate_height * PLATE_INSET,
        -plate_width * PLATE_INSET,
        -plate_height * PLATE_INSET,
    )
    scale = min(inner.width() / LETTER_FRAME.width(), inner.height() / LETTER_FRAME.height())
    strip_transform = QTransform()
    strip_transform.translate(strip.x(), strip.y())
    strip_transform.scale(strip_width / view_box.width(), strip_width / view_box.width())
    strip_transform.translate(-view_box.x(), -view_box.y())
    return StageLayout(plate, strip, caption_baseline, scale, strip_transform)


def blend(first: QColor, second: QColor, amount: float) -> QColor:
    """Mix ``amount`` (0..1) of ``second`` into ``first``, keeping the first's alpha."""
    return QColor(
        round(first.red() + (second.red() - first.red()) * amount),
        round(first.green() + (second.green() - first.green()) * amount),
        round(first.blue() + (second.blue() - first.blue()) * amount),
        first.alpha(),
    )


def with_alpha(color: QColor, alpha: int) -> QColor:
    """Return a copy of ``color`` with the given alpha."""
    copy = QColor(color)
    copy.setAlpha(alpha)
    return copy


@dataclass(frozen=True)
class LoaderColors:
    """Scene colours derived from the widget palette so both themes read correctly."""

    wall: QColor
    plate: QColor
    plate_edge: QColor
    kerf: QColor
    hot_core: QColor
    hot_mid: QColor
    hot_glow: QColor
    spark: QColor
    paint: QColor
    paint_light: QColor
    ghost: QColor
    caption: QColor
    light: bool

    @classmethod
    def from_palette(cls, palette: QPalette) -> "LoaderColors":
        """Pick the hot and paint tones that stay visible on this palette's plate colour."""
        wall = palette.color(QPalette.ColorRole.Window)
        plate = palette.color(QPalette.ColorRole.AlternateBase)
        text = palette.color(QPalette.ColorRole.WindowText)
        accent = palette.color(QPalette.ColorRole.Highlight)
        light = plate.lightness() >= 128
        return cls(
            wall=wall,
            plate=plate,
            plate_edge=blend(plate, text, 0.22),
            kerf=blend(plate, text, 0.5),
            hot_core=QColor("#ff8a00") if light else QColor("#fff4dc"),
            hot_mid=QColor("#ff4d00") if light else QColor("#ffb347"),
            hot_glow=QColor("#ff2d00") if light else QColor("#ff6a1a"),
            spark=QColor("#ff6a00") if light else QColor("#ffd166"),
            paint=accent,
            paint_light=blend(accent, QColor("#ffffff"), 0.35),
            ghost=palette.color(QPalette.ColorRole.Mid),
            caption=palette.color(QPalette.ColorRole.PlaceholderText),
            light=light,
        )
