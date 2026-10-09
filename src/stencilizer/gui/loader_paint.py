"""Frame painting for the loading view: a plate is cut, dropped, sprayed and lifted per letter."""

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QTransform,
)

from stencilizer.gui.loader_art import (
    Particles,
    band_sprite,
    clamp,
    ease_in_out,
    ease_in_quad,
    ease_out_cubic,
    glow_sprite,
    plate_pixmap_rect,
    render_plate,
    render_strip,
)
from stencilizer.gui.loader_effects import (
    paint_glow,
    paint_head,
    paint_mist,
    paint_sparks,
    paint_trail,
    render_ring,
)
from stencilizer.gui.loader_scene import (
    CutPath,
    LoaderColors,
    StageLayout,
    Wordmark,
    blend,
    centred_transform,
    compute_layout,
    load_wordmark,
    with_alpha,
)

PHASES: tuple[tuple[str, float], ...] = (
    ("plate", 0.25),
    ("cut", 1.2),
    ("drop", 0.4),
    ("spray", 0.65),
    ("lift", 0.25),
    ("fly", 0.35),
)
CYCLE_SECONDS = sum(duration for _, duration in PHASES)
CAPTIONS = {
    "plate": "LOADING PLATE",
    "cut": "CUTTING",
    "drop": "DROPPING THE SLUG",
    "spray": "SPRAYING",
    "lift": "LIFTING",
    "fly": "PLACING",
}
SPARK_CAP = 160
MIST_CAP = 120
MAX_STEP = 0.1
RING_WIDTH = 60.0
SPRAY_PAD = 70.0


@dataclass(frozen=True)
class FrameState:
    """Which letter is on the plate, which phase it is in and how far along (0..1)."""

    index: int
    phase: str
    u: float


def frame_state(time: float, letter_count: int) -> FrameState:
    """Map elapsed seconds onto the looping per-letter timeline."""
    cycle, offset = divmod(max(time, 0.0), CYCLE_SECONDS)
    index = int(cycle) % letter_count
    for name, duration in PHASES:
        if offset < duration:
            return FrameState(index, name, offset / duration)
        offset -= duration
    return FrameState(index, PHASES[-1][0], 1.0)


class Scene:
    """Everything the loader paints, with its caches and particle state; palette-aware."""

    def __init__(self, palette: QPalette, reduced_motion: bool) -> None:
        self.wordmark: Wordmark = load_wordmark()
        self.reduced_motion = reduced_motion
        self.colors = LoaderColors.from_palette(palette)
        self.sparks = Particles(cap=SPARK_CAP, gravity=1100.0, drag=1.2)
        self.mist = Particles(cap=MIST_CAP, gravity=60.0, drag=2.0)
        self._layout: StageLayout | None = None
        self._layout_key: tuple[float, float, float] | None = None
        self._ratio = 1.0
        self._cuts: list[CutPath] = []
        self._stage_paths: list[QPainterPath] = []
        self._rings: dict[int, tuple[QPixmap, QPointF]] = {}
        self._plate = QPixmap()
        self._strip = QPixmap()
        self._hot_glow = QPixmap()
        self._tail_glow = QPixmap()
        self._ambient_hot = QPixmap()
        self._ambient_paint = QPixmap()
        self._band = QPixmap()
        self._last_time: float | None = None

    @property
    def letter_count(self) -> int:
        """Number of letters the loop cycles through."""
        return len(self.wordmark.letters)

    def set_palette(self, palette: QPalette) -> None:
        """Re-derive the colours and drop the cached artwork."""
        self.colors = LoaderColors.from_palette(palette)
        self._layout_key = None

    def reset(self) -> None:
        """Forget particles and the previous frame time (a fresh start)."""
        self.sparks.clear()
        self.mist.clear()
        self._last_time = None

    def _ensure_layout(self, width: float, bottom: float, ratio: float) -> StageLayout:
        key = (width, bottom, ratio)
        if self._layout is None or self._layout_key != key:
            self._layout = compute_layout(width, bottom, self.wordmark.view_box)
            self._layout_key = key
            self._ratio = ratio
            self._rebuild(self._layout)
        return self._layout

    def _rebuild(self, layout: StageLayout) -> None:
        self._cuts = [
            CutPath(letter, layout.stage_transform(letter)) for letter in self.wordmark.letters
        ]
        self._stage_paths = [
            layout.stage_transform(letter).map(letter.path) for letter in self.wordmark.letters
        ]
        self._rings.clear()
        self._plate = render_plate(layout.plate, self.colors, self._ratio)
        self._strip = render_strip(
            self.wordmark, layout.strip, layout.strip_transform, self.colors, self._ratio
        )
        ratio, colors = self._ratio, self.colors
        ambient = layout.plate.width() * 0.9
        self._hot_glow = glow_sprite(colors.hot_glow, 36.0 + 60.0 * layout.letter_scale, ratio)
        self._tail_glow = glow_sprite(colors.hot_glow, 32.0, ratio, 0.55)
        self._ambient_hot = glow_sprite(colors.hot_glow, ambient, ratio, 0.2)
        self._ambient_paint = glow_sprite(colors.paint, ambient, ratio, 0.18)
        pad = layout.letter_scale * SPRAY_PAD
        self._band = band_sprite(colors.paint, pad, layout.plate.height() * 0.8, ratio, 0.9)

    def _ring(self, layout: StageLayout, index: int) -> tuple[QPixmap, QPointF]:
        ring = self._rings.get(index)
        if ring is None:
            ring = render_ring(
                self._stage_paths[index],
                self.colors.paint,
                layout.letter_scale * RING_WIDTH,
                self._ratio,
            )
            self._rings[index] = ring
        return ring

    def paint(self, painter: QPainter, rect: QRectF, bottom: float, time: float) -> FrameState:
        """Paint one frame at ``time`` seconds into the ``rect`` whose scene ends at ``bottom``."""
        layout = self._ensure_layout(rect.width(), bottom, painter.device().devicePixelRatio())
        state = frame_state(time, self.letter_count)
        dt = 0.0 if self._last_time is None else clamp(time - self._last_time, 0.0, MAX_STEP)
        self._last_time = time
        self.sparks.advance(dt)
        self.mist.advance(dt)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(rect, self.colors.wall)
        self._paint_strip(painter, layout, state)
        self._paint_plate(painter, layout, state)
        handler = {
            "cut": self._paint_cut,
            "drop": self._paint_drop,
            "spray": self._paint_spray,
            "lift": self._paint_lift,
            "fly": self._paint_fly,
        }.get(state.phase)
        if handler is not None:
            handler(painter, layout, state, dt)
        if self.sparks.items:
            paint_sparks(painter, self.colors, self.sparks)
        if self.mist.items:
            paint_mist(painter, self.colors, self.mist)
        self._paint_caption(painter, layout, state)
        return state

    def _paint_strip(self, painter: QPainter, layout: StageLayout, state: FrameState) -> None:
        painter.drawPixmap(layout.strip.topLeft() - QPointF(4.0, 4.0), self._strip)
        done, alpha = state.index, 255
        if state.index == 0 and state.phase == "plate":
            done, alpha = self.letter_count, int((1.0 - ease_in_out(state.u)) * 255)
        color = with_alpha(self.colors.paint, alpha)
        for letter in self.wordmark.letters[:done]:
            painter.fillPath(layout.strip_transform.map(letter.path), color)

    def _paint_plate(self, painter: QPainter, layout: StageLayout, state: FrameState) -> None:
        if state.phase == "fly":
            return
        fade, scale = 0.0, 1.0
        if state.phase == "plate":
            self._ring(layout, state.index)  # render the halo now, while frames are cheap
            eased = ease_out_cubic(state.u)
            fade, scale = 1.0 - eased, 1.0 + 0.04 * (1.0 - eased)
        elif state.phase == "lift":
            eased = ease_in_out(state.u)
            fade, scale = eased, 1.0 + 0.06 * eased
        if self.reduced_motion:
            scale = 1.0
        painter.save()
        if scale != 1.0:
            center = layout.plate.center()
            painter.translate(center)
            painter.scale(scale, scale)
            painter.translate(-center)
        painter.drawPixmap(plate_pixmap_rect(layout.plate).topLeft(), self._plate)
        if state.phase == "lift":
            self._paint_ring(painter, layout, state.index, 1.0)
            self._paint_hole(painter, state.index)
        if fade > 0.0:
            cover = with_alpha(self.colors.wall, int(fade * 255))
            painter.fillRect(plate_pixmap_rect(layout.plate), cover)
        painter.restore()

    def _paint_hole(self, painter: QPainter, index: int) -> None:
        path = self._stage_paths[index]
        painter.fillPath(path, self.colors.wall)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(blend(self.colors.wall, self.colors.plate_edge, 0.6), 1.0))
        painter.drawPath(path)

    def _paint_ring(
        self, painter: QPainter, layout: StageLayout, index: int, amount: float
    ) -> None:
        pixmap, origin = self._ring(layout, index)
        painter.setOpacity(amount)
        painter.drawPixmap(origin, pixmap)
        painter.setOpacity(1.0)

    def _paint_cut(
        self, painter: QPainter, layout: StageLayout, state: FrameState, dt: float
    ) -> None:
        cut = self._cuts[state.index]
        distance = state.u * cut.total
        head = cut.position(distance)
        paint_glow(painter, self._ambient_hot, head)
        paint_trail(painter, self.colors, cut, distance, 1.0, self._tail_glow)
        paint_glow(painter, self._hot_glow, head)
        paint_head(painter, self.colors, head)
        if not self.reduced_motion:
            tangent = cut.direction(distance)
            self.sparks.emit(
                dt,
                rate=260.0,
                origin=head,
                spread=QPointF(2.0, 2.0),
                velocity=QPointF(-tangent.x() * 150.0, -tangent.y() * 150.0 - 60.0),
                jitter=280.0 * max(layout.letter_scale, 0.3),
                life=(0.2, 0.55),
            )

    def _paint_drop(
        self, painter: QPainter, layout: StageLayout, state: FrameState, _dt: float
    ) -> None:
        cut = self._cuts[state.index]
        self._paint_hole(painter, state.index)
        paint_trail(
            painter, self.colors, cut, cut.total, 1.0 - state.u, self._tail_glow, kerf=False
        )
        fall = 0.0 if self.reduced_motion else ease_in_quad(state.u)
        alpha = int((1.0 - (state.u if self.reduced_motion else fall * 0.6)) * 255)
        painter.save()
        painter.setClipRect(layout.plate)
        center = self._stage_paths[state.index].boundingRect().center()
        painter.translate(0.0, fall * layout.plate.height() * 1.1)
        painter.translate(center)
        painter.rotate(fall * 7.0)
        painter.translate(-center)
        painter.setPen(QPen(with_alpha(self.colors.plate_edge, alpha), 1.0))
        painter.setBrush(with_alpha(blend(self.colors.plate, self.colors.wall, 0.2), alpha))
        painter.drawPath(self._stage_paths[state.index])
        painter.restore()

    def _paint_spray(
        self, painter: QPainter, layout: StageLayout, state: FrameState, dt: float
    ) -> None:
        path = self._stage_paths[state.index]
        bounds = path.boundingRect()
        pad = layout.letter_scale * SPRAY_PAD
        eased = ease_in_out(state.u)
        front = bounds.left() - pad + (bounds.width() + 2.0 * pad) * eased
        nozzle = QPointF(front, bounds.center().y())
        paint_glow(painter, self._ambient_paint, nozzle)
        self._paint_ring(painter, layout, state.index, eased)
        self._paint_hole(painter, state.index)
        feather = max(bounds.width() * 0.18, 18.0)
        gradient = QLinearGradient(QPointF(front - feather, 0.0), QPointF(front, 0.0))
        gradient.setColorAt(0.0, self.colors.paint)
        gradient.setColorAt(0.65, with_alpha(self.colors.paint_light, 200))
        gradient.setColorAt(1.0, with_alpha(self.colors.paint_light, 0))
        painter.fillPath(path, gradient)
        band_height = self._band.height() / self._ratio
        painter.drawPixmap(QPointF(front - pad / 2.0, nozzle.y() - band_height / 2.0), self._band)
        if not self.reduced_motion:
            self.mist.emit(
                dt,
                rate=200.0,
                origin=nozzle,
                spread=QPointF(pad * 0.4, bounds.height() + pad),
                velocity=QPointF(110.0, 10.0),
                jitter=80.0,
                life=(0.25, 0.6),
            )

    def _paint_lift(
        self, painter: QPainter, _layout: StageLayout, state: FrameState, _dt: float
    ) -> None:
        painter.fillPath(self._stage_paths[state.index], self.colors.paint)

    def _paint_fly(
        self, painter: QPainter, layout: StageLayout, state: FrameState, _dt: float
    ) -> None:
        letter = self.wordmark.letters[state.index]
        eased = ease_in_out(state.u)
        if self.reduced_motion:
            fading = with_alpha(self.colors.paint, int((1.0 - eased) * 255))
            painter.fillPath(self._stage_paths[state.index], fading)
            rising = with_alpha(self.colors.paint, int(eased * 255))
            painter.fillPath(layout.strip_transform.map(letter.path), rising)
            return
        source = letter.bounds.center()
        start, end = layout.plate.center(), layout.strip_transform.map(source)
        target = start + (end - start) * eased
        scale = layout.letter_scale + (layout.strip_transform.m11() - layout.letter_scale) * eased
        transform: QTransform = centred_transform(source, target, scale)
        painter.fillPath(transform.map(letter.path), self.colors.paint)

    def _paint_caption(self, painter: QPainter, layout: StageLayout, state: FrameState) -> None:
        text = f"{CAPTIONS[state.phase]}   {state.index + 1:02d} / {self.letter_count:02d}"
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118.0)
        painter.setFont(font)
        painter.setPen(self.colors.caption)
        width = painter.fontMetrics().horizontalAdvance(text)
        x = layout.strip.center().x() - width / 2.0
        painter.drawText(QPointF(x, layout.caption_baseline), text)
        painter.setPen(QPen(with_alpha(self.colors.ghost, 200), 1.0))
        y = layout.caption_baseline - 4.0
        gap = 14.0
        painter.drawLine(QPointF(layout.strip.left(), y), QPointF(x - gap, y))
        painter.drawLine(QPointF(x + width + gap, y), QPointF(layout.strip.right(), y))
