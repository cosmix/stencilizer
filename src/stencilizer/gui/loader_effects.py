"""Cutter, spray and particle drawing for the loading view."""

from collections import defaultdict

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from stencilizer.gui.loader_art import Particles, clamp
from stencilizer.gui.loader_scene import CutPath, LoaderColors, with_alpha

TAIL_FRACTION = 0.14
TAIL_MIN = 40.0
TAIL_BLITS = 12
TAIL_BLIT_MAX = 30.0
TAIL_BLIT_MIN = 8.0
SPARK_STREAK = 0.03
RING_STEPS = 7


def alpha_pen(color: QColor, alpha: float, width: float) -> QPen:
    """Round-capped pen of ``color`` at ``alpha`` (0..1)."""
    pen = QPen(with_alpha(color, int(clamp(alpha) * 255)), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def paint_trail(
    painter: QPainter,
    colors: LoaderColors,
    cut: CutPath,
    distance: float,
    heat: float,
    glow: QPixmap,
    kerf: bool = True,
) -> None:
    """Draw the scored kerf up to ``distance`` with a glowing tail scaled by ``heat`` (0..1).

    The tail's halo is a row of small sprite blits rather than a wide antialiased stroke,
    which costs a fraction of the time at 2x. ``kerf=False`` skips the scored line when the
    hole's own edge already shows it.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if kerf:
        painter.setPen(QPen(colors.kerf, 1.4))
        for piece in cut.segment(0.0, distance):
            painter.drawPolyline(piece)
    if heat <= 0.0:
        return
    span = max(cut.total * TAIL_FRACTION, TAIL_MIN)
    source = QRectF(glow.rect())
    for step in range(TAIL_BLITS):
        fraction = step / (TAIL_BLITS - 1)
        size = (TAIL_BLIT_MAX - (TAIL_BLIT_MAX - TAIL_BLIT_MIN) * fraction) * heat
        center = cut.position(distance - span * fraction)
        target = QRectF(center.x() - size / 2.0, center.y() - size / 2.0, size, size)
        painter.drawPixmap(target, glow, source)
    near = distance - span * 0.4
    painter.setPen(alpha_pen(colors.hot_mid, 0.5 * heat, 3.0))
    for piece in cut.segment(distance - span, near):
        painter.drawPolyline(piece)
    painter.setPen(alpha_pen(colors.hot_mid, 0.9 * heat, 3.6))
    for piece in cut.segment(near, distance):
        painter.drawPolyline(piece)
    painter.setPen(alpha_pen(colors.hot_core, heat, 1.5))
    for piece in cut.segment(near, distance):
        painter.drawPolyline(piece)


def paint_glow(painter: QPainter, sprite: QPixmap, center: QPointF) -> None:
    """Blit a glow sprite at its own size, centred on ``center``."""
    size = sprite.width() / sprite.devicePixelRatio()
    painter.drawPixmap(QPointF(center.x() - size / 2.0, center.y() - size / 2.0), sprite)


def paint_head(painter: QPainter, colors: LoaderColors, position: QPointF) -> None:
    """Draw the white-hot cutter tip."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(with_alpha(colors.hot_mid, 150))
    painter.drawEllipse(position, 5.0, 5.0)
    painter.setBrush(colors.hot_core)
    painter.drawEllipse(position, 2.6, 2.6)


def paint_sparks(painter: QPainter, colors: LoaderColors, sparks: Particles) -> None:
    """Draw sparks as velocity streaks, batched by tone and brightness."""
    tones = (colors.hot_core, colors.spark, colors.hot_glow)
    buckets: dict[tuple[int, int], list[QLineF]] = defaultdict(list)
    for spark in sparks.items:
        age = spark.age / spark.life
        tone = 0 if age < 0.3 else 1 if age < 0.65 else 2
        level = int((1.0 - age) * 3.99)
        buckets[tone, level].append(
            QLineF(
                spark.x,
                spark.y,
                spark.x - spark.vx * SPARK_STREAK,
                spark.y - spark.vy * SPARK_STREAK,
            )
        )
    for (tone, level), lines in buckets.items():
        painter.setPen(alpha_pen(tones[tone], (level + 1) / 4.0, 1.6))
        painter.drawLines(lines)


def paint_mist(painter: QPainter, colors: LoaderColors, mist: Particles) -> None:
    """Draw paint droplets as round dots, batched by brightness."""
    buckets: dict[int, list[QPointF]] = defaultdict(list)
    for drop in mist.items:
        level = int((1.0 - drop.age / drop.life) * 3.99)
        buckets[level].append(QPointF(drop.x, drop.y))
    for level, points in buckets.items():
        painter.setPen(alpha_pen(colors.paint, 0.6 * (level + 1) / 4.0, 2.0 + level * 0.4))
        painter.drawPoints(QPolygonF(points))


def render_ring(
    path: QPainterPath, color: QColor, width: float, ratio: float
) -> tuple[QPixmap, QPointF]:
    """Pre-render the soft overspray halo around ``path`` and return it with its origin."""
    bounds = path.boundingRect().adjusted(-width, -width, width, width)
    pixmap = QPixmap(max(round(bounds.width() * ratio), 1), max(round(bounds.height() * ratio), 1))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(-bounds.topLeft())
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for step in range(RING_STEPS):
        fraction = 1.0 - step / RING_STEPS
        painter.setPen(alpha_pen(color, 0.05 + 0.03 * step, width * fraction))
        painter.drawPath(path)
    painter.end()
    return pixmap, bounds.topLeft()
