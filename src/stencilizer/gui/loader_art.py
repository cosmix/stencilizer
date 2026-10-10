"""Cached artwork and particle effects for the loading view."""

from dataclasses import dataclass, field
from math import cos, sin
from random import Random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QTransform,
)

from stencilizer.gui.loader_scene import LoaderColors, Wordmark, blend, with_alpha

PLATE_SHADOW = 18.0
GLOW_STOPS = ((0.0, 235), (0.18, 170), (0.4, 80), (0.7, 20), (1.0, 0))


def ease_out_cubic(u: float) -> float:
    """Decelerate: fast start, soft landing."""
    return 1.0 - (1.0 - u) ** 3


def ease_in_quad(u: float) -> float:
    """Accelerate: gentle start, like a dropped piece."""
    return u * u


def ease_in_out(u: float) -> float:
    """Smoothstep between 0 and 1."""
    return u * u * (3.0 - 2.0 * u)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound ``value`` to ``[low, high]``."""
    return low if value < low else high if value > high else value


def glow_sprite(color: QColor, size: float, ratio: float, strength: float = 1.0) -> QPixmap:
    """Render a soft radial glow of ``color`` into a square pixmap at the given pixel ratio.

    ``strength`` scales the alpha so the sprite can be blitted without a painter opacity,
    which keeps large blits on the fast path.
    """
    pixels = max(round(size * ratio), 2)
    pixmap = QPixmap(pixels, pixels)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    gradient = QRadialGradient(QPointF(size / 2.0, size / 2.0), size / 2.0)
    for position, alpha in GLOW_STOPS:
        gradient.setColorAt(position, with_alpha(color, round(alpha * strength)))
    painter = QPainter(pixmap)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRect(QRectF(0.0, 0.0, float(size), float(size)))
    painter.end()
    return pixmap


def band_sprite(
    color: QColor, width: float, height: float, ratio: float, strength: float
) -> QPixmap:
    """Render a vertical mist band (a glow stretched to ``width`` x ``height``)."""
    pixmap = QPixmap(max(round(width * ratio), 2), max(round(height * ratio), 2))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    source = glow_sprite(color, 64.0, 1.0, strength)
    painter.drawPixmap(QRectF(0.0, 0.0, width, height), source, QRectF(source.rect()))
    painter.end()
    return pixmap


def _paint_shadow(painter: QPainter, rect: QRectF, radius: float, dark: QColor) -> None:
    for step in range(6):
        spread = PLATE_SHADOW * (step + 1) / 6.0
        painter.setBrush(with_alpha(dark, 14 - step * 2))
        painter.drawRoundedRect(
            rect.adjusted(-spread, -spread * 0.6, spread, spread * 1.3),
            radius + spread,
            radius + spread,
        )


def _paint_brushing(painter: QPainter, rect: QRectF, colors: LoaderColors) -> None:
    rng = Random(7)
    lighter = blend(colors.plate, QColor("#ffffff"), 0.5 if colors.light else 0.9)
    darker = blend(colors.plate, QColor("#000000"), 0.5)
    for _ in range(int(rect.height() * 0.9)):
        y = rect.top() + rng.random() * rect.height()
        x0 = rect.left() + rng.random() * rect.width() * 0.6
        x1 = min(rect.right(), x0 + rect.width() * (0.15 + rng.random() * 0.6))
        alpha = 3 + int(rng.random() * 7)
        color = lighter if rng.random() < 0.5 else darker
        painter.setPen(QPen(with_alpha(color, alpha), 1.0))
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))


def _paint_rivets(painter: QPainter, rect: QRectF, colors: LoaderColors) -> None:
    inset = min(rect.width(), rect.height()) * 0.055
    radius = max(inset * 0.28, 2.5)
    hole = blend(colors.plate, colors.wall, 0.75)
    for x in (rect.left() + inset, rect.right() - inset):
        for y in (rect.top() + inset, rect.bottom() - inset):
            painter.setPen(QPen(with_alpha(colors.plate_edge, 170), 1.0))
            painter.setBrush(hole)
            painter.drawEllipse(QPointF(x, y), radius, radius)


def render_plate(rect: QRectF, colors: LoaderColors, ratio: float) -> QPixmap:
    """Draw the sheet-metal plate with its shadow into a pixmap; the plate origin is at offset."""
    size = plate_pixmap_rect(rect)
    pixmap = QPixmap(max(round(size.width() * ratio), 1), max(round(size.height() * ratio), 1))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(-size.topLeft())
    radius = min(rect.width(), rect.height()) * 0.035
    painter.setPen(Qt.PenStyle.NoPen)
    _paint_shadow(painter, rect, radius, QColor("#000000"))
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    tint = QColor("#ffffff") if colors.light else blend(colors.plate, QColor("#ffffff"), 0.08)
    gradient.setColorAt(0.0, tint)
    gradient.setColorAt(0.55, colors.plate)
    gradient.setColorAt(1.0, blend(colors.plate, QColor("#000000"), 0.06 if colors.light else 0.25))
    painter.setBrush(gradient)
    painter.drawRoundedRect(rect, radius, radius)
    painter.setClipPath(_rounded(rect, radius))
    _paint_brushing(painter, rect, colors)
    painter.setClipping(False)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(colors.plate_edge, 1.0))
    painter.drawRoundedRect(rect, radius, radius)
    _paint_rivets(painter, rect, colors)
    painter.end()
    return pixmap


def plate_pixmap_rect(rect: QRectF) -> QRectF:
    """Return the area a ``render_plate`` pixmap covers: the plate plus its shadow."""
    return rect.adjusted(-PLATE_SHADOW * 1.2, -PLATE_SHADOW, PLATE_SHADOW * 1.2, PLATE_SHADOW * 1.6)


def _rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def render_strip(
    wordmark: Wordmark, strip: QRectF, transform: QTransform, colors: LoaderColors, ratio: float
) -> QPixmap:
    """Draw the ghost wordmark (thin outlines of every letter) into a pixmap covering ``strip``."""
    pad = 4.0
    size = strip.adjusted(-pad, -pad, pad, pad)
    pixmap = QPixmap(max(round(size.width() * ratio), 1), max(round(size.height() * ratio), 1))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(-size.topLeft())
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(with_alpha(colors.ghost, 255), 1.2))
    for letter in wordmark.letters:
        painter.drawPath(transform.map(letter.path))
    painter.end()
    return pixmap


@dataclass
class Particle:
    """One spark or paint droplet: position, velocity and life in seconds."""

    x: float
    y: float
    vx: float
    vy: float
    age: float
    life: float


@dataclass
class Particles:
    """A bounded pool of particles advanced by wall time; emission stops at the cap."""

    cap: int
    gravity: float
    drag: float
    rng: Random = field(default_factory=lambda: Random(11))
    items: list[Particle] = field(default_factory=list)
    _debt: float = 0.0

    def clear(self) -> None:
        """Drop every particle and pending emission."""
        self.items.clear()
        self._debt = 0.0

    def advance(self, dt: float) -> None:
        """Move every particle by ``dt`` seconds and retire the ones past their life."""
        keep: list[Particle] = []
        damping = max(0.0, 1.0 - self.drag * dt)
        for particle in self.items:
            particle.age += dt
            if particle.age >= particle.life:
                continue
            particle.vy += self.gravity * dt
            particle.vx *= damping
            particle.vy *= damping
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            keep.append(particle)
        self.items = keep

    def emit(
        self,
        dt: float,
        rate: float,
        origin: QPointF,
        spread: QPointF,
        velocity: QPointF,
        jitter: float,
        life: tuple[float, float],
    ) -> None:
        """Spawn ``rate`` particles per second around ``origin`` while under the cap."""
        self._debt += rate * dt
        rng = self.rng
        while self._debt >= 1.0:
            self._debt -= 1.0
            if len(self.items) >= self.cap:
                continue
            angle = rng.random() * 6.283185
            speed = rng.random() * jitter
            self.items.append(
                Particle(
                    x=origin.x() + (rng.random() - 0.5) * spread.x(),
                    y=origin.y() + (rng.random() - 0.5) * spread.y(),
                    vx=velocity.x() + speed * cos(angle),
                    vy=velocity.y() + speed * sin(angle),
                    age=0.0,
                    life=life[0] + rng.random() * (life[1] - life[0]),
                )
            )
