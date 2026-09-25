"""Convert stencilizer glyph outlines into Qt paint primitives."""

from typing import Any

from fontTools.pens.qtPen import QtPen  # type: ignore[import-untyped]
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QTransform

from stencilizer.domain import Glyph, PointType


def draw_glyph(glyph: Glyph, pen: Any) -> None:
    """Draw a domain glyph into a fontTools-compatible pen."""
    for contour in glyph.contours:
        points = contour.points
        if not points:
            continue

        pen.moveTo(points[0].to_tuple())
        index = 1
        while index < len(points):
            point = points[index]
            if point.point_type == PointType.ON_CURVE:
                pen.lineTo(point.to_tuple())
                index += 1
            elif point.point_type == PointType.OFF_CURVE_QUAD:
                quad_points = [point.to_tuple()]
                index += 1
                while index < len(points):
                    next_point = points[index]
                    quad_points.append(next_point.to_tuple())
                    index += 1
                    if next_point.point_type != PointType.OFF_CURVE_QUAD:
                        break
                pen.qCurveTo(*quad_points)
            elif point.point_type == PointType.OFF_CURVE_CUBIC and index + 2 < len(points):
                pen.curveTo(
                    point.to_tuple(), points[index + 1].to_tuple(), points[index + 2].to_tuple()
                )
                index += 3
            else:
                index += 1
        pen.closePath()


def glyph_path(glyph: Glyph) -> QPainterPath:
    """Return a winding-filled Qt path for a glyph's contours."""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    draw_glyph(glyph, QtPen(None, path=path))
    return path


def glyph_frame(glyph: Glyph, ascender: int, descender: int) -> QRectF:
    """Return the font-unit frame containing the glyph and vertical metrics."""
    bounds = [contour.bounding_box() for contour in glyph.contours if contour.points]
    advance_width = glyph.metadata.advance_width
    if not bounds:
        return QRectF(0.0, float(descender), float(advance_width), float(ascender - descender))

    x0 = min(0.0, *(bound[0] for bound in bounds))
    x1 = max(float(advance_width), *(bound[2] for bound in bounds))
    y0 = min(float(descender), *(bound[1] for bound in bounds))
    y1 = max(float(ascender), *(bound[3] for bound in bounds))
    return QRectF(x0, y0, x1 - x0, y1 - y0)


def font_to_widget_transform(frame: QRectF, target: QRectF) -> QTransform:
    """Scale a y-up font frame uniformly into a y-down widget target."""
    if frame.width() == 0.0 or frame.height() == 0.0:
        return QTransform()

    scale = min(target.width() / frame.width(), target.height() / frame.height())
    transform = QTransform()
    transform.translate(target.center().x(), target.center().y())
    transform.scale(scale, -scale)
    transform.translate(-frame.center().x(), -frame.center().y())
    return transform


def render_glyph_image(
    glyph: Glyph, frame: QRectF, size: int, foreground: QColor, background: QColor
) -> QImage:
    """Rasterize a glyph into a square image with a two-pixel inset."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(background)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    target = QRectF(2.0, 2.0, float(size - 4), float(size - 4))
    painter.setTransform(font_to_widget_transform(frame, target))
    painter.fillPath(glyph_path(glyph), foreground)
    painter.end()
    return image
