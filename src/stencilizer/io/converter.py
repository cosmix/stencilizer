"""Converters between fonttools and domain models.

This module handles the conversion between fonttools representations
and our domain models (Glyph, Contour, Point).
"""

from typing import Any

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata


def fonttools_glyph_to_domain(
    name: str,
    fonttools_glyph: Any,
    font: TTFont
) -> Glyph:
    """Convert fonttools glyph to domain Glyph model.

    Handles both TrueType (quadratic curves) and OpenType/CFF (cubic curves).
    Uses a RecordingPen to extract the glyph outline as a series of
    drawing commands, then converts these to Contour objects.

    Args:
        name: Name of the glyph
        fonttools_glyph: The fonttools glyph object from GlyphSet
        font: The TTFont object for accessing metadata

    Returns:
        Domain Glyph model

    Raises:
        Exception: If glyph cannot be converted
    """
    pen = RecordingPen()
    fonttools_glyph.draw(pen)

    contours = _recording_to_contours(pen.value)

    metadata = _extract_glyph_metadata(name, font)

    glyph = Glyph(metadata=metadata, contours=contours)

    if hasattr(fonttools_glyph, "_glyph"):
        raw_glyph = fonttools_glyph._glyph
        is_composite = hasattr(raw_glyph, "isComposite") and raw_glyph.isComposite()
        glyph._is_composite = is_composite
    else:
        glyph._is_composite = False

    return glyph


def domain_glyph_to_fonttools(
    glyph: Glyph,
    original_glyph: Any,
    font: TTFont
) -> None:
    """Update fonttools glyph from domain model.

    Converts domain Glyph back to fonttools representation and updates
    the glyph in place. Handles both TrueType and OpenType/CFF formats.

    Args:
        glyph: Domain glyph model with modifications
        original_glyph: Original fonttools glyph object to update
        font: The TTFont object

    Raises:
        NotImplementedError: If glyph format is not supported
    """
    is_truetype = "glyf" in font

    if is_truetype:
        _update_truetype_glyph(glyph, original_glyph, font)
    elif "CFF " in font:
        _update_cff_glyph(glyph, original_glyph, font)
    else:
        raise NotImplementedError("Unsupported font format")


def _recording_to_contours(recording: list[tuple[str, tuple[Any, ...]]]) -> list[Contour]:
    """Convert RecordingPen recording to list of Contour objects.

    The RecordingPen records drawing commands like:
    - ('moveTo', ((x, y),))
    - ('lineTo', ((x, y),))
    - ('qCurveTo', ((x1, y1), (x2, y2), ...))  # Quadratic
    - ('curveTo', ((x1, y1), (x2, y2), (x3, y3)))  # Cubic
    - ('closePath', ())

    Args:
        recording: List of drawing commands from RecordingPen

    Returns:
        List of Contour objects
    """
    contours: list[Contour] = []
    current_points: list[Point] = []

    for command, args in recording:
        if command == "moveTo":
            if current_points:
                contours.append(Contour(points=current_points))
                current_points = []

            x, y = args[0]
            current_points.append(Point(x, y, PointType.ON_CURVE))

        elif command == "lineTo":
            x, y = args[0]
            current_points.append(Point(x, y, PointType.ON_CURVE))

        elif command == "qCurveTo":
            for i, (x, y) in enumerate(args):
                if i < len(args) - 1:
                    current_points.append(Point(x, y, PointType.OFF_CURVE_QUAD))
                else:
                    current_points.append(Point(x, y, PointType.ON_CURVE))

        elif command == "curveTo":
            x1, y1 = args[0]
            x2, y2 = args[1]
            x3, y3 = args[2]
            current_points.append(Point(x1, y1, PointType.OFF_CURVE_CUBIC))
            current_points.append(Point(x2, y2, PointType.OFF_CURVE_CUBIC))
            current_points.append(Point(x3, y3, PointType.ON_CURVE))

        elif command == "closePath" or command == "endPath":
            if current_points:
                contours.append(Contour(points=current_points))
                current_points = []

    if current_points:
        contours.append(Contour(points=current_points))

    return contours


def _extract_glyph_metadata(name: str, font: TTFont) -> GlyphMetadata:
    """Extract glyph metadata from font.

    Args:
        name: Glyph name
        font: The TTFont object

    Returns:
        GlyphMetadata object
    """
    hmtx = font.get("hmtx")
    advance_width = 0
    lsb = 0

    if hmtx and name in hmtx.metrics:
        advance_width, lsb = hmtx.metrics[name]

    cmap = font.getBestCmap()
    unicode_value = None

    if cmap:
        for code_point, glyph_name in cmap.items():
            if glyph_name == name:
                unicode_value = code_point
                break

    return GlyphMetadata(
        name=name,
        unicode=unicode_value,
        advance_width=advance_width,
        left_side_bearing=lsb
    )


def _update_truetype_glyph(glyph: Glyph, original_glyph: Any, font: TTFont) -> None:
    """Update TrueType glyph from domain model.

    Args:
        glyph: Domain glyph model
        original_glyph: Original fonttools glyph
        font: The TTFont object
    """
    glyf_table = font["glyf"]
    glyph_name = glyph.name

    pen = TTGlyphPen(font.getGlyphSet())

    for contour in glyph.contours:
        if not contour.points:
            continue

        first_point = contour.points[0]
        pen.moveTo((first_point.x, first_point.y))

        i = 1
        while i < len(contour.points):
            point = contour.points[i]

            if point.point_type == PointType.ON_CURVE:
                pen.lineTo((point.x, point.y))
                i += 1

            elif point.point_type == PointType.OFF_CURVE_QUAD:
                quad_points = [(point.x, point.y)]
                i += 1

                while i < len(contour.points):
                    next_point = contour.points[i]
                    if next_point.point_type == PointType.OFF_CURVE_QUAD:
                        quad_points.append((next_point.x, next_point.y))
                        i += 1
                    else:
                        quad_points.append((next_point.x, next_point.y))
                        i += 1
                        break

                pen.qCurveTo(*quad_points)

            else:
                i += 1

        pen.closePath()

    new_glyph = pen.glyph()
    glyf_table[glyph_name] = new_glyph


def _update_cff_glyph(glyph: Glyph, original_glyph: Any, font: TTFont) -> None:
    """Update CFF/OpenType glyph from domain model.

    Args:
        glyph: Domain glyph model
        original_glyph: Original fonttools glyph
        font: The TTFont object
    """
    cff_table = font["CFF "]
    top_dict = cff_table.cff.topDictIndex[0]
    charstrings = top_dict.CharStrings
    glyph_name = glyph.name

    pen = T2CharStringPen(width=glyph.metadata.advance_width, glyphSet=font.getGlyphSet())

    for contour in glyph.contours:
        if not contour.points:
            continue

        first_point = contour.points[0]
        pen.moveTo((first_point.x, first_point.y))

        i = 1
        while i < len(contour.points):
            point = contour.points[i]

            if point.point_type == PointType.ON_CURVE:
                pen.lineTo((point.x, point.y))
                i += 1

            elif point.point_type == PointType.OFF_CURVE_CUBIC:
                if i + 2 < len(contour.points):
                    p1 = point
                    p2 = contour.points[i + 1]
                    p3 = contour.points[i + 2]

                    pen.curveTo(
                        (p1.x, p1.y),
                        (p2.x, p2.y),
                        (p3.x, p3.y)
                    )
                    i += 3
                else:
                    i += 1

            else:
                i += 1

        pen.closePath()

    charstring = pen.getCharString()
    charstrings[glyph_name] = charstring
