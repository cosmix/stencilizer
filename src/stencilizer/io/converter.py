"""Converters between fonttools and domain models.

This module handles the conversion between fonttools representations
and our domain models (Glyph, Contour, Point).
"""

from typing import Any

from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.pens.t2CharStringPen import T2CharStringPen  # type: ignore[import-untyped]
from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.exceptions import GlyphProcessingError


def fonttools_glyph_to_domain(
    name: str,
    fonttools_glyph: Any,
    font: TTFont,
    unicode_by_name: dict[str, int] | None = None,
) -> Glyph:
    """Convert recorded outlines and metadata, normalizing CFF winding to TrueType."""
    pen = RecordingPen()
    fonttools_glyph.draw(pen)

    contours = _recording_to_contours(pen.value)

    # CFF fonts use opposite winding convention from TrueType.
    # Reverse contour points to normalize to TrueType convention.
    is_cff = "CFF " in font or "CFF2" in font
    if is_cff:
        for contour in contours:
            contour.points = list(reversed(contour.points))

    metadata = _extract_glyph_metadata(name, font, unicode_by_name)

    glyph = Glyph(metadata=metadata, contours=contours)

    if hasattr(fonttools_glyph, "_glyph"):
        raw_glyph = fonttools_glyph._glyph
        is_composite = hasattr(raw_glyph, "isComposite") and raw_glyph.isComposite()
        glyph._is_composite = is_composite
    else:
        glyph._is_composite = False

    return glyph


def domain_glyph_to_fonttools(glyph: Glyph, original_glyph: Any, font: TTFont) -> None:
    """Update fonttools glyph from domain model.

    Converts domain Glyph back to fonttools representation and updates
    the glyph in place. Handles both TrueType and OpenType/CFF formats.

    Args:
        glyph: Domain glyph model with modifications
        original_glyph: Original fonttools glyph object to update
        font: The TTFont object

    Raises:
        GlyphProcessingError: If the font has no glyf, CFF or CFF2 table
    """
    is_truetype = "glyf" in font

    if is_truetype:
        _update_truetype_glyph(glyph, original_glyph, font)
    elif "CFF " in font:
        _update_cff_glyph(glyph, original_glyph, font)
    elif "CFF2" in font:
        _update_cff2_glyph(glyph, original_glyph, font)
    else:
        raise GlyphProcessingError(glyph.name, "font has no glyf, CFF or CFF2 table")


def _recording_to_contours(recording: list[tuple[str, tuple[Any, ...]]]) -> list[Contour]:
    """Convert RecordingPen commands to contours, preserving curve controls."""
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
            _append_quadratic_points(current_points, args)

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


def _append_quadratic_points(points: list[Point], args: tuple[Any, ...]) -> None:
    all_off_curve = args[-1] is None
    positions = args[:-1] if all_off_curve else args
    if all_off_curve and not points:
        first_x, first_y = positions[0]
        last_x, last_y = positions[-1]
        points.append(Point((first_x + last_x) / 2, (first_y + last_y) / 2))
    for index, (x, y) in enumerate(positions):
        point_type = (
            PointType.OFF_CURVE_QUAD
            if all_off_curve or index < len(positions) - 1
            else PointType.ON_CURVE
        )
        points.append(Point(x, y, point_type))


def build_unicode_by_name(font: TTFont) -> dict[str, int]:
    """Map each glyph name to its lowest-listed code point in the best cmap."""
    mapping: dict[str, int] = {}
    for code_point, glyph_name in (font.getBestCmap() or {}).items():
        mapping.setdefault(glyph_name, code_point)
    return mapping


def _extract_glyph_metadata(
    name: str, font: TTFont, unicode_by_name: dict[str, int] | None = None
) -> GlyphMetadata:
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

    if unicode_by_name is None:
        unicode_by_name = build_unicode_by_name(font)
    unicode_value = unicode_by_name.get(name)

    return GlyphMetadata(
        name=name, unicode=unicode_value, advance_width=advance_width, left_side_bearing=lsb
    )


def _update_truetype_glyph(glyph: Glyph, _: Any, font: TTFont) -> None:
    """Update TrueType glyph from domain model.

    Args:
        glyph: Domain glyph model
        _: Original fonttools glyph (unused)
        font: The TTFont object
    """
    glyf_table = font["glyf"]
    glyph_name = glyph.name

    pen = TTGlyphPen(font.getGlyphSet())

    for contour in glyph.contours:
        _draw_closed_contour(pen, contour.points, PointType.OFF_CURVE_QUAD)

    new_glyph = pen.glyph()
    glyf_table[glyph_name] = new_glyph


def _cyclic_points(points: list[Point], control_type: PointType) -> list[Point]:
    """Start a closed contour at an on-curve point without changing its edges."""
    if not points:
        return []
    for index, point in enumerate(points):
        if point.point_type == PointType.ON_CURVE:
            return points[index:] + points[:index]
    if control_type != PointType.OFF_CURVE_QUAD:
        raise ValueError("Cubic contour has no on-curve point")
    first, last = points[0], points[-1]
    implied = Point((first.x + last.x) / 2, (first.y + last.y) / 2)
    return [implied, *points]


def _draw_closed_contour(pen: Any, points: list[Point], control_type: PointType) -> None:
    ordered = _cyclic_points(points, control_type)
    if not ordered:
        return
    first = ordered[0]
    pen.moveTo((first.x, first.y))
    controls: list[tuple[float, float]] = []
    for point in ordered[1:]:
        position = (point.x, point.y)
        if point.point_type == control_type:
            controls.append(position)
        elif point.point_type == PointType.ON_CURVE:
            _emit_segment(pen, controls, position, control_type)
            controls = []
        else:
            raise ValueError(f"Unexpected {point.point_type.value} control in contour")
    if controls:
        _emit_segment(pen, controls, (first.x, first.y), control_type)
    pen.closePath()


def _emit_segment(
    pen: Any,
    controls: list[tuple[float, float]],
    endpoint: tuple[float, float],
    control_type: PointType,
) -> None:
    if not controls:
        pen.lineTo(endpoint)
    elif control_type == PointType.OFF_CURVE_QUAD:
        pen.qCurveTo(*controls, endpoint)
    elif len(controls) == 2:
        pen.curveTo(*controls, endpoint)
    else:
        raise ValueError("Cubic segment requires exactly two control points")


def _store_cff_charstring(
    glyph: Glyph, pen: T2CharStringPen, charstrings: Any, private: Any, global_subrs: Any
) -> None:
    """Draw ``glyph`` on ``pen`` and store the compiled charstring under its name.

    Domain contours use TrueType winding (normalized on read), so each contour's points
    are reversed to restore the CFF winding convention.
    """
    for contour in glyph.contours:
        _draw_closed_contour(pen, list(reversed(contour.points)), PointType.OFF_CURVE_CUBIC)

    charstring = pen.getCharString(private=private, globalSubrs=global_subrs, optimize=False)
    charstrings[glyph.name] = charstring


def _fd_private_dict(top_dict: Any, charstrings: Any, name: str) -> Any:
    """Private dict of the font dict (FDArray entry) that governs glyph ``name``.

    A font without FDSelect reports no selector and uses FDArray[0].
    """
    _, fd_index = charstrings.getItemAndSelector(name)
    return top_dict.FDArray[fd_index or 0].Private


def _update_cff_glyph(glyph: Glyph, _: Any, font: TTFont) -> None:
    """Update CFF/OpenType glyph from domain model.

    Args:
        glyph: Domain glyph model
        _: Original fonttools glyph (unused)
        font: The TTFont object
    """
    cff_table = font["CFF "]
    top_dict = cff_table.cff.topDictIndex[0]
    charstrings = top_dict.CharStrings
    private = getattr(top_dict, "Private", None)
    if private is None:
        # CID-keyed fonts have no top-level Private; the glyph's FDSelect entry names it.
        private = _fd_private_dict(top_dict, charstrings, glyph.name)
    # The pen writes ``width`` verbatim, but a CFF charstring stores it as the offset from
    # the private dict's nominalWidthX.
    width = glyph.metadata.advance_width - private.nominalWidthX
    pen = T2CharStringPen(width=width, glyphSet=font.getGlyphSet())
    _store_cff_charstring(glyph, pen, charstrings, private, cff_table.cff.GlobalSubrs)


def _update_cff2_glyph(glyph: Glyph, _: Any, font: TTFont) -> None:
    """Update a static CFF2 glyph from the domain model.

    CFF2 charstrings carry no width (advances live in hmtx), and the private dict
    comes from the glyph's FDArray entry because CFF2 has no top-level one.

    Args:
        glyph: Domain glyph model
        _: Original fonttools glyph (unused)
        font: The TTFont object
    """
    cff_table = font["CFF2"]
    top_dict = cff_table.cff.topDictIndex[0]
    charstrings = top_dict.CharStrings
    private = _fd_private_dict(top_dict, charstrings, glyph.name)
    pen = T2CharStringPen(width=None, glyphSet=font.getGlyphSet(), CFF2=True)
    _store_cff_charstring(glyph, pen, charstrings, private, cff_table.cff.GlobalSubrs)
