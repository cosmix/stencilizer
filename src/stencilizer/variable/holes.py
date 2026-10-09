"""Count the counters an outline's non-zero fill encloses.

``GlyphAnalyzer`` finds islands by contour nesting, so it misses a counter closed by a
self-intersecting contour or by a hairline of ink between two cut edges that should
coincide. The union through skia-pathops sees both.
"""

from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.crossings import Pt, polygon_area
from stencilizer.variable.overlaps import union_polygons

# Holes smaller than this, in font units squared, are float noise along coincident edges.
HOLE_AREA_FLOOR = 1.0


def _loops(polygon: list[Pt]) -> list[list[Pt]]:
    """``polygon`` split at every vertex it passes twice, so a single-point touch closes."""
    loops: list[list[Pt]] = []
    stack: list[Pt] = []
    seen: dict[Pt, int] = {}
    for point in polygon:
        start = seen.get(point)
        if start is None:
            seen[point] = len(stack)
            stack.append(point)
            continue
        loops.append(stack[start:])
        for dropped in stack[start + 1 :]:
            del seen[dropped]
        del stack[start + 1 :]
    loops.append(stack)
    return loops


def _holes(polygons: list[list[Pt]]) -> int | None:
    """Hole loops of the union of ``polygons``; None when skia-pathops cannot resolve it."""
    union = union_polygons(polygons)
    if union is None:
        return None
    # The union's outer contours run clockwise, so its holes have a positive signed area.
    return sum(
        1 for polygon in union for loop in _loops(polygon) if polygon_area(loop) > HOLE_AREA_FLOOR
    )


def enclosed_counters(glyph: Glyph, upm: int) -> int | None:
    """Counters enclosed by ``glyph``'s non-zero fill; None when they cannot be counted.

    The union is taken with the contours as given and reversed, and the larger count
    wins: skia-pathops resolves a tangle of near-coincident edges differently by
    direction, and the CFF2 writer stores contours reversed. A glyph that cannot be
    flattened (a non-positive ``upm``, a cubic contour without an on-curve point) is
    uncountable.
    """
    try:
        tolerance = curve_tolerance(upm)
        polygons = [
            [(p.x, p.y) for p in flatten_contour(contour, tolerance).points]
            for contour in glyph.contours
            if contour.points
        ]
    except ValueError:
        return None
    forward = _holes(polygons)
    backward = _holes([polygon[::-1] for polygon in polygons])
    if forward is None or backward is None:
        return None
    return max(forward, backward)
