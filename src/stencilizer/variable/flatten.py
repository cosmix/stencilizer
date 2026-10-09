"""Flatten a variable glyph to polygons with one structure shared by every master."""

import math
from itertools import pairwise

from stencilizer.core.curve import curve_tolerance
from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable.model import VariableGlyph

MAX_SUBDIVISIONS = 64

_Pt = tuple[float, float]
_Segment = tuple[_Pt, ...]


def _mid(a: _Pt, b: _Pt) -> _Pt:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _segments(points: list[Point], name: str) -> list[_Segment]:
    """Bezier segments of a closed contour, started and walked as ``core.curve`` does."""
    first_on = next((i for i, p in enumerate(points) if p.point_type == PointType.ON_CURVE), None)
    if first_on is None:
        if any(p.point_type != PointType.OFF_CURVE_QUAD for p in points):
            raise VariationDataError(name, "cubic contour without an on-curve point")
        ordered = [_mid((points[-1].x, points[-1].y), (points[0].x, points[0].y))]
        ordered_types = [PointType.ON_CURVE]
        ordered += [(p.x, p.y) for p in points]
        ordered_types += [p.point_type for p in points]
    else:
        rotated = points[first_on:] + points[:first_on]
        ordered = [(p.x, p.y) for p in rotated]
        ordered_types = [p.point_type for p in rotated]
    ordered.append(ordered[0])
    ordered_types.append(PointType.ON_CURVE)
    return _walk(ordered, ordered_types, name)


def _walk(ordered: list[_Pt], types: list[PointType], name: str) -> list[_Segment]:
    segments: list[_Segment] = []
    current = ordered[0]
    i = 1
    while i < len(ordered):
        if types[i] == PointType.ON_CURVE:
            segments.append((current, ordered[i]))
            current = ordered[i]
            i += 1
        elif types[i] == PointType.OFF_CURVE_QUAD:
            following_on = types[i + 1] == PointType.ON_CURVE
            end = ordered[i + 1] if following_on else _mid(ordered[i], ordered[i + 1])
            segments.append((current, ordered[i], end))
            current = end
            i += 2 if following_on else 1
        else:
            if i + 2 >= len(ordered) or types[i + 1] != PointType.OFF_CURVE_CUBIC:
                raise VariationDataError(name, "cubic segment without two control points")
            if types[i + 2] != PointType.ON_CURVE:
                raise VariationDataError(name, "cubic segment without an on-curve endpoint")
            segments.append((current, ordered[i], ordered[i + 1], ordered[i + 2]))
            current = ordered[i + 2]
            i += 3
    return segments


def _curvature_bound(segment: _Segment) -> float:
    """Upper bound of |B''(t)| over [0, 1]."""
    if len(segment) < 3:
        return 0.0
    diffs = [
        math.hypot(a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1])
        for a, b, c in zip(segment, segment[1:], segment[2:], strict=False)
    ]
    return (len(segment) - 1) * (len(segment) - 2) * max(diffs)


def _subdivisions(segments: list[_Segment], tolerance: float, name: str) -> int:
    """Smallest n whose uniform-t chords deviate at most ``tolerance`` in every master.

    The chord of a parameter piece of width h deviates from the curve by at most
    max|B''| * h^2 / 8.
    """
    bound = max(_curvature_bound(s) for s in segments)
    count = max(1, math.ceil(math.sqrt(bound / (8 * tolerance)) - 1e-9))
    if count > MAX_SUBDIVISIONS:
        raise VariationDataError(name, f"curve needs more than {MAX_SUBDIVISIONS} subdivisions")
    return count


def _bezier(segment: _Segment, t: float) -> _Pt:
    level = list(segment)
    while len(level) > 1:
        level = [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for a, b in pairwise(level)]
    return level[0]


def _sample(segment: _Segment, count: int) -> list[_Pt]:
    """Points at t = 1/count .. 1; the last one is the exact end point."""
    return [_bezier(segment, k / count) for k in range(1, count)] + [segment[-1]]


def _flatten_contours(glyphs: list[Glyph], upm: int) -> list[list[Contour]]:
    """Flatten contour ``i`` of every glyph with shared subdivision counts."""
    tolerance = curve_tolerance(upm)
    name = glyphs[0].name
    result: list[list[Contour]] = [[] for _ in glyphs]
    for index, reference in enumerate(glyphs[0].contours):
        if all(p.point_type == PointType.ON_CURVE for p in reference.points):
            for out, glyph in zip(result, glyphs, strict=True):
                out.append(
                    Contour(list(glyph.contours[index].points), glyph.contours[index].direction)
                )
            continue
        per_glyph = [_segments(g.contours[index].points, name) for g in glyphs]
        counts = [
            _subdivisions([segs[k] for segs in per_glyph], tolerance, name)
            for k in range(len(per_glyph[0]))
        ]
        for out, glyph, segs in zip(result, glyphs, per_glyph, strict=True):
            points = [Point(*segs[0][0])]
            for segment, count in zip(segs, counts, strict=True):
                points += [Point(x, y) for x, y in _sample(segment, count)]
            points.pop()  # the last segment ends at the contour start
            out.append(Contour(points, direction=glyph.contours[index].direction))
    return result


def flatten_compatible(vg: VariableGlyph, upm: int) -> VariableGlyph:
    """Polygon version of ``vg`` with identical point structure in the default and every master."""
    glyphs = [vg.default, *vg.masters]
    flat = _flatten_contours(glyphs, upm)
    rebuilt = [
        Glyph(metadata=g.metadata, contours=c, _is_composite=g._is_composite)
        for g, c in zip(glyphs, flat, strict=True)
    ]
    return VariableGlyph(rebuilt[0], vg.supports, tuple(rebuilt[1:]), vg.axis_tags, vg.cff2)
