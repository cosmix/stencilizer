"""Convert curved outlines to bounded-error line contours for geometry surgery."""

import math

from stencilizer.domain import Contour, Point, PointType


def curve_tolerance(upm: int) -> float:
    """Maximum control-polygon distance to a chord: 0.25 units at 1000 UPM."""
    if upm <= 0:
        raise ValueError("UPM must be positive")
    return 0.25 * upm / 1000


def _midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2)


def _distance_to_chord(control: Point, start: Point, end: Point) -> float:
    dx, dy = end.x - start.x, end.y - start.y
    squared = dx * dx + dy * dy
    if squared == 0:
        return math.hypot(control.x - start.x, control.y - start.y)
    projection = ((control.x - start.x) * dx + (control.y - start.y) * dy) / squared
    projection = min(1.0, max(0.0, projection))
    return math.hypot(control.x - start.x - projection * dx, control.y - start.y - projection * dy)


def _flatten_bezier(
    points: tuple[Point, ...], tolerance: float, output: list[Point], depth: int = 0
) -> None:
    start, end = points[0], points[-1]
    if all(_distance_to_chord(p, start, end) <= tolerance for p in points[1:-1]):
        output.append(Point(end.x, end.y))
        return
    if depth >= 32:
        raise ValueError("Curve cannot be flattened within tolerance")
    levels: list[list[Point]] = [list(points)]
    while len(levels[-1]) > 1:
        levels.append([_midpoint(a, b) for a, b in zip(levels[-1], levels[-1][1:], strict=False)])
    left = tuple(level[0] for level in levels)
    right = tuple(level[-1] for level in reversed(levels))
    _flatten_bezier(left, tolerance, output, depth + 1)
    _flatten_bezier(right, tolerance, output, depth + 1)


def flatten_contour(contour: Contour, tolerance: float) -> Contour:
    """Flatten cyclic Bezier segments within the control-hull distance tolerance."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Curve tolerance must be positive and finite")
    points = contour.points
    if not points or all(p.point_type == PointType.ON_CURVE for p in points):
        return contour
    first_on = next((i for i, p in enumerate(points) if p.point_type == PointType.ON_CURVE), None)
    if first_on is None:
        if not all(p.point_type == PointType.OFF_CURVE_QUAD for p in points):
            raise ValueError("A cubic contour requires an on-curve point")
        start = _midpoint(points[-1], points[0])
        ordered = [start, *points]
    else:
        ordered = points[first_on:] + points[:first_on]
        start = ordered[0]
    ordered.append(start)
    output = [Point(start.x, start.y)]
    current = start
    i = 1
    while i < len(ordered):
        point = ordered[i]
        if point.point_type == PointType.ON_CURVE:
            output.append(Point(point.x, point.y))
            current = point
            i += 1
        elif point.point_type == PointType.OFF_CURVE_QUAD:
            following = ordered[i + 1]
            end = (
                following
                if following.point_type == PointType.ON_CURVE
                else _midpoint(point, following)
            )
            _flatten_bezier((current, point, end), tolerance, output)
            current = end
            i += 1 if following.point_type != PointType.ON_CURVE else 2
        elif point.point_type == PointType.OFF_CURVE_CUBIC:
            if i + 2 >= len(ordered) or ordered[i + 1].point_type != PointType.OFF_CURVE_CUBIC:
                raise ValueError("Cubic segment requires two control points")
            end = ordered[i + 2]
            if end.point_type != PointType.ON_CURVE:
                raise ValueError("Cubic segment requires an on-curve endpoint")
            _flatten_bezier((current, point, ordered[i + 1], end), tolerance, output)
            current = end
            i += 3
    if len(output) > 1 and output[-1] == output[0]:
        output.pop()
    return Contour(output, direction=contour.direction)
