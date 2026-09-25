"""Segment traversal and joining for a fixed-coordinate bridge line."""

from stencilizer.core.axis import Axis
from stencilizer.core.curve import flatten_contour
from stencilizer.domain import Contour, Point, PointType


def _intersection(p1: Point, p2: Point, line: float, axis: Axis, epsilon: float) -> Point:
    start, end = axis.coord(p1), axis.coord(p2)
    if abs(end - start) < epsilon:
        other = (axis.cross(p1) + axis.cross(p2)) / 2
    else:
        t = (line - start) / (end - start)
        other = axis.cross(p1) + t * (axis.cross(p2) - axis.cross(p1))
    return axis.point(line, other, PointType.ON_CURVE)


def _on_side(point: Point, line: float, axis: Axis, lower: bool) -> bool:
    value = axis.coord(point)
    return value <= line if lower else value >= line


def collect_segments(
    points: list[Point],
    line: float,
    axis: Axis,
    lower: bool,
    epsilon: float,
    *,
    require_existing_wrap: bool = False,
) -> list[list[Point]]:
    """Collect contour runs on one side, retaining traversal and wrap order."""
    points = flatten_contour(Contour(points), epsilon * 250).points
    if not points:
        return []

    segments: list[list[Point]] = []
    current: list[Point] = []
    was_on = _on_side(points[0], line, axis, lower)
    previous = points[0]
    if was_on:
        current = [Point(previous.x, previous.y, previous.point_type)]
    for point in points[1:]:
        now_on = _on_side(point, line, axis, lower)
        if now_on:
            if not was_on:
                current = [_intersection(previous, point, line, axis, epsilon)]
            current.append(Point(point.x, point.y, point.point_type))
        elif was_on and current:
            current.append(_intersection(previous, point, line, axis, epsilon))
            segments.append(current)
            current = []
        was_on = now_on
        previous = point
    first_on = _on_side(points[0], line, axis, lower)
    if was_on and not first_on:
        current.append(_intersection(previous, points[0], line, axis, epsilon))
        segments.append(current)
    elif was_on and first_on and current:
        if require_existing_wrap:
            # These builders require a previously closed run at the seam.
            _ = segments[0][0]
        if segments:
            segments[0] = current + segments[0]
        else:
            segments.append(current)
    elif was_on and current:
        segments.append(current)
    elif not was_on and first_on and segments:
        segments[0].insert(0, _intersection(previous, points[0], line, axis, epsilon))
    return segments


def build_contour_from_segments(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    bridge_tolerance: float,
) -> list[Point] | None:
    """Join runs along the bridge, preserving the original closing point order."""
    if not segments:
        return None
    points: list[Point] = []
    for index, segment in enumerate(segments):
        if index > 0:
            previous, start = points[-1], segment[0]
            if (
                not (
                    abs(axis.coord(previous) - line) < bridge_tolerance
                    and abs(axis.coord(start) - line) < bridge_tolerance
                )
                and abs(axis.cross(previous) - axis.cross(start)) > bridge_tolerance
            ):
                points.append(axis.point(line, axis.cross(previous), PointType.ON_CURVE))
                points.append(axis.point(line, axis.cross(start), PointType.ON_CURVE))
        points.extend(segment)
    if points and segments:
        last, first = points[-1], points[0]
        if (
            abs(axis.coord(last) - line) < bridge_tolerance
            and abs(axis.coord(first) - line) < bridge_tolerance
        ):
            pass
        elif (
            abs(axis.cross(last) - axis.cross(first)) > bridge_tolerance
            or abs(axis.coord(last) - axis.coord(first)) > bridge_tolerance
        ):
            if abs(axis.coord(last) - line) > bridge_tolerance:
                points.append(axis.point(line, axis.cross(last), PointType.ON_CURVE))
            if abs(axis.coord(first) - line) > bridge_tolerance:
                points.append(axis.point(line, axis.cross(first), PointType.ON_CURVE))
    return points


def clean_points(points: list[Point], point_tolerance: float) -> list[Point]:
    """Remove consecutive points closer than the original coordinate threshold."""
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or (
            abs(point.x - cleaned[-1].x) > point_tolerance
            or abs(point.y - cleaned[-1].y) > point_tolerance
        ):
            cleaned.append(point)
    return cleaned
