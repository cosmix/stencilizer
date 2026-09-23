"""Outer contour construction for horizontally spanning multi-island cuts."""

from stencilizer.core.geometry import (
    compute_winding_direction,
    detect_traversal_direction,
    find_all_edge_crossings,
)
from stencilizer.domain import Contour, Point, PointType, WindingDirection


def _intersect_bridge(p1: Point, p2: Point, bridge_y: float, epsilon: float) -> tuple[float, float]:
    if abs(p2.y - p1.y) < epsilon:
        return ((p1.x + p2.x) / 2, bridge_y)
    t = (bridge_y - p1.y) / (p2.y - p1.y)
    x = p1.x + t * (p2.x - p1.x)
    return (x, bridge_y)


def _traverse_outer(
    outer: Contour,
    start: tuple[int, float, float],
    end: tuple[int, float, float],
    bridge_y: float,
    is_top: bool,
) -> list[Point]:
    outer_points = outer.points
    n_outer = len(outer_points)
    outer_dir = detect_traversal_direction(
        outer_points, start[0], end[0], bridge_y, want_less_than=not is_top, is_x=False
    )
    traverse_points: list[Point] = []
    if outer_dir == -1:
        idx = start[0]
        target = end[0]
        count = 0
        while idx != target and count < n_outer:
            traverse_points.append(outer_points[idx])
            idx = (idx - 1) % n_outer
            count += 1
    else:
        idx = (start[0] + 1) % n_outer
        target = (end[0] + 1) % n_outer
        count = 0
        while idx != target and count < n_outer:
            traverse_points.append(outer_points[idx])
            idx = (idx + 1) % n_outer
            count += 1
    return traverse_points


def _collect_segments(
    traverse_points: list[Point],
    start: tuple[int, float, float],
    end: tuple[int, float, float],
    bridge_y: float,
    is_top: bool,
    epsilon: float,
) -> list[list[Point]]:
    segments: list[list[Point]] = []
    start_pt = Point(start[2], bridge_y, PointType.ON_CURVE)
    current_segment = [start_pt]
    was_on_correct = True
    last_point = start_pt
    for p in traverse_points:
        curr_on_correct = p.y >= bridge_y if is_top else p.y <= bridge_y
        if curr_on_correct:
            if not was_on_correct:
                ix, iy = _intersect_bridge(last_point, p, bridge_y, epsilon)
                current_segment = [Point(ix, iy, PointType.ON_CURVE)]
            current_segment.append(Point(p.x, p.y, p.point_type))
        elif was_on_correct and current_segment:
            ix, iy = _intersect_bridge(last_point, p, bridge_y, epsilon)
            current_segment.append(Point(ix, iy, PointType.ON_CURVE))
            segments.append(current_segment)
            current_segment = []
        was_on_correct = curr_on_correct
        last_point = p
    end_pt = Point(end[2], bridge_y, PointType.ON_CURVE)
    if was_on_correct and current_segment:
        current_segment.append(end_pt)
        segments.append(current_segment)
    elif not was_on_correct:
        ix, iy = _intersect_bridge(last_point, end_pt, bridge_y, epsilon)
        if current_segment:
            current_segment.append(Point(ix, iy, PointType.ON_CURVE))
            segments.append(current_segment)
    return segments


def _connect_segments(
    segments: list[list[Point]], is_top: bool, bridge_y: float, tolerance: float
) -> list[Point]:
    if is_top:
        segments.sort(key=lambda seg: max(pt.x for pt in seg), reverse=True)
    else:
        segments.sort(key=lambda seg: min(pt.x for pt in seg))
    points: list[Point] = []
    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = points[-1]
            seg_start = seg[0]
            if abs(prev_end.x - seg_start.x) > tolerance:
                if abs(prev_end.y - bridge_y) > tolerance:
                    points.append(Point(prev_end.x, bridge_y, PointType.ON_CURVE))
                points.append(Point(seg_start.x, bridge_y, PointType.ON_CURVE))
        points.extend(seg)
    if points and len(points) > 2:
        last_pt = points[-1]
        first_pt = points[0]
        if abs(last_pt.x - first_pt.x) > tolerance or abs(last_pt.y - first_pt.y) > tolerance:
            if abs(last_pt.y - bridge_y) > tolerance:
                points.append(Point(last_pt.x, bridge_y, PointType.ON_CURVE))
            if (
                abs(first_pt.y - bridge_y) > tolerance
                and abs(first_pt.x - points[-1].x) > tolerance
            ):
                points.append(Point(first_pt.x, bridge_y, PointType.ON_CURVE))
    return points


def _finish_outer(points: list[Point], duplicate_tolerance: float) -> Contour | None:
    cleaned_points: list[Point] = []
    for p in points:
        if not cleaned_points or (
            abs(p.x - cleaned_points[-1].x) > duplicate_tolerance
            or abs(p.y - cleaned_points[-1].y) > duplicate_tolerance
        ):
            cleaned_points.append(p)
    if len(cleaned_points) < 3:
        return None
    direction = compute_winding_direction(cleaned_points)
    if direction != WindingDirection.CLOCKWISE:
        cleaned_points = list(reversed(cleaned_points))
        direction = WindingDirection.CLOCKWISE
    return Contour(points=cleaned_points, direction=direction)


def build_outer_portion_multi_island_axis(
    outer: Contour,
    bridge_y: float,
    is_top: bool,
    *,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> Contour | None:
    """Build the horizontal outer portion, including stems between islands."""
    try:
        outer_crossings = find_all_edge_crossings(outer, bridge_y, False, epsilon=epsilon)
        if len(outer_crossings) < 2:
            return None
        sorted_crossings = sorted(outer_crossings, key=lambda c: c[2])
        if is_top:
            start_crossing, end_crossing = sorted_crossings[-1], sorted_crossings[0]
        else:
            start_crossing, end_crossing = sorted_crossings[0], sorted_crossings[-1]
        traverse_points = _traverse_outer(outer, start_crossing, end_crossing, bridge_y, is_top)
        segments = _collect_segments(
            traverse_points, start_crossing, end_crossing, bridge_y, is_top, epsilon
        )
        if not segments:
            return None
        points = _connect_segments(segments, is_top, bridge_y, connection_tolerance)
        return _finish_outer(points, duplicate_tolerance)
    except Exception:
        return None
