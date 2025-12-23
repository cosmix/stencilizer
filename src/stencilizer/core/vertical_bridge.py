"""Vertical bridge contour operations.

This module handles creating contours with vertical bridges (TOP/BOTTOM gaps),
which splits glyphs into LEFT/RIGHT pieces.
"""

from stencilizer.core.geometry import (
    compute_winding_direction,
    detect_traversal_direction,
    find_all_edge_crossings,
    signed_area,
)
from stencilizer.domain import Contour, Point, PointType, WindingDirection


def create_vertical_bridge_contours(
    inner: Contour,
    outer: Contour,
    center_x: float,
    half_width: float,
    inner_min_y: float,
    inner_max_y: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
) -> list[Contour]:
    """Create contours with top/bottom bridges (splits into left/right pieces).

    Each piece consists of TWO contours:
    - Outer portion (CW) - the filled outer boundary
    - Inner portion (CCW) - the hole boundary

    The inner remains a separate hole contour, NOT merged with outer.
    Also handles nested contours (like R's counter in ®) that cross the bridge.
    """
    bridge_left = center_x - half_width
    bridge_right = center_x + half_width

    # Find ALL crossings for both bridge lines on both contours
    outer_left_crossings = find_all_edge_crossings(outer, bridge_left, True)
    outer_right_crossings = find_all_edge_crossings(outer, bridge_right, True)
    inner_left_crossings = find_all_edge_crossings(inner, bridge_left, True)
    inner_right_crossings = find_all_edge_crossings(inner, bridge_right, True)

    # Need at least 2 crossings on outer (above and below inner) for each bridge
    outer_left_above = [c for c in outer_left_crossings if c[2] > inner_max_y]
    outer_left_below = [c for c in outer_left_crossings if c[2] < inner_min_y]
    outer_right_above = [c for c in outer_right_crossings if c[2] > inner_max_y]
    outer_right_below = [c for c in outer_right_crossings if c[2] < inner_min_y]

    if not (outer_left_above and outer_left_below and
            outer_right_above and outer_right_below and
            inner_left_crossings and inner_right_crossings):
        return [outer, inner]

    result = []

    # Build LEFT piece: outer portion + inner portion (as separate hole)
    left_outer = build_outer_portion_vertical(
        outer, bridge_left, outer_left_crossings, inner_min_y, inner_max_y, is_left=True
    )
    left_inner = build_inner_portion_vertical(
        inner, bridge_left, inner_left_crossings, is_left=True
    )
    if left_outer:
        result.append(left_outer)
    if left_inner:
        result.append(left_inner)

    # Build RIGHT piece: outer portion + inner portion (as separate hole)
    right_outer = build_outer_portion_vertical(
        outer, bridge_right, outer_right_crossings, inner_min_y, inner_max_y, is_left=False
    )
    right_inner = build_inner_portion_vertical(
        inner, bridge_right, inner_right_crossings, is_left=False
    )
    if right_outer:
        result.append(right_outer)
    if right_inner:
        result.append(right_inner)

    # Also split any NESTED contours that cross the bridge lines
    if all_contours:
        processed_set = {id(outer), id(inner)}
        inner_bbox = inner.bounding_box()
        outer_area = signed_area(outer.points)

        for contour in all_contours:
            if id(contour) in processed_set:
                continue

            # Check if contour is geometrically within the INNER's region
            c_bbox = contour.bounding_box()
            c_center_x = (c_bbox[0] + c_bbox[2]) / 2
            c_center_y = (c_bbox[1] + c_bbox[3]) / 2

            # Must be contained within inner's bounding box (truly nested)
            if not (inner_bbox[0] < c_center_x < inner_bbox[2] and
                    inner_bbox[1] < c_center_y < inner_bbox[3]):
                continue

            # Check if this is a structural element (like Theta's bar)
            # Structural elements have same winding as outer and span across the bridge
            contour_area = signed_area(contour.points)
            same_winding = (contour_area < 0) == (outer_area < 0)
            if same_winding:
                # Check if it spans across the bridge gap
                spans_bridge = c_bbox[0] < bridge_left and c_bbox[2] > bridge_right
                if spans_bridge:
                    # This is a structural element - add unchanged, don't split
                    result.append(contour)
                    if processed_nested is not None:
                        processed_nested.append(contour)
                    continue

            # Check if contour crosses the bridge lines
            left_crossings = find_all_edge_crossings(contour, bridge_left, True)
            right_crossings = find_all_edge_crossings(contour, bridge_right, True)

            # Determine target winding based on original contour type
            # Negative signed area = CW = outer contour
            # Positive signed area = CCW = inner/hole contour
            is_outer_contour = signed_area(contour.points) < 0
            target_winding = WindingDirection.CLOCKWISE if is_outer_contour else WindingDirection.COUNTER_CLOCKWISE

            if not left_crossings and not right_crossings:
                # Contour doesn't cross either bridge - check if it's in the gap or outside
                if c_bbox[2] <= bridge_left or c_bbox[0] >= bridge_right:
                    # Entirely outside the gap - add unchanged
                    result.append(contour)
                    if processed_nested is not None:
                        processed_nested.append(contour)
                # else: Contour is entirely within the gap - drop it (correct for stencil)
                continue

            # Track original as processed
            if processed_nested is not None:
                processed_nested.append(contour)

            # Handle contours that cross only one bridge line
            if left_crossings and not right_crossings:
                # Only crosses left bridge - split on left, keep right portion if outside gap
                if c_bbox[0] >= bridge_right:
                    # Entirely right of right bridge after considering crossings - shouldn't happen
                    result.append(contour)
                else:
                    # Split on left bridge, take the left portion (outside the gap)
                    left_nested = build_inner_portion_vertical(
                        contour, bridge_left, left_crossings, is_left=True, target_winding=target_winding
                    )
                    if left_nested:
                        result.append(left_nested)
                continue

            if right_crossings and not left_crossings:
                # Only crosses right bridge - split on right, keep right portion (outside the gap)
                if c_bbox[2] <= bridge_left:
                    # Entirely left of left bridge - shouldn't happen
                    result.append(contour)
                else:
                    # Split on right bridge, take the right portion (outside the gap)
                    right_nested = build_inner_portion_vertical(
                        contour, bridge_right, right_crossings, is_left=False, target_winding=target_winding
                    )
                    if right_nested:
                        result.append(right_nested)
                continue

            # Contour crosses both bridges - split into left and right portions
            left_nested = build_inner_portion_vertical(
                contour, bridge_left, left_crossings, is_left=True, target_winding=target_winding
            )
            right_nested = build_inner_portion_vertical(
                contour, bridge_right, right_crossings, is_left=False, target_winding=target_winding
            )

            if left_nested:
                result.append(left_nested)
            if right_nested:
                result.append(right_nested)

    return result if result else [outer, inner]


def build_outer_portion_vertical(
    outer: Contour,
    bridge_x: float,
    outer_crossings: list[tuple[int, float, float]],
    inner_min_y: float,
    inner_max_y: float,
    is_left: bool,
) -> Contour | None:
    """Build outer portion for vertical bridge (CW contour).

    For LEFT piece: collect segments with X <= bridge_x
    For RIGHT piece: collect segments with X >= bridge_x
    """
    try:
        outer_above = [c for c in outer_crossings if c[2] > inner_max_y]
        outer_below = [c for c in outer_crossings if c[2] < inner_min_y]

        if not outer_above or not outer_below:
            return None

        outer_top = max(outer_above, key=lambda c: c[2])
        outer_bot = min(outer_below, key=lambda c: c[2])

        outer_points = outer.points
        n_outer = len(outer_points)

        def on_correct_side(x: float) -> bool:
            if is_left:
                return x <= bridge_x
            else:
                return x >= bridge_x

        def intersect_bridge(p1: Point, p2: Point) -> tuple[float, float]:
            if abs(p2.x - p1.x) < 0.001:
                return (bridge_x, (p1.y + p2.y) / 2)
            t = (bridge_x - p1.x) / (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            return (bridge_x, y)

        # Collect all segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []

        if is_left:
            outer_start = outer_bot
            outer_end = outer_top
        else:
            outer_start = outer_top
            outer_end = outer_bot

        outer_dir = detect_traversal_direction(
            outer_points, outer_start[0], outer_end[0],
            bridge_x, want_less_than=is_left, is_x=True
        )

        # Build traversal list
        traverse_points: list[Point] = []
        if outer_dir == -1:
            idx = outer_start[0]
            target = outer_end[0]
            count = 0
            while idx != target and count < n_outer:
                traverse_points.append(outer_points[idx])
                idx = (idx - 1) % n_outer
                count += 1
        else:
            idx = (outer_start[0] + 1) % n_outer
            target = (outer_end[0] + 1) % n_outer
            count = 0
            while idx != target and count < n_outer:
                traverse_points.append(outer_points[idx])
                idx = (idx + 1) % n_outer
                count += 1

        # Add start point on bridge line
        start_pt = Point(bridge_x, outer_start[2], PointType.ON_CURVE)
        current_segment = [start_pt]
        was_on_correct = True
        last_point = start_pt

        for p in traverse_points:
            curr_on_correct = on_correct_side(p.x)

            if curr_on_correct:
                if not was_on_correct:
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment = [Point(ix, iy, PointType.ON_CURVE)]
                current_segment.append(Point(p.x, p.y, p.point_type))
            else:
                if was_on_correct and current_segment:
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                    segments.append(current_segment)
                    current_segment = []

            was_on_correct = curr_on_correct
            last_point = p

        # Handle end
        end_pt = Point(bridge_x, outer_end[2], PointType.ON_CURVE)
        if was_on_correct and current_segment:
            current_segment.append(end_pt)
            segments.append(current_segment)
        elif not was_on_correct:
            ix, iy = intersect_bridge(last_point, Point(bridge_x, outer_end[2], PointType.ON_CURVE))
            if current_segment:
                current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                segments.append(current_segment)

        if not segments:
            return None

        # Build final contour by connecting segments along bridge line
        if is_left:
            segments.sort(key=lambda seg: min(pt.y for pt in seg))
        else:
            segments.sort(key=lambda seg: max(pt.y for pt in seg), reverse=True)

        points: list[Point] = []
        for seg in segments:
            points.extend(seg)

        # Remove duplicate consecutive points
        cleaned_points: list[Point] = []
        for p in points:
            if not cleaned_points or (abs(p.x - cleaned_points[-1].x) > 0.5 or abs(p.y - cleaned_points[-1].y) > 0.5):
                cleaned_points.append(p)

        if len(cleaned_points) < 3:
            return None

        # Enforce CLOCKWISE for outer contours
        direction = compute_winding_direction(cleaned_points)
        if direction != WindingDirection.CLOCKWISE:
            cleaned_points = list(reversed(cleaned_points))
            direction = WindingDirection.CLOCKWISE
        return Contour(points=cleaned_points, direction=direction)

    except Exception:
        return None


def build_inner_portion_vertical(
    inner: Contour,
    bridge_x: float,
    inner_crossings: list[tuple[int, float, float]],
    is_left: bool,
    target_winding: WindingDirection | None = None,
) -> Contour | None:
    """Build inner portion for vertical bridge.

    Args:
        inner: The contour to split
        bridge_x: X position of the bridge line
        inner_crossings: All crossings of the contour with the bridge line
        is_left: If True, return the left portion; otherwise right portion
        target_winding: Desired winding direction. If None, defaults to CCW (hole).

    The inner must be traced in the OPPOSITE direction from outer to be a hole.
    """
    if target_winding is None:
        target_winding = WindingDirection.COUNTER_CLOCKWISE
    try:
        if len(inner_crossings) < 2:
            return None

        sorted_crossings = sorted(inner_crossings, key=lambda c: c[2])
        inner_bot = sorted_crossings[0]
        inner_top = sorted_crossings[-1]

        inner_points = inner.points
        n_inner = len(inner_points)

        def on_correct_side(x: float) -> bool:
            if is_left:
                return x <= bridge_x
            else:
                return x >= bridge_x

        def intersect_bridge(p1: Point, p2: Point) -> tuple[float, float]:
            if abs(p2.x - p1.x) < 0.001:
                return (bridge_x, (p1.y + p2.y) / 2)
            t = (bridge_x - p1.x) / (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            return (bridge_x, y)

        # REVERSED from outer: inner traced opposite direction to create hole
        if is_left:
            inner_start = inner_top
            inner_end = inner_bot
        else:
            inner_start = inner_bot
            inner_end = inner_top

        inner_dir = detect_traversal_direction(
            inner_points, inner_start[0], inner_end[0],
            bridge_x, want_less_than=is_left, is_x=True
        )

        # Build traversal list
        traverse_points: list[Point] = []
        if inner_dir == -1:
            idx = inner_start[0]
            target = inner_end[0]
            count = 0
            while idx != target and count < n_inner:
                traverse_points.append(inner_points[idx])
                idx = (idx - 1) % n_inner
                count += 1
        else:
            idx = (inner_start[0] + 1) % n_inner
            target = (inner_end[0] + 1) % n_inner
            count = 0
            while idx != target and count < n_inner:
                traverse_points.append(inner_points[idx])
                idx = (idx + 1) % n_inner
                count += 1

        # Collect all segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []

        start_pt = Point(bridge_x, inner_start[2], PointType.ON_CURVE)
        current_segment = [start_pt]
        was_on_correct = True
        last_point = start_pt

        for p in traverse_points:
            curr_on_correct = on_correct_side(p.x)

            if curr_on_correct:
                if not was_on_correct:
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment = [Point(ix, iy, PointType.ON_CURVE)]
                current_segment.append(Point(p.x, p.y, p.point_type))
            else:
                if was_on_correct and current_segment:
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                    segments.append(current_segment)
                    current_segment = []

            was_on_correct = curr_on_correct
            last_point = p

        # Handle end
        end_pt = Point(bridge_x, inner_end[2], PointType.ON_CURVE)
        if was_on_correct and current_segment:
            current_segment.append(end_pt)
            segments.append(current_segment)
        elif not was_on_correct:
            ix, iy = intersect_bridge(last_point, Point(bridge_x, inner_end[2], PointType.ON_CURVE))
            if current_segment:
                current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                segments.append(current_segment)

        if not segments:
            return None

        # Build final contour
        if is_left:
            segments.sort(key=lambda seg: max(pt.y for pt in seg), reverse=True)
        else:
            segments.sort(key=lambda seg: min(pt.y for pt in seg))

        points: list[Point] = []
        for seg in segments:
            points.extend(seg)

        # Remove duplicate consecutive points
        cleaned_points: list[Point] = []
        for p in points:
            if not cleaned_points or (abs(p.x - cleaned_points[-1].x) > 0.5 or abs(p.y - cleaned_points[-1].y) > 0.5):
                cleaned_points.append(p)

        if len(cleaned_points) < 3:
            return None

        # Enforce target winding direction
        direction = compute_winding_direction(cleaned_points)
        if direction != target_winding:
            cleaned_points = list(reversed(cleaned_points))
            direction = target_winding
        return Contour(points=cleaned_points, direction=direction)

    except Exception:
        return None
