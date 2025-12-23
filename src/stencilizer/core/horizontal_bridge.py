"""Horizontal bridge contour operations.

This module handles creating contours with horizontal bridges (LEFT/RIGHT gaps),
which splits glyphs into TOP/BOTTOM pieces.
"""

from stencilizer.core.geometry import (
    compute_winding_direction,
    detect_traversal_direction,
    find_all_edge_crossings,
    signed_area,
)
from stencilizer.domain import Contour, Point, PointType, WindingDirection


def create_horizontal_bridge_contours(
    inner: Contour,
    outer: Contour,
    center_y: float,
    half_width: float,
    inner_min_x: float,
    inner_max_x: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
) -> list[Contour]:
    """Create contours with left/right bridges (splits into top/bottom pieces).

    Each piece consists of TWO contours:
    - Outer portion (CW) - the filled outer boundary
    - Inner portion (CCW) - the hole boundary

    The inner remains a separate hole contour, NOT merged with outer.
    Also handles nested contours that cross the bridge.
    """
    bridge_bottom = center_y - half_width
    bridge_top = center_y + half_width

    # Find ALL crossings for both bridge lines on both contours
    outer_top_crossings = find_all_edge_crossings(outer, bridge_top, False)
    outer_bot_crossings = find_all_edge_crossings(outer, bridge_bottom, False)
    inner_top_crossings = find_all_edge_crossings(inner, bridge_top, False)
    inner_bot_crossings = find_all_edge_crossings(inner, bridge_bottom, False)

    # Need at least 2 crossings on outer (left and right of inner) for each bridge
    outer_top_left = [c for c in outer_top_crossings if c[2] < inner_min_x]
    outer_top_right = [c for c in outer_top_crossings if c[2] > inner_max_x]
    outer_bot_left = [c for c in outer_bot_crossings if c[2] < inner_min_x]
    outer_bot_right = [c for c in outer_bot_crossings if c[2] > inner_max_x]

    if not (outer_top_left and outer_top_right and
            outer_bot_left and outer_bot_right and
            inner_top_crossings and inner_bot_crossings):
        return [outer, inner]

    result = []

    # Build TOP piece: outer portion + inner portion (as separate hole)
    top_outer = build_outer_portion_horizontal(
        outer, bridge_top, outer_top_crossings, inner_min_x, inner_max_x, is_top=True
    )
    top_inner = build_inner_portion_horizontal(
        inner, bridge_top, inner_top_crossings, is_top=True
    )
    if top_outer:
        result.append(top_outer)
    if top_inner:
        result.append(top_inner)

    # Build BOTTOM piece: outer portion + inner portion (as separate hole)
    bot_outer = build_outer_portion_horizontal(
        outer, bridge_bottom, outer_bot_crossings, inner_min_x, inner_max_x, is_top=False
    )
    bot_inner = build_inner_portion_horizontal(
        inner, bridge_bottom, inner_bot_crossings, is_top=False
    )
    if bot_outer:
        result.append(bot_outer)
    if bot_inner:
        result.append(bot_inner)

    # Also split any NESTED contours that cross the bridge lines
    if all_contours:
        processed_set = {id(outer), id(inner)}
        inner_bbox = inner.bounding_box()

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

            # Check if contour crosses the bridge lines
            top_crossings = find_all_edge_crossings(contour, bridge_top, False)
            bot_crossings = find_all_edge_crossings(contour, bridge_bottom, False)

            # Determine target winding based on original contour type
            # Negative signed area = CW = outer contour
            # Positive signed area = CCW = inner/hole contour
            is_outer_contour = signed_area(contour.points) < 0
            target_winding = WindingDirection.CLOCKWISE if is_outer_contour else WindingDirection.COUNTER_CLOCKWISE

            if not top_crossings and not bot_crossings:
                # Contour doesn't cross either bridge - check if it's in the gap or outside
                if c_bbox[3] <= bridge_bottom or c_bbox[1] >= bridge_top:
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
            if top_crossings and not bot_crossings:
                # Only crosses top bridge - split on top, keep top portion (outside the gap)
                if c_bbox[1] >= bridge_top:
                    result.append(contour)
                else:
                    top_nested = build_inner_portion_horizontal(
                        contour, bridge_top, top_crossings, is_top=True, target_winding=target_winding
                    )
                    if top_nested:
                        result.append(top_nested)
                continue

            if bot_crossings and not top_crossings:
                # Only crosses bottom bridge - split on bottom, keep bottom portion (outside the gap)
                if c_bbox[3] <= bridge_bottom:
                    result.append(contour)
                else:
                    bot_nested = build_inner_portion_horizontal(
                        contour, bridge_bottom, bot_crossings, is_top=False, target_winding=target_winding
                    )
                    if bot_nested:
                        result.append(bot_nested)
                continue

            # Contour crosses both bridges - split into top and bottom portions
            top_nested = build_inner_portion_horizontal(
                contour, bridge_top, top_crossings, is_top=True, target_winding=target_winding
            )
            bot_nested = build_inner_portion_horizontal(
                contour, bridge_bottom, bot_crossings, is_top=False, target_winding=target_winding
            )

            if top_nested:
                result.append(top_nested)
            if bot_nested:
                result.append(bot_nested)

    return result if result else [outer, inner]


def build_outer_portion_horizontal(
    outer: Contour,
    bridge_y: float,
    outer_crossings: list[tuple[int, float, float]],
    inner_min_x: float,
    inner_max_x: float,
    is_top: bool,
) -> Contour | None:
    """Build outer portion for horizontal bridge (CW contour).

    For TOP piece: trace from left to right along top side (CW)
    For BOTTOM piece: trace from right to left along bottom side (CW)
    """
    try:
        outer_left = [c for c in outer_crossings if c[2] < inner_min_x]
        outer_right = [c for c in outer_crossings if c[2] > inner_max_x]

        if not outer_left or not outer_right:
            return None

        outer_left_pt = min(outer_left, key=lambda c: c[2])
        outer_right_pt = max(outer_right, key=lambda c: c[2])

        outer_points = outer.points
        n_outer = len(outer_points)

        def on_correct_side(y: float) -> bool:
            if is_top:
                return y >= bridge_y
            else:
                return y <= bridge_y

        def intersect_bridge(p1: Point, p2: Point) -> tuple[float, float]:
            if abs(p2.y - p1.y) < 0.001:
                return ((p1.x + p2.x) / 2, bridge_y)
            t = (bridge_y - p1.y) / (p2.y - p1.y)
            x = p1.x + t * (p2.x - p1.x)
            return (x, bridge_y)

        if is_top:
            outer_start = outer_left_pt
            outer_end = outer_right_pt
        else:
            outer_start = outer_right_pt
            outer_end = outer_left_pt

        outer_dir = detect_traversal_direction(
            outer_points, outer_start[0], outer_end[0],
            bridge_y, want_less_than=not is_top, is_x=False
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

        # Collect all segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []

        start_pt = Point(outer_start[2], bridge_y, PointType.ON_CURVE)
        current_segment = [start_pt]
        was_on_correct = True
        last_point = start_pt

        for p in traverse_points:
            curr_on_correct = on_correct_side(p.y)

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
        end_pt = Point(outer_end[2], bridge_y, PointType.ON_CURVE)
        if was_on_correct and current_segment:
            current_segment.append(end_pt)
            segments.append(current_segment)
        elif not was_on_correct:
            ix, iy = intersect_bridge(last_point, Point(outer_end[2], bridge_y, PointType.ON_CURVE))
            if current_segment:
                current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                segments.append(current_segment)

        if not segments:
            return None

        # Build final contour by connecting segments along bridge line
        if is_top:
            segments.sort(key=lambda seg: min(pt.x for pt in seg))
        else:
            segments.sort(key=lambda seg: max(pt.x for pt in seg), reverse=True)

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


def build_inner_portion_horizontal(
    inner: Contour,
    bridge_y: float,
    inner_crossings: list[tuple[int, float, float]],
    is_top: bool,
    target_winding: WindingDirection | None = None,
) -> Contour | None:
    """Build inner portion for horizontal bridge.

    Args:
        inner: The contour to split
        bridge_y: Y position of the bridge line
        inner_crossings: All crossings of the contour with the bridge line
        is_top: If True, return the top portion; otherwise bottom portion
        target_winding: Desired winding direction. If None, defaults to CCW (hole).

    The inner must be traced in the OPPOSITE direction from outer to be a hole.
    """
    if target_winding is None:
        target_winding = WindingDirection.COUNTER_CLOCKWISE
    try:
        if len(inner_crossings) < 2:
            return None

        sorted_crossings = sorted(inner_crossings, key=lambda c: c[2])
        inner_left = sorted_crossings[0]
        inner_right = sorted_crossings[-1]

        inner_points = inner.points
        n_inner = len(inner_points)

        def on_correct_side(y: float) -> bool:
            if is_top:
                return y >= bridge_y
            else:
                return y <= bridge_y

        def intersect_bridge(p1: Point, p2: Point) -> tuple[float, float]:
            if abs(p2.y - p1.y) < 0.001:
                return ((p1.x + p2.x) / 2, bridge_y)
            t = (bridge_y - p1.y) / (p2.y - p1.y)
            x = p1.x + t * (p2.x - p1.x)
            return (x, bridge_y)

        # REVERSED from outer: inner traced opposite direction to create hole
        if is_top:
            inner_start = inner_right
            inner_end = inner_left
        else:
            inner_start = inner_left
            inner_end = inner_right

        inner_dir = detect_traversal_direction(
            inner_points, inner_start[0], inner_end[0],
            bridge_y, want_less_than=not is_top, is_x=False
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

        start_pt = Point(inner_start[2], bridge_y, PointType.ON_CURVE)
        current_segment = [start_pt]
        was_on_correct = True
        last_point = start_pt

        for p in traverse_points:
            curr_on_correct = on_correct_side(p.y)

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
        end_pt = Point(inner_end[2], bridge_y, PointType.ON_CURVE)
        if was_on_correct and current_segment:
            current_segment.append(end_pt)
            segments.append(current_segment)
        elif not was_on_correct:
            ix, iy = intersect_bridge(last_point, Point(inner_end[2], bridge_y, PointType.ON_CURVE))
            if current_segment:
                current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                segments.append(current_segment)

        if not segments:
            return None

        # Build final contour
        if is_top:
            segments.sort(key=lambda seg: max(pt.x for pt in seg), reverse=True)
        else:
            segments.sort(key=lambda seg: min(pt.x for pt in seg))

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
