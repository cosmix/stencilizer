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

            # Skip grandchildren: contours that are inside ANOTHER filled contour
            # that's also inside the hole. They should be processed with their direct parent.
            is_grandchild = False
            for other in all_contours:
                if other is contour or other is inner or other is outer:
                    continue
                other_area = signed_area(other.points)
                if other_area < 0:  # CW = filled (potential parent)
                    other_bbox = other.bounding_box()
                    # Check if contour's center is inside this other filled contour
                    if (other_bbox[0] < c_center_x < other_bbox[2] and
                        other_bbox[1] < c_center_y < other_bbox[3]):
                        # This other filled contour must also be inside the hole
                        other_center_x = (other_bbox[0] + other_bbox[2]) / 2
                        other_center_y = (other_bbox[1] + other_bbox[3]) / 2
                        if (inner_bbox[0] < other_center_x < inner_bbox[2] and
                            inner_bbox[1] < other_center_y < inner_bbox[3]):
                            is_grandchild = True
                            break
            if is_grandchild:
                continue

            # Check if this is a filled contour (CW) ENTIRELY INSIDE the hole.
            # Such contours are self-contained geometry (like Theta's bar, or any
            # decorative element inside a hole). They should NEVER be split by
            # bridges - the bridge connects outer to inner, not nested elements.
            is_filled_contour = signed_area(contour.points) < 0  # CW = filled
            entirely_inside_hole = (
                c_bbox[0] >= inner_bbox[0] and c_bbox[2] <= inner_bbox[2] and
                c_bbox[1] >= inner_bbox[1] and c_bbox[3] <= inner_bbox[3]
            )

            if is_filled_contour and entirely_inside_hole:
                # Check if this nested outer has its own holes (like P in ℗ or R in ®).
                # If so, skip it here - it will be processed in nested_outers section.
                has_own_holes = False
                if all_contours:
                    for other in all_contours:
                        if other is contour or other is inner or other is outer:
                            continue
                        other_area = signed_area(other.points)
                        if other_area > 0:  # CCW = hole
                            other_bbox = other.bounding_box()
                            # Check if this hole is inside the filled contour
                            if (c_bbox[0] < other_bbox[0] and c_bbox[2] > other_bbox[2] and
                                c_bbox[1] < other_bbox[1] and c_bbox[3] > other_bbox[3]):
                                has_own_holes = True
                                break

                if not has_own_holes:
                    # Preserve simple self-contained geometry (no children)
                    result.append(contour)
                    if processed_nested is not None:
                        processed_nested.append(contour)
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

        # Each segment's endpoints on bridge_y need connecting edges to form closed contour
        points: list[Point] = []
        for i, seg in enumerate(segments):
            if i > 0:
                # Add connecting edge along bridge line from previous segment to this one
                prev_end = points[-1]
                seg_start = seg[0]
                # Both should be on bridge_y - add horizontal edge
                if abs(prev_end.y - bridge_y) < 1 and abs(seg_start.y - bridge_y) < 1:
                    # Already on bridge line - just add the segment
                    pass
                else:
                    # Need to connect via bridge line if gap exists
                    if abs(prev_end.x - seg_start.x) > 1:
                        # Add intermediate point on bridge line
                        points.append(Point(prev_end.x, bridge_y, PointType.ON_CURVE))
                        points.append(Point(seg_start.x, bridge_y, PointType.ON_CURVE))
            points.extend(seg)

        # Close the contour by connecting last segment back to first along bridge line
        if points and segments:
            last_pt = points[-1]
            first_pt = points[0]
            if abs(last_pt.y - bridge_y) < 1 and abs(first_pt.y - bridge_y) < 1:
                # Both on bridge - add closing edge if they're not adjacent
                if abs(last_pt.x - first_pt.x) > 1:
                    pass  # Will naturally close
            elif abs(last_pt.x - first_pt.x) > 1 or abs(last_pt.y - first_pt.y) > 1:
                # Need to close via bridge line
                if abs(last_pt.y - bridge_y) > 1:
                    points.append(Point(last_pt.x, bridge_y, PointType.ON_CURVE))
                if abs(first_pt.y - bridge_y) > 1:
                    points.append(Point(first_pt.x, bridge_y, PointType.ON_CURVE))

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

    For TOP piece: collect ALL segments with Y >= bridge_y
    For BOTTOM piece: collect ALL segments with Y <= bridge_y

    This traverses the ENTIRE contour to collect all segments on the correct
    side, then connects them along the bridge line. This handles complex
    shapes like R that have multiple disconnected segments on each side.

    Args:
        inner: The contour to split
        bridge_y: Y position of the bridge line
        inner_crossings: All crossings of the contour with the bridge line
        is_top: If True, return the top portion; otherwise bottom portion
        target_winding: Desired winding direction. If None, defaults to CCW (hole).
    """
    if target_winding is None:
        target_winding = WindingDirection.COUNTER_CLOCKWISE
    try:
        if len(inner_crossings) < 2:
            return None

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

        # Traverse the ENTIRE contour to collect ALL segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []
        was_on_correct = on_correct_side(inner_points[0].y)
        last_point = inner_points[0]

        # Start from the first point
        if was_on_correct:
            current_segment = [Point(last_point.x, last_point.y, last_point.point_type)]

        for i in range(1, n_inner):
            p = inner_points[i]
            curr_on_correct = on_correct_side(p.y)

            if curr_on_correct:
                if not was_on_correct:
                    # Crossed from wrong side to correct side
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment = [Point(ix, iy, PointType.ON_CURVE)]
                current_segment.append(Point(p.x, p.y, p.point_type))
            else:
                if was_on_correct and current_segment:
                    # Crossed from correct side to wrong side
                    ix, iy = intersect_bridge(last_point, p)
                    current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                    segments.append(current_segment)
                    current_segment = []

            was_on_correct = curr_on_correct
            last_point = p

        # Handle wraparound: check edge from last point to first point
        first_on_correct = on_correct_side(inner_points[0].y)
        if was_on_correct and not first_on_correct:
            # Last segment needs to be closed
            ix, iy = intersect_bridge(last_point, inner_points[0])
            current_segment.append(Point(ix, iy, PointType.ON_CURVE))
            segments.append(current_segment)
        elif was_on_correct and first_on_correct and current_segment:
            # Merge with first segment if both are on correct side
            if segments and abs(segments[0][0].y - bridge_y) < 1:
                # First segment starts with a crossing - extend it
                segments[0] = current_segment + segments[0]
            else:
                # First segment starts with an actual point - merge
                segments[0] = current_segment + segments[0]
        elif was_on_correct and current_segment:
            segments.append(current_segment)
        elif not was_on_correct and first_on_correct:
            # Crossing into first segment
            ix, iy = intersect_bridge(last_point, inner_points[0])
            if segments:
                segments[0].insert(0, Point(ix, iy, PointType.ON_CURVE))

        if not segments:
            return None

        # Sort segments by X position for proper connection
        # For inner (hole), trace in opposite direction from outer
        if is_top:
            segments.sort(key=lambda seg: max(pt.x for pt in seg), reverse=True)
        else:
            segments.sort(key=lambda seg: min(pt.x for pt in seg))

        # Build final contour by connecting segments along bridge line
        # Each segment's endpoints on bridge_y need connecting edges to form closed contour
        points: list[Point] = []
        for i, seg in enumerate(segments):
            if i > 0:
                # Add connecting edge along bridge line from previous segment to this one
                prev_end = points[-1]
                seg_start = seg[0]
                # Both should be on bridge_y - add horizontal edge
                if abs(prev_end.y - bridge_y) < 1 and abs(seg_start.y - bridge_y) < 1:
                    # Already on bridge line - just add the segment
                    pass
                else:
                    # Need to connect via bridge line if gap exists
                    if abs(prev_end.x - seg_start.x) > 1:
                        # Add intermediate point on bridge line
                        points.append(Point(prev_end.x, bridge_y, PointType.ON_CURVE))
                        points.append(Point(seg_start.x, bridge_y, PointType.ON_CURVE))
            points.extend(seg)

        # Close the contour by connecting last segment back to first along bridge line
        if points and segments:
            last_pt = points[-1]
            first_pt = points[0]
            if abs(last_pt.y - bridge_y) < 1 and abs(first_pt.y - bridge_y) < 1:
                # Both on bridge - add closing edge if they're not adjacent
                if abs(last_pt.x - first_pt.x) > 1:
                    pass  # Will naturally close
            elif abs(last_pt.x - first_pt.x) > 1 or abs(last_pt.y - first_pt.y) > 1:
                # Need to close via bridge line
                if abs(last_pt.y - bridge_y) > 1:
                    points.append(Point(last_pt.x, bridge_y, PointType.ON_CURVE))
                if abs(first_pt.y - bridge_y) > 1:
                    points.append(Point(first_pt.x, bridge_y, PointType.ON_CURVE))

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
