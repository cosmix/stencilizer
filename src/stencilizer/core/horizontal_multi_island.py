"""Horizontal multi-island bridge operations.

This module handles glyphs with multiple horizontally-arranged islands
(like 'Phi' φ) using spanning horizontal bridges.
"""

from stencilizer.core.geometry import (
    compute_winding_direction,
    detect_traversal_direction,
    find_all_edge_crossings,
    find_edge_crossing,
    signed_area,
)
from stencilizer.core.horizontal_bridge import (
    build_inner_portion_horizontal,
)
from stencilizer.domain import Contour, Point, PointType, WindingDirection


def classify_obstruction_horizontal(
    contour: Contour,
    outer: Contour,
    _gap_left: float,
    _gap_right: float,
) -> str:
    """Classify obstruction type: 'structural', 'island', or 'unknown'.

    Structural elements (like vertical bars) have the same winding as outer
    and span most of the height. These should not block spanning bridges.

    Args:
        contour: The contour to classify
        outer: The outer contour of the glyph
        _gap_left: Left X coordinate of the gap between islands (unused)
        _gap_right: Right X coordinate of the gap between islands (unused)

    Returns:
        'structural' if it's a structural element (like a vertical bar)
        'island' if it's likely an enclosed island
        'unknown' otherwise
    """
    contour_area = signed_area(contour.points)
    outer_area = signed_area(outer.points)

    # Same winding as outer = structural element (like vertical bar)
    same_winding = (contour_area < 0) == (outer_area < 0)

    if same_winding:
        bbox = contour.bounding_box()
        outer_bbox = outer.bounding_box()
        # If it spans most of the height, it's structural
        if bbox[1] <= outer_bbox[1] + 20 and bbox[3] >= outer_bbox[3] - 20:
            return "structural"

    # Opposite winding = probably an island
    if not same_winding:
        return "island"

    return "unknown"


def has_spanning_obstruction_horizontal(
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    bridge_bottom_y: float,
    bridge_top_y: float,
) -> bool:
    """Check if a contour blocks the horizontal bridge path between islands.

    This handles cases where a vertical element would block horizontal
    spanning bridges. Structural elements that have the same winding as
    the outer contour are not treated as obstructions.
    """
    if len(inners) < 2:
        return False

    # Sort inners by X (left to right)
    sorted_inners = sorted(inners, key=lambda c: c.bounding_box()[0])

    for i in range(len(sorted_inners) - 1):
        left_bbox = sorted_inners[i].bounding_box()
        right_bbox = sorted_inners[i + 1].bounding_box()
        gap_left = left_bbox[2]
        gap_right = right_bbox[0]

        for contour in all_contours:
            if contour is outer or contour in inners:
                continue

            bbox = contour.bounding_box()
            if bbox[1] <= bridge_top_y and bbox[3] >= bridge_bottom_y:
                if bbox[0] <= gap_left and bbox[2] >= gap_right:
                    # Classify the obstruction
                    obstruction_type = classify_obstruction_horizontal(contour, outer, gap_left, gap_right)
                    if obstruction_type != "structural":
                        return True
                if bbox[0] >= gap_left and bbox[2] <= gap_right:
                    # Classify the obstruction
                    obstruction_type = classify_obstruction_horizontal(contour, outer, gap_left, gap_right)
                    if obstruction_type != "structural":
                        return True

    return False


def build_outer_portion_multi_island_horizontal(
    outer: Contour,
    bridge_y: float,
    is_top: bool,
) -> Contour | None:
    """Build outer portion for multi-island horizontal bridge.

    Unlike the single-island version, this includes ALL parts of the outer
    contour above/below the bridge line (including stems between islands).

    Args:
        outer: The outer contour
        bridge_y: Y position of the bridge line
        is_top: If True, build the TOP portion; otherwise build BOTTOM

    Returns:
        The portion of outer above/below bridge_y, or None if failed
    """
    try:
        outer_crossings = find_all_edge_crossings(outer, bridge_y, False)
        if len(outer_crossings) < 2:
            return None

        # Sort crossings by X position
        sorted_crossings = sorted(outer_crossings, key=lambda c: c[2])

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

        # For TOP: start from rightmost crossing, traverse to collect all points above bridge_y
        # For BOTTOM: start from leftmost crossing, traverse to collect all points below bridge_y
        if is_top:
            # Start at rightmost crossing, go CCW (backward) to leftmost
            start_crossing = sorted_crossings[-1]
            end_crossing = sorted_crossings[0]
        else:
            # Start at leftmost crossing, go CW (forward) to rightmost
            start_crossing = sorted_crossings[0]
            end_crossing = sorted_crossings[-1]

        # Determine traversal direction
        outer_dir = detect_traversal_direction(
            outer_points, start_crossing[0], end_crossing[0],
            bridge_y, want_less_than=not is_top, is_x=False
        )

        # Build traversal list
        traverse_points: list[Point] = []
        if outer_dir == -1:
            idx = start_crossing[0]
            target = end_crossing[0]
            count = 0
            while idx != target and count < n_outer:
                traverse_points.append(outer_points[idx])
                idx = (idx - 1) % n_outer
                count += 1
        else:
            idx = (start_crossing[0] + 1) % n_outer
            target = (end_crossing[0] + 1) % n_outer
            count = 0
            while idx != target and count < n_outer:
                traverse_points.append(outer_points[idx])
                idx = (idx + 1) % n_outer
                count += 1

        # Collect all segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []

        start_pt = Point(start_crossing[2], bridge_y, PointType.ON_CURVE)
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
        end_pt = Point(end_crossing[2], bridge_y, PointType.ON_CURVE)
        if was_on_correct and current_segment:
            current_segment.append(end_pt)
            segments.append(current_segment)
        elif not was_on_correct:
            ix, iy = intersect_bridge(last_point, end_pt)
            if current_segment:
                current_segment.append(Point(ix, iy, PointType.ON_CURVE))
                segments.append(current_segment)

        if not segments:
            return None

        # Sort segments by X position
        if is_top:
            segments.sort(key=lambda seg: max(pt.x for pt in seg), reverse=True)
        else:
            segments.sort(key=lambda seg: min(pt.x for pt in seg))

        # Build final contour by connecting segments along bridge line
        points: list[Point] = []
        for i, seg in enumerate(segments):
            if i > 0:
                prev_end = points[-1]
                seg_start = seg[0]
                if abs(prev_end.x - seg_start.x) > 1:
                    # Add horizontal connection along bridge line
                    if abs(prev_end.y - bridge_y) > 1:
                        points.append(Point(prev_end.x, bridge_y, PointType.ON_CURVE))
                    points.append(Point(seg_start.x, bridge_y, PointType.ON_CURVE))
            points.extend(seg)

        # Close the contour
        if points and len(points) > 2:
            last_pt = points[-1]
            first_pt = points[0]
            if abs(last_pt.x - first_pt.x) > 1 or abs(last_pt.y - first_pt.y) > 1:
                if abs(last_pt.y - bridge_y) > 1:
                    points.append(Point(last_pt.x, bridge_y, PointType.ON_CURVE))
                if abs(first_pt.y - bridge_y) > 1 and abs(first_pt.x - points[-1].x) > 1:
                    points.append(Point(first_pt.x, bridge_y, PointType.ON_CURVE))

        # Remove duplicate consecutive points
        cleaned_points: list[Point] = []
        for p in points:
            if not cleaned_points or (
                abs(p.x - cleaned_points[-1].x) > 0.5 or
                abs(p.y - cleaned_points[-1].y) > 0.5
            ):
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


def merge_multi_island_horizontal(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
) -> list[Contour]:
    """Merge outer with multiple inner contours using a single horizontal cut.

    Creates a single horizontal line through all contours, splitting the glyph
    into TOP and BOTTOM pieces.

    Args:
        outer: The outer contour containing the islands
        inners: List of inner contours (islands) to bridge
        bridge_width: Width of the bridge gap
        all_contours: Optional list of all contours for obstruction checking

    Returns:
        List of new contours if successful, or original contours if not possible
    """
    if not inners:
        return [outer]

    # Compute bridge Y position from the COMMON (overlapping) Y range of all inners
    common_min_y = max(inner.bounding_box()[1] for inner in inners)
    common_max_y = min(inner.bounding_box()[3] for inner in inners)

    if common_max_y <= common_min_y:
        return [outer, *inners]

    center_y = (common_min_y + common_max_y) / 2.0
    available_height = common_max_y - common_min_y

    if bridge_width > available_height:
        bridge_width = available_height * 0.9

    half_width = bridge_width / 2.0
    bridge_bottom = center_y - half_width
    bridge_top = center_y + half_width

    # Check for obstructions
    if all_contours and has_spanning_obstruction_horizontal(
        outer, inners, all_contours, bridge_bottom, bridge_top
    ):
        return [outer, *inners]

    all_inner_min_x = min(inner.bounding_box()[0] for inner in inners)
    all_inner_max_x = max(inner.bounding_box()[2] for inner in inners)

    inners_sorted = sorted(inners, key=lambda c: c.bounding_box()[0])

    try:
        outer_top_left = find_edge_crossing(
            outer, bridge_top, False, constraint_min=all_inner_max_x
        )
        outer_top_right = find_edge_crossing(
            outer, bridge_top, False, constraint_max=all_inner_min_x
        )
        outer_bot_left = find_edge_crossing(
            outer, bridge_bottom, False, constraint_min=all_inner_max_x
        )
        outer_bot_right = find_edge_crossing(
            outer, bridge_bottom, False, constraint_max=all_inner_min_x
        )

        if not all([outer_top_left, outer_top_right, outer_bot_left, outer_bot_right]):
            return [outer, *inners]

        # Find crossings for each inner
        inner_crossings = []
        for inner in inners_sorted:
            inner_bbox = inner.bounding_box()
            inner_mid_x = (inner_bbox[0] + inner_bbox[2]) / 2

            top_left = find_edge_crossing(
                inner, bridge_top, False, constraint_max=inner_mid_x, pick_extreme=True
            )
            top_right = find_edge_crossing(
                inner, bridge_top, False, constraint_min=inner_mid_x, pick_extreme=True
            )
            bot_left = find_edge_crossing(
                inner, bridge_bottom, False, constraint_max=inner_mid_x, pick_extreme=True
            )
            bot_right = find_edge_crossing(
                inner, bridge_bottom, False, constraint_min=inner_mid_x, pick_extreme=True
            )

            if not all([top_left, top_right, bot_left, bot_right]):
                return [outer, *inners]

            inner_crossings.append(
                {
                    "inner": inner,
                    "top_left": top_left,
                    "top_right": top_right,
                    "bot_left": bot_left,
                    "bot_right": bot_right,
                    "min_x": inner_bbox[0],
                    "max_x": inner_bbox[2],
                }
            )

        # Build TOP piece
        result = []

        # Use the multi-island version that includes stems between islands
        top_outer = build_outer_portion_multi_island_horizontal(
            outer, bridge_top, is_top=True
        )
        if top_outer:
            result.append(top_outer)

        for crossing_data in inner_crossings:
            inner = crossing_data["inner"]
            inner_top_crossings = find_all_edge_crossings(inner, bridge_top, False)
            top_inner = build_inner_portion_horizontal(
                inner, bridge_top, inner_top_crossings, is_top=True
            )
            if top_inner:
                result.append(top_inner)

        # Build BOTTOM piece
        bot_outer = build_outer_portion_multi_island_horizontal(
            outer, bridge_bottom, is_top=False
        )
        if bot_outer:
            result.append(bot_outer)

        for crossing_data in inner_crossings:
            inner = crossing_data["inner"]
            inner_bot_crossings = find_all_edge_crossings(inner, bridge_bottom, False)
            bot_inner = build_inner_portion_horizontal(
                inner, bridge_bottom, inner_bot_crossings, is_top=False
            )
            if bot_inner:
                result.append(bot_inner)

        # Also split any NESTED contours that cross the bridge lines
        if all_contours:
            processed_set = {id(outer)} | {id(inner) for inner in inners}
            outer_bbox = outer.bounding_box()

            for contour in all_contours:
                if id(contour) in processed_set:
                    continue

                c_bbox = contour.bounding_box()
                c_center_x = (c_bbox[0] + c_bbox[2]) / 2
                c_center_y = (c_bbox[1] + c_bbox[3]) / 2

                if not (
                    outer_bbox[0] < c_center_x < outer_bbox[2]
                    and outer_bbox[1] < c_center_y < outer_bbox[3]
                ):
                    continue

                # Skip grandchildren: contours that are inside ANOTHER filled contour
                # that's also inside any of the holes. They should be processed with their direct parent.
                is_grandchild = False
                for other in all_contours:
                    if other is contour or other is outer or other in inners:
                        continue
                    other_area = signed_area(other.points)
                    if other_area < 0:  # CW = filled (potential parent)
                        other_bbox = other.bounding_box()
                        # Check if contour's center is inside this other filled contour
                        if (other_bbox[0] < c_center_x < other_bbox[2] and
                            other_bbox[1] < c_center_y < other_bbox[3]):
                            # This other filled contour must also be inside one of the holes
                            other_center_x = (other_bbox[0] + other_bbox[2]) / 2
                            other_center_y = (other_bbox[1] + other_bbox[3]) / 2
                            for inner in inners:
                                inner_bbox = inner.bounding_box()
                                if (inner_bbox[0] < other_center_x < inner_bbox[2] and
                                    inner_bbox[1] < other_center_y < inner_bbox[3]):
                                    is_grandchild = True
                                    break
                            if is_grandchild:
                                break
                if is_grandchild:
                    continue

                # Check if this is a filled contour (CW) ENTIRELY INSIDE any inner hole.
                # Such contours are self-contained geometry that should be preserved.
                is_filled_contour = signed_area(contour.points) < 0  # CW = filled
                entirely_inside_any_hole = False
                for inner in inners:
                    inner_bbox = inner.bounding_box()
                    if (c_bbox[0] >= inner_bbox[0] and c_bbox[2] <= inner_bbox[2] and
                        c_bbox[1] >= inner_bbox[1] and c_bbox[3] <= inner_bbox[3]):
                        entirely_inside_any_hole = True
                        break

                if is_filled_contour and entirely_inside_any_hole:
                    # Check if this nested outer has its own holes (like P in ℗ or R in ®).
                    # If so, skip it here - it will be processed in nested_outers section.
                    has_own_holes = False
                    if all_contours:
                        for other in all_contours:
                            if other is contour or other is outer or other in inners:
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

                # Check if this is a structural element (like a vertical bar)
                # Structural elements should NOT be split - add unchanged
                obs_type = classify_obstruction_horizontal(contour, outer, all_inner_min_x, all_inner_max_x)
                if obs_type == "structural":
                    result.append(contour)
                    if processed_nested is not None:
                        processed_nested.append(contour)
                    continue

                top_crossings = find_all_edge_crossings(contour, bridge_top, False)
                bot_crossings = find_all_edge_crossings(contour, bridge_bottom, False)

                # Determine target winding based on original contour type
                is_outer_contour = signed_area(contour.points) < 0
                target_winding = WindingDirection.CLOCKWISE if is_outer_contour else WindingDirection.COUNTER_CLOCKWISE

                if not top_crossings and not bot_crossings:
                    # Contour doesn't cross either bridge - add if outside gap
                    if c_bbox[3] <= bridge_bottom or c_bbox[1] >= bridge_top:
                        result.append(contour)
                    continue

                # Handle contours that cross only one bridge line
                if top_crossings and not bot_crossings:
                    if c_bbox[1] >= bridge_top:
                        result.append(contour)
                    else:
                        top_nested_result = build_inner_portion_horizontal(
                            contour, bridge_top, top_crossings, is_top=True, target_winding=target_winding
                        )
                        if top_nested_result:
                            result.append(top_nested_result)
                    continue

                if bot_crossings and not top_crossings:
                    if c_bbox[3] <= bridge_bottom:
                        result.append(contour)
                    else:
                        bot_nested_result = build_inner_portion_horizontal(
                            contour, bridge_bottom, bot_crossings, is_top=False, target_winding=target_winding
                        )
                        if bot_nested_result:
                            result.append(bot_nested_result)
                    continue

                # Contour crosses both bridges - split into top and bottom portions
                top_nested_result = build_inner_portion_horizontal(
                    contour, bridge_top, top_crossings, is_top=True, target_winding=target_winding
                )
                bot_nested_result = build_inner_portion_horizontal(
                    contour, bridge_bottom, bot_crossings, is_top=False, target_winding=target_winding
                )

                if top_nested_result:
                    result.append(top_nested_result)
                if bot_nested_result:
                    result.append(bot_nested_result)

        if len(result) >= 4:
            return result
        else:
            return [outer, *inners]

    except Exception:
        return [outer, *inners]
