"""Horizontal multi-island bridge operations.

This module handles glyphs with multiple horizontally-arranged islands
(like 'Phi' φ) using spanning horizontal bridges.
"""

from stencilizer.core.geometry import (
    find_all_edge_crossings,
    find_edge_crossing,
    signed_area,
)
from stencilizer.core.horizontal_bridge import (
    build_inner_portion_horizontal,
    build_outer_portion_horizontal,
)
from stencilizer.domain import Contour, WindingDirection


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


def merge_multi_island_horizontal(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    all_contours: list[Contour] | None = None,
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

        top_outer = build_outer_portion_horizontal(
            outer,
            bridge_top,
            find_all_edge_crossings(outer, bridge_top, False),
            all_inner_min_x,
            all_inner_max_x,
            is_top=True,
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
        bot_outer = build_outer_portion_horizontal(
            outer,
            bridge_bottom,
            find_all_edge_crossings(outer, bridge_bottom, False),
            all_inner_min_x,
            all_inner_max_x,
            is_top=False,
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

                # Check if this is a structural element (like a vertical bar)
                # Structural elements should NOT be split - add unchanged
                obs_type = classify_obstruction_horizontal(contour, outer, all_inner_min_x, all_inner_max_x)
                if obs_type == "structural":
                    result.append(contour)
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
