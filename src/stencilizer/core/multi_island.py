"""Multi-island bridge operations.

This module handles glyphs with multiple islands (like '8', 'B', 'Phi')
using spanning vertical bridges.
"""

from stencilizer.core.geometry import (
    find_all_edge_crossings,
    find_edge_crossing,
    signed_area,
)
from stencilizer.core.vertical_bridge import (
    build_inner_portion_vertical,
    build_outer_portion_vertical,
)

from stencilizer.domain import Contour, WindingDirection


def classify_obstruction(
    contour: Contour,
    outer: Contour,
    gap_bottom: float,
    gap_top: float,
    bridge_left: float | None = None,
    bridge_right: float | None = None,
) -> str:
    """Classify obstruction type: 'structural', 'island', or 'unknown'.

    Structural elements (like Theta's bar) have the same winding as outer
    and connect the left/right halves of the glyph. These should not be split.

    Args:
        contour: The contour to classify
        outer: The outer contour of the glyph
        gap_bottom: Bottom Y coordinate of the gap between islands
        gap_top: Top Y coordinate of the gap between islands
        bridge_left: Left X coordinate of the bridge (optional)
        bridge_right: Right X coordinate of the bridge (optional)

    Returns:
        'structural' if it's a structural element (like Theta's bar)
        'island' if it's likely an enclosed island
        'unknown' otherwise
    """
    contour_area = signed_area(contour.points)
    outer_area = signed_area(outer.points)

    # Same winding as outer = potential structural element
    same_winding = (contour_area < 0) == (outer_area < 0)

    if same_winding:
        bbox = contour.bounding_box()
        outer_bbox = outer.bounding_box()

        # Check if it spans across the bridge gap region (horizontal bar like Theta)
        # A structural bar should:
        # 1. Span across the bridge gap (left side to right side)
        # 2. Be within the vertical gap between islands
        if bridge_left is not None and bridge_right is not None:
            spans_bridge = bbox[0] < bridge_left and bbox[2] > bridge_right
            in_gap_region = bbox[1] <= gap_top and bbox[3] >= gap_bottom
            if spans_bridge and in_gap_region:
                return "structural"

        # Fallback: check if it spans most of the outer width
        if bbox[0] <= outer_bbox[0] + 20 and bbox[2] >= outer_bbox[2] - 20:
            return "structural"

    # Opposite winding = probably an island
    if not same_winding:
        return "island"

    return "unknown"


def has_spanning_obstruction(
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    bridge_left_x: float,
    bridge_right_x: float,
) -> bool:
    """Check if a contour blocks the vertical bridge path between islands.

    This handles cases like Greek Theta where a horizontal bar would
    block vertical spanning bridges. Structural elements (like Theta's bar)
    that have the same winding as the outer contour are not treated as
    obstructions.
    """
    if len(inners) < 2:
        return False

    # Sort inners by Y (bottom to top)
    sorted_inners = sorted(inners, key=lambda c: c.bounding_box()[1])

    for i in range(len(sorted_inners) - 1):
        lower_bbox = sorted_inners[i].bounding_box()
        upper_bbox = sorted_inners[i + 1].bounding_box()
        gap_bottom = lower_bbox[3]
        gap_top = upper_bbox[1]

        for contour in all_contours:
            if contour is outer or contour in inners:
                continue

            bbox = contour.bounding_box()
            if bbox[0] <= bridge_right_x and bbox[2] >= bridge_left_x:
                if bbox[1] <= gap_bottom and bbox[3] >= gap_top:
                    # Classify the obstruction
                    obstruction_type = classify_obstruction(contour, outer, gap_bottom, gap_top)
                    if obstruction_type != "structural":
                        return True
                if bbox[1] >= gap_bottom and bbox[3] <= gap_top:
                    # Classify the obstruction
                    obstruction_type = classify_obstruction(contour, outer, gap_bottom, gap_top)
                    if obstruction_type != "structural":
                        return True

    return False


def merge_multi_island_vertical(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    all_contours: list[Contour] | None = None,
) -> list[Contour]:
    """Merge outer with multiple inner contours using a single vertical cut.

    Creates a single vertical line through all contours, splitting the glyph
    into LEFT and RIGHT pieces.
    """
    if not inners:
        return [outer]

    # Compute bridge X position from the COMMON (overlapping) X range of all inners
    common_min_x = max(inner.bounding_box()[0] for inner in inners)
    common_max_x = min(inner.bounding_box()[2] for inner in inners)

    if common_max_x <= common_min_x:
        return [outer, *inners]

    center_x = (common_min_x + common_max_x) / 2.0
    available_width = common_max_x - common_min_x

    if bridge_width > available_width:
        bridge_width = available_width * 0.9

    half_width = bridge_width / 2.0
    bridge_left = center_x - half_width
    bridge_right = center_x + half_width

    # Check for obstructions
    if all_contours and has_spanning_obstruction(
        outer, inners, all_contours, bridge_left, bridge_right
    ):
        return [outer, *inners]

    all_inner_min_y = min(inner.bounding_box()[1] for inner in inners)
    all_inner_max_y = max(inner.bounding_box()[3] for inner in inners)

    inners_sorted = sorted(inners, key=lambda c: c.bounding_box()[1])

    try:
        outer_top_left = find_edge_crossing(
            outer, bridge_left, True, constraint_min=all_inner_max_y
        )
        outer_top_right = find_edge_crossing(
            outer, bridge_right, True, constraint_min=all_inner_max_y
        )
        outer_bot_left = find_edge_crossing(
            outer, bridge_left, True, constraint_max=all_inner_min_y
        )
        outer_bot_right = find_edge_crossing(
            outer, bridge_right, True, constraint_max=all_inner_min_y
        )

        if not all([outer_top_left, outer_top_right, outer_bot_left, outer_bot_right]):
            return [outer, *inners]

        # Find crossings for each inner
        inner_crossings = []
        for inner in inners_sorted:
            inner_bbox = inner.bounding_box()
            inner_mid_y = (inner_bbox[1] + inner_bbox[3]) / 2

            left_top = find_edge_crossing(
                inner, bridge_left, True, constraint_min=inner_mid_y, pick_extreme=True
            )
            left_bot = find_edge_crossing(
                inner, bridge_left, True, constraint_max=inner_mid_y, pick_extreme=True
            )
            right_top = find_edge_crossing(
                inner, bridge_right, True, constraint_min=inner_mid_y, pick_extreme=True
            )
            right_bot = find_edge_crossing(
                inner, bridge_right, True, constraint_max=inner_mid_y, pick_extreme=True
            )

            if not all([left_top, left_bot, right_top, right_bot]):
                return [outer, *inners]

            inner_crossings.append(
                {
                    "inner": inner,
                    "left_top": left_top,
                    "left_bot": left_bot,
                    "right_top": right_top,
                    "right_bot": right_bot,
                    "min_y": inner_bbox[1],
                    "max_y": inner_bbox[3],
                }
            )

        # Build LEFT piece
        result = []

        left_outer = build_outer_portion_vertical(
            outer,
            bridge_left,
            find_all_edge_crossings(outer, bridge_left, True),
            all_inner_min_y,
            all_inner_max_y,
            is_left=True,
        )
        if left_outer:
            result.append(left_outer)

        for crossing_data in inner_crossings:
            inner = crossing_data["inner"]
            inner_left_crossings = find_all_edge_crossings(inner, bridge_left, True)
            left_inner = build_inner_portion_vertical(
                inner, bridge_left, inner_left_crossings, is_left=True
            )
            if left_inner:
                result.append(left_inner)

        # Build RIGHT piece
        right_outer = build_outer_portion_vertical(
            outer,
            bridge_right,
            find_all_edge_crossings(outer, bridge_right, True),
            all_inner_min_y,
            all_inner_max_y,
            is_left=False,
        )
        if right_outer:
            result.append(right_outer)

        for crossing_data in inner_crossings:
            inner = crossing_data["inner"]
            inner_right_crossings = find_all_edge_crossings(inner, bridge_right, True)
            right_inner = build_inner_portion_vertical(
                inner, bridge_right, inner_right_crossings, is_left=False
            )
            if right_inner:
                result.append(right_inner)

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

                # Check if this is a structural element (like Theta's bar)
                # Structural elements should NOT be split - add unchanged
                obs_type = classify_obstruction(
                    contour, outer, all_inner_min_y, all_inner_max_y,
                    bridge_left=bridge_left, bridge_right=bridge_right
                )
                if obs_type == "structural":
                    result.append(contour)
                    continue

                left_crossings = find_all_edge_crossings(contour, bridge_left, True)
                right_crossings = find_all_edge_crossings(contour, bridge_right, True)

                # Determine target winding based on original contour type
                is_outer_contour = signed_area(contour.points) < 0
                target_winding = WindingDirection.CLOCKWISE if is_outer_contour else WindingDirection.COUNTER_CLOCKWISE

                if not left_crossings and not right_crossings:
                    # Contour doesn't cross either bridge - add if outside gap
                    if c_bbox[2] <= bridge_left or c_bbox[0] >= bridge_right:
                        result.append(contour)
                    continue

                # Handle contours that cross only one bridge line
                if left_crossings and not right_crossings:
                    if c_bbox[0] >= bridge_right:
                        result.append(contour)
                    else:
                        left_nested = build_inner_portion_vertical(
                            contour, bridge_left, left_crossings, is_left=True, target_winding=target_winding
                        )
                        if left_nested:
                            result.append(left_nested)
                    continue

                if right_crossings and not left_crossings:
                    if c_bbox[2] <= bridge_left:
                        result.append(contour)
                    else:
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

        if len(result) >= 4:
            return result
        else:
            return [outer, *inners]

    except Exception:
        return [outer, *inners]
