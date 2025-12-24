"""Multi-island bridge operations.

This module handles glyphs with multiple islands (like '8', 'B', 'Phi')
using spanning vertical bridges.
"""

import logging

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

logger = logging.getLogger(__name__)


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
                    # Classify the obstruction - pass bridge coordinates for better detection
                    obstruction_type = classify_obstruction(
                        contour, outer, gap_bottom, gap_top,
                        bridge_left=bridge_left_x, bridge_right=bridge_right_x
                    )
                    if obstruction_type != "structural":
                        return True
                if bbox[1] >= gap_bottom and bbox[3] <= gap_top:
                    # Classify the obstruction
                    obstruction_type = classify_obstruction(
                        contour, outer, gap_bottom, gap_top,
                        bridge_left=bridge_left_x, bridge_right=bridge_right_x
                    )
                    if obstruction_type != "structural":
                        return True

    return False


def has_horizontal_bar_gap(
    inners: list[Contour],
    outer: Contour | None = None,
    min_gap: float = 100,
    min_bar_aspect: float = 2.0,
) -> bool:
    """Check if there's a horizontal bar gap between vertically-stacked islands.

    When two holes have a Y gap between them AND the gap forms a true horizontal
    bar (wider than tall), no bridge is needed because the bar already connects
    the left and right halves.

    This distinguishes true horizontal bars (like U+0472 Theta) from:
    - Vertical connectors (like 'g' which has a tall narrow connector)
    - Waists (like '8' which needs spanning bridges)

    Args:
        inners: List of inner contours (islands/holes)
        outer: The outer contour (used to check for bar geometry)
        min_gap: Minimum Y gap size to be considered a structural bar
        min_bar_aspect: Minimum width/height aspect ratio for a true bar (default 2.0)

    Returns:
        True if there's a horizontal bar gap between islands
    """
    if len(inners) < 2:
        return False

    # Sort by Y (bottom to top)
    sorted_inners = sorted(inners, key=lambda c: c.bounding_box()[1])

    for i in range(len(sorted_inners) - 1):
        lower_bbox = sorted_inners[i].bounding_box()
        upper_bbox = sorted_inners[i + 1].bounding_box()

        # Check if there's a Y gap (no overlap)
        gap_height = upper_bbox[1] - lower_bbox[3]  # upper_min_y - lower_max_y

        if gap_height < min_gap:
            continue

        # Check if they overlap in X (so the gap could form a bar)
        x_overlap = min(lower_bbox[2], upper_bbox[2]) - max(lower_bbox[0], upper_bbox[0])

        if x_overlap <= 0:
            continue

        # Check aspect ratio: a horizontal bar should be wider than tall
        # - g has aspect ~0.56 (taller than wide) - vertical connector, NOT a bar
        # - 8 has aspect ~1.44 (roughly square) - waist, needs spanning bridges
        # - U+0472 has aspect ~3.31 (much wider than tall) - TRUE horizontal bar
        gap_aspect = x_overlap / gap_height
        if gap_aspect < min_bar_aspect:
            # Gap is not wide enough to be a horizontal bar
            continue

        # The gap exists with proper aspect. Check if outer spans across.
        if outer is None:
            # Without outer, assume it's a bar (backward compatibility)
            return True

        gap_min_y = lower_bbox[3]
        gap_max_y = upper_bbox[1]
        holes_left_x = min(lower_bbox[0], upper_bbox[0])
        holes_right_x = max(lower_bbox[2], upper_bbox[2])

        # Check if outer has points spanning across the gap on BOTH sides
        # Also track X positions at different Y levels to detect diagonal strokes
        left_points_in_gap: list[tuple[float, float]] = []  # (x, y) for left side
        right_points_in_gap: list[tuple[float, float]] = []  # (x, y) for right side
        for p in outer.points:
            if gap_min_y <= p.y <= gap_max_y:
                if p.x < holes_left_x:
                    left_points_in_gap.append((p.x, p.y))
                if p.x > holes_right_x:
                    right_points_in_gap.append((p.x, p.y))

        # Only a bar if outer spans from left to right across the gap
        if not left_points_in_gap or not right_points_in_gap:
            continue

        # Check for diagonal stroke: if X position varies significantly with Y,
        # it's a diagonal stroke (like Ø), not a horizontal bar (like Θ)
        # A horizontal bar has consistent X edges; a diagonal has X that changes with Y
        #
        # Key insight: A circle's curve has SOME X variation, but less than the gap height.
        # A diagonal stroke has X variation roughly equal to or greater than Y span.
        # Use ratio of X_variation / gap_height to distinguish.
        def is_diagonal_stroke(points: list[tuple[float, float]], gap_h: float) -> bool:
            if len(points) < 2:
                return False
            x_values = [p[0] for p in points]
            x_range = max(x_values) - min(x_values)
            # A diagonal stroke has X variation comparable to Y span (ratio > 0.5)
            # A circle's curve has much less X variation relative to gap height (ratio < 0.3)
            # Use 0.4 as threshold
            if gap_h <= 0:
                return False
            return (x_range / gap_h) > 0.4

        # If either edge shows diagonal characteristics, it's not a horizontal bar
        if is_diagonal_stroke(left_points_in_gap, gap_height) or is_diagonal_stroke(right_points_in_gap, gap_height):
            continue

        # It's a true horizontal bar
        return True

    return False


def merge_multi_island_vertical(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
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
        logger.debug(
            "Multi-island merge failed: no common X overlap (common_min_x=%.1f, common_max_x=%.1f)",
            common_min_x, common_max_x
        )
        return [outer, *inners]

    center_x = (common_min_x + common_max_x) / 2.0
    available_width = common_max_x - common_min_x

    if bridge_width > available_width:
        logger.debug(
            "Bridge width (%.1f) exceeds available width (%.1f), clamping to %.1f",
            bridge_width, available_width, available_width * 0.9
        )
        bridge_width = available_width * 0.9

    half_width = bridge_width / 2.0
    bridge_left = center_x - half_width
    bridge_right = center_x + half_width

    # Check for obstructions
    if all_contours and has_spanning_obstruction(
        outer, inners, all_contours, bridge_left, bridge_right
    ):
        logger.debug(
            "Multi-island merge aborted: spanning obstruction detected at x=[%.1f, %.1f]",
            bridge_left, bridge_right
        )
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
            missing = []
            if not outer_top_left:
                missing.append("outer_top_left")
            if not outer_top_right:
                missing.append("outer_top_right")
            if not outer_bot_left:
                missing.append("outer_bot_left")
            if not outer_bot_right:
                missing.append("outer_bot_right")
            logger.debug(
                "Multi-island merge failed: missing outer crossings %s",
                missing
            )
            return [outer, *inners]

        # Find crossings for each inner
        # NOTE: We use actual crossing Y positions to determine top/bottom,
        # NOT the bounding box midpoint. This is critical for diagonal islands
        # (like slashed zeros) where crossings at a given X may all be on one
        # side of the bounding box center.
        inner_crossings = []
        for inner in inners_sorted:
            inner_bbox = inner.bounding_box()

            # Get ALL crossings for each bridge line
            left_all = find_all_edge_crossings(inner, bridge_left, True)
            right_all = find_all_edge_crossings(inner, bridge_right, True)

            # Need at least 2 crossings on each side
            if len(left_all) < 2 or len(right_all) < 2:
                logger.debug(
                    "Multi-island merge failed: insufficient crossings for island at y=[%.1f, %.1f] "
                    "(left=%d, right=%d)",
                    inner_bbox[1], inner_bbox[3], len(left_all), len(right_all)
                )
                return [outer, *inners]

            # Sort by Y and take extremes - this works for diagonal islands
            # where crossings may be clustered on one side of bbox center
            left_sorted = sorted(left_all, key=lambda c: c[2])  # Sort by Y ascending
            right_sorted = sorted(right_all, key=lambda c: c[2])

            # Bottom is lowest Y, top is highest Y
            left_bot = left_sorted[0]
            left_top = left_sorted[-1]
            right_bot = right_sorted[0]
            right_top = right_sorted[-1]

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
        left_inners_built = 0
        right_inners_built = 0
        num_inners = len(inner_crossings)

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
                left_inners_built += 1

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
                right_inners_built += 1

        # Verify ALL core portions were built - if any inner failed, fall back
        # We need: left_outer, right_outer, and both left+right for EVERY inner
        if not left_outer or not right_outer:
            logger.debug(
                "Multi-island merge failed: outer portion build failed (left=%s, right=%s)",
                left_outer is not None, right_outer is not None
            )
            return [outer, *inners]
        if left_inners_built != num_inners or right_inners_built != num_inners:
            logger.debug(
                "Multi-island merge failed: inner portions incomplete "
                "(left=%d/%d, right=%d/%d)",
                left_inners_built, num_inners, right_inners_built, num_inners
            )
            return [outer, *inners]

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

                    if has_own_holes:
                        # Has children - skip it here, will be processed in nested_outers section
                        continue
                    # Childless nested outer (inverted island) - fall through to crossing check
                    # to see if it needs to be split. If it crosses the bridge, it will be split.
                    # If it doesn't cross, it will be preserved and handled in nested_outers section.

                # Also check if this is a structural element in the GAP between islands
                # (like a horizontal bar connecting left and right halves)
                spans_bridge_horizontally = c_bbox[0] < bridge_left and c_bbox[2] > bridge_right
                in_gap_between_islands = True
                for inner in inners:
                    inner_bbox = inner.bounding_box()
                    if (inner_bbox[0] < c_center_x < inner_bbox[2] and
                        inner_bbox[1] < c_center_y < inner_bbox[3]):
                        in_gap_between_islands = False
                        break

                if is_filled_contour and spans_bridge_horizontally and in_gap_between_islands:
                    # Structural element in gap between islands - preserve unchanged
                    result.append(contour)
                    if processed_nested is not None:
                        processed_nested.append(contour)
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

    except Exception as e:
        logger.debug(
            "Multi-island merge failed with exception: %s",
            str(e)
        )
        return [outer, *inners]
