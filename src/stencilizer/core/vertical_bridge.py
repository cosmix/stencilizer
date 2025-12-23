"""Vertical bridge contour operations.

This module handles creating contours with vertical bridges (TOP/BOTTOM gaps),
which splits glyphs into LEFT/RIGHT pieces.
"""

from stencilizer.core.geometry import (
    compute_winding_direction,
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

    For LEFT piece: collect ALL segments with X <= bridge_x
    For RIGHT piece: collect ALL segments with X >= bridge_x

    This traverses the ENTIRE contour to collect all segments on the correct
    side, then connects them along the bridge line. This handles complex
    shapes like @ that have multiple disconnected segments on each side.
    """
    try:
        outer_above = [c for c in outer_crossings if c[2] > inner_max_y]
        outer_below = [c for c in outer_crossings if c[2] < inner_min_y]

        if not outer_above or not outer_below:
            return None

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

        # Traverse the ENTIRE contour to collect ALL segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []
        was_on_correct = on_correct_side(outer_points[0].x)
        last_point = outer_points[0]

        # Start from the first point
        if was_on_correct:
            current_segment = [Point(last_point.x, last_point.y, last_point.point_type)]

        for i in range(1, n_outer):
            p = outer_points[i]
            curr_on_correct = on_correct_side(p.x)

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
        first_on_correct = on_correct_side(outer_points[0].x)
        if was_on_correct and not first_on_correct:
            # Last segment needs to be closed
            ix, iy = intersect_bridge(last_point, outer_points[0])
            current_segment.append(Point(ix, iy, PointType.ON_CURVE))
            segments.append(current_segment)
        elif was_on_correct and first_on_correct and current_segment:
            # Merge with first segment if both are on correct side
            if segments and segments[0][0].x == bridge_x:
                # First segment starts with a crossing - extend it
                segments[0] = current_segment + segments[0]
            else:
                # First segment starts with an actual point - merge
                segments[0] = current_segment + segments[0]
        elif was_on_correct and current_segment:
            segments.append(current_segment)
        elif not was_on_correct and first_on_correct:
            # Crossing into first segment
            ix, iy = intersect_bridge(last_point, outer_points[0])
            if segments:
                segments[0].insert(0, Point(ix, iy, PointType.ON_CURVE))

        if not segments:
            return None

        # Sort segments by Y position for proper connection
        # For LEFT: connect from bottom to top along the bridge line
        # For RIGHT: connect from top to bottom along the bridge line
        if is_left:
            segments.sort(key=lambda seg: min(pt.y for pt in seg))
        else:
            segments.sort(key=lambda seg: max(pt.y for pt in seg), reverse=True)

        # Build final contour by connecting segments along bridge line
        # Each segment's endpoints on bridge_x need connecting edges to form closed contour
        points: list[Point] = []
        for i, seg in enumerate(segments):
            if i > 0:
                # Add connecting edge along bridge line from previous segment to this one
                prev_end = points[-1]
                seg_start = seg[0]
                # Both should be on bridge_x - add vertical edge
                if abs(prev_end.x - bridge_x) < 1 and abs(seg_start.x - bridge_x) < 1:
                    # Already on bridge line - just add the segment
                    pass
                else:
                    # Need to connect via bridge line if gap exists
                    if abs(prev_end.y - seg_start.y) > 1:
                        # Add intermediate point on bridge line
                        points.append(Point(bridge_x, prev_end.y, PointType.ON_CURVE))
                        points.append(Point(bridge_x, seg_start.y, PointType.ON_CURVE))
            points.extend(seg)

        # Close the contour by connecting last segment back to first along bridge line
        if points and segments:
            last_pt = points[-1]
            first_pt = points[0]
            if abs(last_pt.x - bridge_x) < 1 and abs(first_pt.x - bridge_x) < 1:
                # Both on bridge - add closing edge if they're not adjacent
                if abs(last_pt.y - first_pt.y) > 1:
                    pass  # Will naturally close
            elif abs(last_pt.y - first_pt.y) > 1 or abs(last_pt.x - first_pt.x) > 1:
                # Need to close via bridge line
                if abs(last_pt.x - bridge_x) > 1:
                    points.append(Point(bridge_x, last_pt.y, PointType.ON_CURVE))
                if abs(first_pt.x - bridge_x) > 1:
                    points.append(Point(bridge_x, first_pt.y, PointType.ON_CURVE))

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


def _build_contour_from_segments_vertical(
    segments: list[list[Point]],
    bridge_x: float,
) -> list[Point] | None:
    """Build a contour from segments by connecting them along the bridge line.

    Returns the raw points list (not cleaned or winding-enforced).
    """
    if not segments:
        return None

    points: list[Point] = []
    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = points[-1]
            seg_start = seg[0]
            if abs(prev_end.x - bridge_x) < 1 and abs(seg_start.x - bridge_x) < 1:
                pass  # Already on bridge line
            else:
                if abs(prev_end.y - seg_start.y) > 1:
                    points.append(Point(bridge_x, prev_end.y, PointType.ON_CURVE))
                    points.append(Point(bridge_x, seg_start.y, PointType.ON_CURVE))
        points.extend(seg)

    # Close the contour
    if points and segments:
        last_pt = points[-1]
        first_pt = points[0]
        if abs(last_pt.x - bridge_x) < 1 and abs(first_pt.x - bridge_x) < 1:
            if abs(last_pt.y - first_pt.y) > 1:
                pass  # Will naturally close
        elif abs(last_pt.y - first_pt.y) > 1 or abs(last_pt.x - first_pt.x) > 1:
            if abs(last_pt.x - bridge_x) > 1:
                points.append(Point(bridge_x, last_pt.y, PointType.ON_CURVE))
            if abs(first_pt.x - bridge_x) > 1:
                points.append(Point(bridge_x, first_pt.y, PointType.ON_CURVE))

    return points


def _clean_points_vertical(points: list[Point]) -> list[Point]:
    """Remove duplicate consecutive points."""
    cleaned: list[Point] = []
    for p in points:
        if not cleaned or (abs(p.x - cleaned[-1].x) > 0.5 or abs(p.y - cleaned[-1].y) > 0.5):
            cleaned.append(p)
    return cleaned


def build_inner_portion_vertical(
    inner: Contour,
    bridge_x: float,
    inner_crossings: list[tuple[int, float, float]],
    is_left: bool,
    target_winding: WindingDirection | None = None,
) -> Contour | None:
    """Build inner portion for vertical bridge.

    For LEFT piece: collect ALL segments with X <= bridge_x
    For RIGHT piece: collect ALL segments with X >= bridge_x

    This traverses the ENTIRE contour to collect all segments on the correct
    side, then connects them along the bridge line. This handles complex
    shapes like the R-defining contour in ® that have multiple disconnected
    segments on each side.

    Uses a multi-ordering strategy: tries different segment orderings and picks
    the one that naturally produces the correct winding direction (without
    needing reversal), as this indicates geometrically correct construction.

    Args:
        inner: The contour to split
        bridge_x: X position of the bridge line
        inner_crossings: All crossings of the contour with the bridge line
        is_left: If True, return the left portion; otherwise right portion
        target_winding: Desired winding direction. If None, defaults to CCW (hole).
    """
    if target_winding is None:
        target_winding = WindingDirection.COUNTER_CLOCKWISE
    try:
        if len(inner_crossings) < 2:
            return None

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

        # Traverse the ENTIRE contour to collect ALL segments on the correct side
        segments: list[list[Point]] = []
        current_segment: list[Point] = []
        was_on_correct = on_correct_side(inner_points[0].x)
        last_point = inner_points[0]

        # Start from the first point
        if was_on_correct:
            current_segment = [Point(last_point.x, last_point.y, last_point.point_type)]

        for i in range(1, n_inner):
            p = inner_points[i]
            curr_on_correct = on_correct_side(p.x)

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
        first_on_correct = on_correct_side(inner_points[0].x)
        if was_on_correct and not first_on_correct:
            # Last segment needs to be closed
            ix, iy = intersect_bridge(last_point, inner_points[0])
            current_segment.append(Point(ix, iy, PointType.ON_CURVE))
            segments.append(current_segment)
        elif was_on_correct and first_on_correct and current_segment:
            # Merge with first segment if both are on correct side
            if segments and segments[0][0].x == bridge_x:
                segments[0] = current_segment + segments[0]
            else:
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

        # For single segment, no ordering ambiguity
        if len(segments) == 1:
            points = _build_contour_from_segments_vertical(segments, bridge_x)
            if not points:
                return None
            cleaned_points = _clean_points_vertical(points)
            if len(cleaned_points) < 3:
                return None
            direction = compute_winding_direction(cleaned_points)
            if direction != target_winding:
                cleaned_points = list(reversed(cleaned_points))
                direction = target_winding
            return Contour(points=cleaned_points, direction=direction)

        # Multi-segment case: try different orderings and pick the one that
        # NATURALLY produces correct winding (indicates geometrically correct)
        orderings_to_try = [
            # Original traversal order (segments as collected)
            segments[:],
            # Reversed traversal order
            segments[::-1],
            # Bottom-to-top sorted
            sorted(segments, key=lambda seg: min(pt.y for pt in seg)),
            # Top-to-bottom sorted
            sorted(segments, key=lambda seg: max(pt.y for pt in seg), reverse=True),
        ]

        best_result: list[Point] | None = None
        best_needed_reversal = True
        best_area = 0.0

        for ordered_segments in orderings_to_try:
            points = _build_contour_from_segments_vertical(ordered_segments, bridge_x)
            if not points:
                continue
            cleaned = _clean_points_vertical(points)
            if len(cleaned) < 3:
                continue

            direction = compute_winding_direction(cleaned)
            needed_reversal = (direction != target_winding)
            area = abs(signed_area(cleaned))

            # Prefer orderings that naturally produce correct winding
            # Among those, prefer larger area (indicates non-self-intersecting)
            if best_result is None:
                best_result = cleaned
                best_needed_reversal = needed_reversal
                best_area = area
            elif not needed_reversal and best_needed_reversal:
                # This ordering naturally has correct winding - prefer it
                best_result = cleaned
                best_needed_reversal = needed_reversal
                best_area = area
            elif needed_reversal == best_needed_reversal and area > best_area:
                # Same reversal status but larger area
                best_result = cleaned
                best_area = area

        if best_result is None:
            return None

        # Apply reversal if needed
        if best_needed_reversal:
            best_result = list(reversed(best_result))

        return Contour(points=best_result, direction=target_winding)

    except Exception:
        return None
