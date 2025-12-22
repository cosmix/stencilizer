"""Contour surgery module for creating stencil bridges.

This module implements contour surgery by merging outer and inner contours
with bridge connections. Instead of adding hole contours (which create
artifacts in empty space), we modify the outer contour to have notches
that connect to the inner contour at bridge positions.

Key classes:
- ContourMerger: Modifies contours to create bridge gaps
- GlyphTransformer: Transforms glyphs with bridge gaps
"""

from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.domain import Contour, Glyph, Point, WindingDirection


class ContourMerger:
    """Creates bridge gaps by merging outer and inner contours.

    Instead of adding CCW hole contours (which create black artifacts in
    empty space), this class modifies the outer contour to have notches
    that reach the inner contour, creating clean bridge gaps.
    """

    def _compute_winding_direction(self, points: list[Point]) -> WindingDirection:
        """Compute winding direction from signed area of points.

        Uses the shoelace formula to calculate signed area. Positive area
        indicates clockwise winding, negative indicates counter-clockwise.

        Args:
            points: List of points forming a closed contour

        Returns:
            WindingDirection.CLOCKWISE or WindingDirection.COUNTER_CLOCKWISE
        """
        if len(points) < 3:
            return WindingDirection.CLOCKWISE

        # Shoelace formula for signed area
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        area /= 2.0

        # Positive area = clockwise, negative = counter-clockwise
        if area >= 0:
            return WindingDirection.CLOCKWISE
        else:
            return WindingDirection.COUNTER_CLOCKWISE

    def _compute_winding_at_point(
        self, x: float, y: float, contours: list[Contour]
    ) -> int:
        """Compute winding number at a point for a set of contours.

        Uses ray casting to count signed crossings.

        Args:
            x: X coordinate of point
            y: Y coordinate of point
            contours: List of contours to check

        Returns:
            Winding number (0 = outside, non-zero = inside)
        """
        winding = 0

        for contour in contours:
            points = contour.points
            n = len(points)

            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                # Check if edge crosses the horizontal ray from (x, y) going right
                if p1.y <= y < p2.y:  # Upward crossing
                    # Compute x-coordinate of intersection
                    t = (y - p1.y) / (p2.y - p1.y)
                    x_intersect = p1.x + t * (p2.x - p1.x)
                    if x < x_intersect:
                        winding += 1
                elif p2.y <= y < p1.y:  # Downward crossing
                    t = (y - p1.y) / (p2.y - p1.y)
                    x_intersect = p1.x + t * (p2.x - p1.x)
                    if x < x_intersect:
                        winding -= 1

        return winding

    def _is_bridge_path_clear(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        inner: Contour,
        outer: Contour,
        all_contours: list[Contour] | None = None,
    ) -> bool:
        """Check if a bridge path is clear of obstructions.

        A bridge path is blocked if it intersects any contour OTHER than
        the inner and outer contours being merged. This catches cases like
        Theta where the horizontal bar blocks horizontal bridges.

        Args:
            start_x, start_y: Start point (on inner contour)
            end_x, end_y: End point (on outer contour)
            inner: Inner contour (island)
            outer: Outer contour
            all_contours: All contours in glyph (for obstruction check)

        Returns:
            True if path is clear, False if obstructed
        """
        if all_contours is None:
            return True  # Can't check without all contours

        # Check each "other" contour (not inner or outer) for intersection
        for contour in all_contours:
            if contour is inner or contour is outer:
                continue  # Skip the contours we're merging

            # Check if the bridge line segment intersects this contour
            if self._line_intersects_contour(
                start_x, start_y, end_x, end_y, contour
            ):
                return False  # Blocked by this contour

        return True

    def _line_intersects_contour(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        contour: Contour,
    ) -> bool:
        """Check if a line segment intersects a contour.

        Args:
            x1, y1: Start point of line
            x2, y2: End point of line
            contour: Contour to check intersection with

        Returns:
            True if line intersects contour edges
        """
        points = contour.points
        n = len(points)

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]

            if self._segments_intersect(x1, y1, x2, y2, p1.x, p1.y, p2.x, p2.y):
                return True

        return False

    def _segments_intersect(
        self,
        ax1: float, ay1: float, ax2: float, ay2: float,
        bx1: float, by1: float, bx2: float, by2: float,
    ) -> bool:
        """Check if two line segments intersect.

        Uses cross product method to determine intersection.
        """
        def cross(o_x: float, o_y: float, a_x: float, a_y: float, b_x: float, b_y: float) -> float:
            return (a_x - o_x) * (b_y - o_y) - (a_y - o_y) * (b_x - o_x)

        d1 = cross(bx1, by1, bx2, by2, ax1, ay1)
        d2 = cross(bx1, by1, bx2, by2, ax2, ay2)
        d3 = cross(ax1, ay1, ax2, ay2, bx1, by1)
        d4 = cross(ax1, ay1, ax2, ay2, bx2, by2)

        # Check if segments straddle each other (strictly)
        # Use strict inequality to avoid false positives at endpoints
        return d1 * d2 < 0 and d3 * d4 < 0

    def _is_bridge_length_valid(
        self,
        inner_x: float,
        inner_y: float,
        outer_x: float,
        outer_y: float,
        max_length: float,
    ) -> bool:
        """Check if bridge length is within acceptable bounds.

        Args:
            inner_x, inner_y: Point on inner contour
            outer_x, outer_y: Point on outer contour
            max_length: Maximum acceptable bridge length

        Returns:
            True if length is acceptable, False if too long
        """
        import math
        length = math.hypot(outer_x - inner_x, outer_y - inner_y)
        return length <= max_length

    def find_contour_bounds(self, contour: Contour) -> tuple[float, float, float, float]:
        """Find bounding box of contour (min_x, min_y, max_x, max_y)."""
        xs = [p.x for p in contour.points]
        ys = [p.y for p in contour.points]
        return min(xs), min(ys), max(xs), max(ys)

    def find_edge_crossing(
        self,
        contour: Contour,
        coord: float,
        is_x: bool,
        constraint_min: float | None = None,
        constraint_max: float | None = None,
    ) -> tuple[int, float, float] | None:
        """Find where contour edge crosses a coordinate.

        Args:
            contour: The contour to search
            coord: The X or Y coordinate to find crossing at
            is_x: If True, find Y at given X; if False, find X at given Y
            constraint_min: Only return crossings where the other coord > this
            constraint_max: Only return crossings where the other coord < this

        Returns:
            (edge_index, crossing_coord, t_param) or None
        """
        points = contour.points
        n = len(points)
        best = None
        best_other = None

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]

            if is_x:
                # Looking for Y at given X
                c1, c2 = p1.x, p2.x
                o1, o2 = p1.y, p2.y
            else:
                # Looking for X at given Y
                c1, c2 = p1.y, p2.y
                o1, o2 = p1.x, p2.x

            # Check if this edge crosses the coordinate
            if not ((c1 <= coord <= c2) or (c2 <= coord <= c1)):
                continue

            if abs(c2 - c1) < 0.001:
                t = 0.5
                other = (o1 + o2) / 2
            else:
                t = (coord - c1) / (c2 - c1)
                other = o1 + t * (o2 - o1)

            # Apply constraints
            if constraint_min is not None and other <= constraint_min:
                continue
            if constraint_max is not None and other >= constraint_max:
                continue

            # Keep track of best crossing - we want the NEAREST valid crossing to the inner
            # For constraint_min (finding RIGHT/TOP): want SMALLEST other > constraint_min (nearest from right/top)
            # For constraint_max (finding LEFT/BOTTOM): want LARGEST other < constraint_max (nearest from left/bottom)
            if best is None:
                best = (i, other, t)
                best_other = other
            else:
                if constraint_min is not None and best_other is not None and other < best_other:
                    # Want smallest value > constraint_min (nearest to inner from right/top)
                    best = (i, other, t)
                    best_other = other
                elif constraint_max is not None and best_other is not None and other > best_other:
                    # Want largest value < constraint_max (nearest to inner from left/bottom)
                    best = (i, other, t)
                    best_other = other

        return best

    def merge_contours_with_bridges(
        self,
        inner: Contour,
        outer: Contour,
        bridge_width: float,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        all_contours: list[Contour] | None = None,
    ) -> list[Contour]:
        """Merge outer and inner contours with bridge gaps.

        Creates new contours that have bridge gaps built in by
        connecting segments of outer and inner contours.

        Args:
            inner: Inner contour (island/hole)
            outer: Outer contour (parent)
            bridge_width: Width of each bridge
            force_horizontal: If True, prefer horizontal bridges (TOP/BOTTOM pieces)
            force_vertical: If True, prefer vertical bridges (LEFT/RIGHT pieces)
            all_contours: All contours in the glyph (for obstruction checking)

        Returns:
            List of new contours (replacing both outer and inner)
        """
        inner_min_x, inner_min_y, inner_max_x, inner_max_y = self.find_contour_bounds(inner)
        outer_min_x, outer_min_y, outer_max_x, outer_max_y = self.find_contour_bounds(outer)

        min_stroke = 20
        inner_width = inner_max_x - inner_min_x
        inner_height = inner_max_y - inner_min_y
        # Use generous max_stroke limits to support bold fonts with thick strokes
        max_stroke_h = max(inner_width * 1.5, 400)
        max_stroke_v = max(inner_height * 1.5, 400)

        half_width = bridge_width / 2.0
        center_x = (inner_min_x + inner_max_x) / 2.0
        center_y = (inner_min_y + inner_max_y) / 2.0

        # Determine bridge orientation
        stroke_left = inner_min_x - outer_min_x
        stroke_right = outer_max_x - inner_max_x
        stroke_top = outer_max_y - inner_max_y
        stroke_bottom = inner_min_y - outer_min_y

        horizontal_stroke = min(stroke_left, stroke_right)
        vertical_stroke = min(stroke_top, stroke_bottom)

        # Maximum bridge length - bridges longer than this indicate
        # the path goes through complex geometry (stems, bars, etc.)
        # Use 3x the minimum stroke as threshold
        max_bridge_length = max(min(stroke_left, stroke_right, stroke_top, stroke_bottom) * 3, 400)

        # Check which orientations are geometrically possible
        can_vertical = (
            vertical_stroke >= min_stroke
            and stroke_top >= min_stroke
            and stroke_bottom >= min_stroke
            and vertical_stroke <= max_stroke_v
        )
        can_horizontal = (
            horizontal_stroke >= min_stroke
            and stroke_left >= min_stroke
            and stroke_right >= min_stroke
            and horizontal_stroke <= max_stroke_h
        )

        # Find ACTUAL crossing points for bridges and validate them
        # This is critical - we must use real crossing points, not bounding box!

        # Variables to store actual crossing points for later use
        h_outer_left_x: float | None = None
        h_outer_right_x: float | None = None
        v_outer_top_y: float | None = None
        v_outer_bottom_y: float | None = None

        # Validate horizontal bridges using actual crossings
        if can_horizontal:
            # Find where horizontal bridges would actually cross the outer contour
            outer_left_crossing = self.find_edge_crossing(
                outer, center_y, False, constraint_max=inner_min_x
            )
            outer_right_crossing = self.find_edge_crossing(
                outer, center_y, False, constraint_min=inner_max_x
            )

            if outer_left_crossing and outer_right_crossing:
                h_outer_left_x = outer_left_crossing[1]
                h_outer_right_x = outer_right_crossing[1]

                # Check bridge lengths
                left_bridge_len = inner_min_x - h_outer_left_x
                right_bridge_len = h_outer_right_x - inner_max_x

                if left_bridge_len > max_bridge_length or right_bridge_len > max_bridge_length:
                    can_horizontal = False
                elif left_bridge_len < 0 or right_bridge_len < 0:
                    # Crossing is on wrong side - invalid
                    can_horizontal = False
            else:
                can_horizontal = False

        # Validate vertical bridges using actual crossings
        if can_vertical:
            # Find where vertical bridges would actually cross the outer contour
            outer_top_crossing = self.find_edge_crossing(
                outer, center_x, True, constraint_min=inner_max_y
            )
            outer_bottom_crossing = self.find_edge_crossing(
                outer, center_x, True, constraint_max=inner_min_y
            )

            if outer_top_crossing and outer_bottom_crossing:
                v_outer_top_y = outer_top_crossing[1]
                v_outer_bottom_y = outer_bottom_crossing[1]

                # Check bridge lengths
                top_bridge_len = v_outer_top_y - inner_max_y
                bottom_bridge_len = inner_min_y - v_outer_bottom_y

                if top_bridge_len > max_bridge_length or bottom_bridge_len > max_bridge_length:
                    can_vertical = False
                elif top_bridge_len < 0 or bottom_bridge_len < 0:
                    # Crossing is on wrong side - invalid
                    can_vertical = False
            else:
                can_vertical = False

        # Check for obstructions using ACTUAL crossing points (not bounding box!)
        if all_contours and (can_horizontal or can_vertical):
            # For horizontal bridges, check if actual path is clear
            if can_horizontal and h_outer_left_x is not None and h_outer_right_x is not None:
                left_clear = self._is_bridge_path_clear(
                    inner_min_x, center_y, h_outer_left_x, center_y,
                    inner, outer, all_contours
                )
                right_clear = self._is_bridge_path_clear(
                    inner_max_x, center_y, h_outer_right_x, center_y,
                    inner, outer, all_contours
                )

                if not left_clear or not right_clear:
                    can_horizontal = False

            # For vertical bridges, check if actual path is clear
            if can_vertical and v_outer_top_y is not None and v_outer_bottom_y is not None:
                top_clear = self._is_bridge_path_clear(
                    center_x, inner_max_y, center_x, v_outer_top_y,
                    inner, outer, all_contours
                )
                bottom_clear = self._is_bridge_path_clear(
                    center_x, inner_min_y, center_x, v_outer_bottom_y,
                    inner, outer, all_contours
                )

                if not top_clear or not bottom_clear:
                    can_vertical = False

        if not can_horizontal and not can_vertical:
            # No valid bridges, return original contours unchanged
            return [outer, inner]

        # For multi-island cases, force specific orientation
        if force_horizontal:
            if can_horizontal:
                result = self._create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x
                )
                if result != [outer, inner]:
                    return result
            # Fall through to try vertical if horizontal fails
            if can_vertical:
                return self._create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y
                )
            return [outer, inner]

        if force_vertical:
            if can_vertical:
                result = self._create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y
                )
                if result != [outer, inner]:
                    return result
            # Fall through to try horizontal if vertical fails
            if can_horizontal:
                return self._create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x
                )
            return [outer, inner]

        # Check for asymmetric shapes where vertical bridges would cut off parts
        # For glyphs like '6' and '9', the outer extends much more on one side
        # vertically, making horizontal bridges safer
        vertical_asymmetry = max(stroke_top, stroke_bottom) / max(min(stroke_top, stroke_bottom), 1)
        horizontal_asymmetry = max(stroke_left, stroke_right) / max(min(stroke_left, stroke_right), 1)

        # If vertical strokes are highly asymmetric (>2.5x), prefer horizontal bridges
        # to avoid cutting off stems/tails that extend on one side
        force_horizontal_for_asymmetry = (
            can_horizontal
            and vertical_asymmetry > 2.5
            and vertical_asymmetry > horizontal_asymmetry
        )

        # Prefer the orientation with smaller stroke (cleaner cut)
        # unless the shape is asymmetric
        prefer_vertical = (
            can_vertical
            and not force_horizontal_for_asymmetry
            and (not can_horizontal or vertical_stroke <= horizontal_stroke)
        )

        if prefer_vertical:
            # Try vertical bridges first (top/bottom)
            result = self._create_vertical_bridge_contours(
                inner, outer, center_x, half_width, inner_min_y, inner_max_y
            )
            # Fallback to horizontal if vertical failed
            if result == [outer, inner] and can_horizontal:
                result = self._create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x
                )
            # If still failed, try horizontal at bottom of inner (for triangular counters)
            if result == [outer, inner] and can_horizontal:
                bottom_center_y = inner_min_y + half_width + 10  # Just above bottom edge
                result = self._create_horizontal_bridge_contours(
                    inner, outer, bottom_center_y, half_width, inner_min_x, inner_max_x
                )
            return result
        else:
            # Try horizontal bridges first (left/right)
            result = self._create_horizontal_bridge_contours(
                inner, outer, center_y, half_width, inner_min_x, inner_max_x
            )
            # Fallback to vertical if horizontal failed
            if result == [outer, inner] and can_vertical:
                result = self._create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y
                )
            # If still failed, try horizontal at bottom of inner (for triangular counters)
            if result == [outer, inner] and can_horizontal:
                bottom_center_y = inner_min_y + half_width + 10
                result = self._create_horizontal_bridge_contours(
                    inner, outer, bottom_center_y, half_width, inner_min_x, inner_max_x
                )
            return result

    def _create_vertical_bridge_contours(
        self,
        inner: Contour,
        outer: Contour,
        center_x: float,
        half_width: float,
        inner_min_y: float,
        inner_max_y: float,
    ) -> list[Contour]:
        """Create contours with top/bottom bridges (splits into left/right pieces)."""
        bridge_left = center_x - half_width
        bridge_right = center_x + half_width

        # Find where bridges intersect contours
        # For outer: find points at bridge_left and bridge_right, above and below inner
        outer_top_left = self.find_edge_crossing(outer, bridge_left, True, constraint_min=inner_max_y)
        outer_top_right = self.find_edge_crossing(outer, bridge_right, True, constraint_min=inner_max_y)
        outer_bot_left = self.find_edge_crossing(outer, bridge_left, True, constraint_max=inner_min_y)
        outer_bot_right = self.find_edge_crossing(outer, bridge_right, True, constraint_max=inner_min_y)

        # For inner: find points at bridge_left and bridge_right
        inner_top_left = self.find_edge_crossing(inner, bridge_left, True, constraint_min=inner_min_y + (inner_max_y - inner_min_y) * 0.4)
        inner_top_right = self.find_edge_crossing(inner, bridge_right, True, constraint_min=inner_min_y + (inner_max_y - inner_min_y) * 0.4)
        inner_bot_left = self.find_edge_crossing(inner, bridge_left, True, constraint_max=inner_min_y + (inner_max_y - inner_min_y) * 0.6)
        inner_bot_right = self.find_edge_crossing(inner, bridge_right, True, constraint_max=inner_min_y + (inner_max_y - inner_min_y) * 0.6)

        if not all([outer_top_left, outer_top_right, outer_bot_left, outer_bot_right,
                    inner_top_left, inner_top_right, inner_bot_left, inner_bot_right]):
            # Couldn't find all intersection points, return originals
            return [outer, inner]

        # Type narrowing: assert all values are not None after the check above
        assert outer_bot_left is not None
        assert outer_top_left is not None
        assert inner_bot_left is not None
        assert inner_top_left is not None
        assert outer_top_right is not None
        assert outer_bot_right is not None
        assert inner_top_right is not None
        assert inner_bot_right is not None

        # Create LEFT piece: left portion of outer + bridges + left portion of inner
        # Start at BOTTOM, go UP the left side of outer, bridge to inner TOP, go DOWN the left side of inner
        left_piece = self._build_merged_contour(
            outer, inner,
            outer_bot_left, outer_top_left,
            inner_bot_left, inner_top_left,
            bridge_left, is_left=True
        )

        # Create RIGHT piece: right portion of outer + bridges + right portion of inner
        # Start at top, traverse outer clockwise, cross to inner, traverse inner clockwise
        right_piece = self._build_merged_contour(
            outer, inner,
            outer_top_right, outer_bot_right,
            inner_top_right, inner_bot_right,
            bridge_right, is_left=False
        )

        result = []
        if left_piece:
            result.append(left_piece)
        if right_piece:
            result.append(right_piece)

        return result if result else [outer, inner]

    def _create_horizontal_bridge_contours(
        self,
        inner: Contour,
        outer: Contour,
        center_y: float,
        half_width: float,
        inner_min_x: float,
        inner_max_x: float,
    ) -> list[Contour]:
        """Create contours with left/right bridges (splits into top/bottom pieces)."""
        bridge_bottom = center_y - half_width
        bridge_top = center_y + half_width

        # Find where bridges intersect contours
        outer_left_top = self.find_edge_crossing(outer, bridge_top, False, constraint_max=inner_min_x)
        outer_left_bot = self.find_edge_crossing(outer, bridge_bottom, False, constraint_max=inner_min_x)
        outer_right_top = self.find_edge_crossing(outer, bridge_top, False, constraint_min=inner_max_x)
        outer_right_bot = self.find_edge_crossing(outer, bridge_bottom, False, constraint_min=inner_max_x)

        inner_left_top = self.find_edge_crossing(inner, bridge_top, False, constraint_max=inner_min_x + (inner_max_x - inner_min_x) * 0.6)
        inner_left_bot = self.find_edge_crossing(inner, bridge_bottom, False, constraint_max=inner_min_x + (inner_max_x - inner_min_x) * 0.6)
        inner_right_top = self.find_edge_crossing(inner, bridge_top, False, constraint_min=inner_min_x + (inner_max_x - inner_min_x) * 0.4)
        inner_right_bot = self.find_edge_crossing(inner, bridge_bottom, False, constraint_min=inner_min_x + (inner_max_x - inner_min_x) * 0.4)

        if not all([outer_left_top, outer_left_bot, outer_right_top, outer_right_bot,
                    inner_left_top, inner_left_bot, inner_right_top, inner_right_bot]):
            return [outer, inner]

        # Type narrowing: assert all values are not None after the check above
        assert outer_right_top is not None
        assert outer_left_top is not None
        assert inner_right_top is not None
        assert inner_left_top is not None
        assert outer_left_bot is not None
        assert outer_right_bot is not None
        assert inner_left_bot is not None
        assert inner_right_bot is not None

        # Create TOP piece: start at RIGHT, go LEFT along top of outer, bridge to inner, go RIGHT along top of inner
        top_piece = self._build_merged_contour_horizontal(
            outer, inner,
            outer_right_top, outer_left_top,
            inner_right_top, inner_left_top,
            bridge_top, True
        )

        # Create BOTTOM piece: start at LEFT, go RIGHT along bottom of outer, bridge to inner, go LEFT along bottom of inner
        bottom_piece = self._build_merged_contour_horizontal(
            outer, inner,
            outer_left_bot, outer_right_bot,
            inner_left_bot, inner_right_bot,
            bridge_bottom, False
        )

        result = []
        if top_piece:
            result.append(top_piece)
        if bottom_piece:
            result.append(bottom_piece)

        return result if result else [outer, inner]

    def _detect_traversal_direction(
        self,
        contour_points: list[Point],
        start_idx: int,
        end_idx: int,
        threshold: float,
        want_less_than: bool,
        is_x: bool,
    ) -> int:
        """Detect which traversal direction stays on the correct side.

        Args:
            contour_points: Points of the contour
            start_idx: Starting index
            end_idx: Ending index
            threshold: The X or Y value to compare against
            want_less_than: If True, want points with coord < threshold; else > threshold
            is_x: If True, compare X coordinates; else Y

        Returns:
            +1 for forward traversal, -1 for backward traversal
        """
        n = len(contour_points)

        # Try forward direction: count points on correct side
        forward_correct = 0
        forward_total = 0
        idx = (start_idx + 1) % n
        count = 0
        while idx != end_idx and count < n:
            p = contour_points[idx]
            coord = p.x if is_x else p.y
            if want_less_than:
                if coord < threshold:
                    forward_correct += 1
            else:
                if coord > threshold:
                    forward_correct += 1
            forward_total += 1
            idx = (idx + 1) % n
            count += 1

        # Try backward direction: count points on correct side
        backward_correct = 0
        backward_total = 0
        idx = start_idx
        count = 0
        while idx != end_idx and count < n:
            p = contour_points[idx]
            coord = p.x if is_x else p.y
            if want_less_than:
                if coord < threshold:
                    backward_correct += 1
            else:
                if coord > threshold:
                    backward_correct += 1
            backward_total += 1
            idx = (idx - 1) % n
            count += 1

        # Choose direction with higher percentage of correct points
        forward_pct = forward_correct / max(forward_total, 1)
        backward_pct = backward_correct / max(backward_total, 1)

        return -1 if backward_pct > forward_pct else 1

    def _build_merged_contour(
        self,
        outer: Contour,
        inner: Contour,
        outer_start: tuple[int, float, float],
        outer_end: tuple[int, float, float],
        inner_start: tuple[int, float, float],
        inner_end: tuple[int, float, float],
        bridge_x: float,
        is_left: bool,
    ) -> Contour | None:
        """Build a merged contour from outer and inner segments with vertical bridges.

        Detects correct traversal direction based on which path stays on the
        correct side of the bridge (left or right of bridge_x).

        IMPORTANT: Preserves point_type (ON_CURVE/OFF_CURVE) for proper curve rendering.
        """
        from stencilizer.domain import PointType

        try:
            points = []
            outer_points = outer.points
            inner_points = inner.points
            n_outer = len(outer_points)
            n_inner = len(inner_points)

            outer_start_idx, outer_start_y, _ = outer_start
            outer_end_idx, outer_end_y, _ = outer_end
            inner_start_idx, inner_start_y, _ = inner_start
            inner_end_idx, inner_end_y, _ = inner_end

            # Detect correct traversal direction for outer contour
            # LEFT piece wants points with X < bridge_x
            # RIGHT piece wants points with X > bridge_x
            outer_dir = self._detect_traversal_direction(
                outer_points, outer_start_idx, outer_end_idx,
                bridge_x, want_less_than=is_left, is_x=True
            )

            # Detect correct traversal direction for inner contour
            inner_dir = self._detect_traversal_direction(
                inner_points, inner_end_idx, inner_start_idx,
                bridge_x, want_less_than=is_left, is_x=True
            )

            # Start point on outer (at bridge) - ON_CURVE for sharp corner
            points.append(Point(bridge_x, outer_start_y, PointType.ON_CURVE))

            # Traverse outer from start to end, PRESERVING point_type
            if outer_dir == -1:
                idx = outer_start_idx
                count = 0
                while idx != outer_end_idx and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx - 1) % n_outer
                    count += 1
            else:
                idx = (outer_start_idx + 1) % n_outer
                count = 0
                while idx != (outer_end_idx + 1) % n_outer and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx + 1) % n_outer
                    count += 1

            # End point on outer (at bridge) - ON_CURVE for sharp corner
            points.append(Point(bridge_x, outer_end_y, PointType.ON_CURVE))

            # Bridge to inner - ON_CURVE for sharp corner
            points.append(Point(bridge_x, inner_end_y, PointType.ON_CURVE))

            # Traverse inner from end to start, PRESERVING point_type
            if inner_dir == -1:
                idx = inner_end_idx
                count = 0
                while idx != inner_start_idx and count < n_inner:
                    p = inner_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx - 1) % n_inner
                    count += 1
            else:
                idx = (inner_end_idx + 1) % n_inner
                count = 0
                while idx != (inner_start_idx + 1) % n_inner and count < n_inner:
                    p = inner_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx + 1) % n_inner
                    count += 1

            # Bridge back to outer - ON_CURVE for sharp corner
            points.append(Point(bridge_x, inner_start_y, PointType.ON_CURVE))

            if len(points) < 3:
                return None

            # Compute winding direction from actual geometry
            direction = self._compute_winding_direction(points)
            return Contour(points=points, direction=direction)
        except Exception:
            return None

    def _build_merged_contour_horizontal(
        self,
        outer: Contour,
        inner: Contour,
        outer_start: tuple[int, float, float],
        outer_end: tuple[int, float, float],
        inner_start: tuple[int, float, float],
        inner_end: tuple[int, float, float],
        bridge_y: float,
        is_top: bool,
    ) -> Contour | None:
        """Build a merged contour from outer and inner segments with horizontal bridges.

        Detects correct traversal direction based on which path stays on the
        correct side of the bridge (above or below bridge_y).

        IMPORTANT: Preserves point_type (ON_CURVE/OFF_CURVE) for proper curve rendering.
        """
        from stencilizer.domain import PointType

        try:
            points = []
            outer_points = outer.points
            inner_points = inner.points
            n_outer = len(outer_points)
            n_inner = len(inner_points)

            outer_start_idx, outer_start_x, _ = outer_start
            outer_end_idx, outer_end_x, _ = outer_end
            inner_start_idx, inner_start_x, _ = inner_start
            inner_end_idx, inner_end_x, _ = inner_end

            # Detect correct traversal direction for outer contour
            # TOP piece wants points with Y > bridge_y
            # BOTTOM piece wants points with Y < bridge_y
            outer_dir = self._detect_traversal_direction(
                outer_points, outer_start_idx, outer_end_idx,
                bridge_y, want_less_than=not is_top, is_x=False
            )

            # Detect correct traversal direction for inner contour
            inner_dir = self._detect_traversal_direction(
                inner_points, inner_end_idx, inner_start_idx,
                bridge_y, want_less_than=not is_top, is_x=False
            )

            # Start point on outer (at bridge) - ON_CURVE for sharp corner
            points.append(Point(outer_start_x, bridge_y, PointType.ON_CURVE))

            # Traverse outer from start to end, PRESERVING point_type
            if outer_dir == -1:
                idx = outer_start_idx
                count = 0
                while idx != outer_end_idx and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx - 1) % n_outer
                    count += 1
            else:
                idx = (outer_start_idx + 1) % n_outer
                count = 0
                while idx != (outer_end_idx + 1) % n_outer and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx + 1) % n_outer
                    count += 1

            # End point on outer (at bridge) - ON_CURVE for sharp corner
            points.append(Point(outer_end_x, bridge_y, PointType.ON_CURVE))

            # Bridge to inner - ON_CURVE for sharp corner
            points.append(Point(inner_end_x, bridge_y, PointType.ON_CURVE))

            # Traverse inner from end to start, PRESERVING point_type
            if inner_dir == -1:
                idx = inner_end_idx
                count = 0
                while idx != inner_start_idx and count < n_inner:
                    p = inner_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx - 1) % n_inner
                    count += 1
            else:
                idx = (inner_end_idx + 1) % n_inner
                count = 0
                while idx != (inner_start_idx + 1) % n_inner and count < n_inner:
                    p = inner_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx + 1) % n_inner
                    count += 1

            # Bridge back to outer - ON_CURVE for sharp corner
            points.append(Point(inner_start_x, bridge_y, PointType.ON_CURVE))

            if len(points) < 3:
                return None

            # Compute winding direction from actual geometry
            direction = self._compute_winding_direction(points)
            return Contour(points=points, direction=direction)
        except Exception:
            return None

    def merge_multi_island_vertical(
        self,
        outer: Contour,
        inners: list[Contour],
        bridge_width: float,
    ) -> list[Contour]:
        """Merge outer with multiple inner contours using a single vertical cut.

        Creates a single vertical line through all contours, splitting the glyph
        into LEFT and RIGHT pieces. This is ideal for vertically-stacked islands
        like '8' and 'B' where a single vertical bridge line is desired.

        Args:
            outer: Outer contour (parent)
            inners: List of inner contours (islands), sorted by Y position (top first)
            bridge_width: Width of the bridge gap

        Returns:
            List of 2 contours [left_piece, right_piece], or original contours if failed
        """
        if not inners:
            return [outer]

        # Compute common bridge X position from all inners
        all_inner_min_x = min(inner.bounding_box()[0] for inner in inners)
        all_inner_max_x = max(inner.bounding_box()[2] for inner in inners)
        center_x = (all_inner_min_x + all_inner_max_x) / 2.0
        half_width = bridge_width / 2.0
        bridge_left = center_x - half_width
        bridge_right = center_x + half_width

        # Get bounds for constraint calculations
        all_inner_min_y = min(inner.bounding_box()[1] for inner in inners)
        all_inner_max_y = max(inner.bounding_box()[3] for inner in inners)

        # Sort inners by Y position (bottom to top for processing)
        inners_sorted = sorted(inners, key=lambda c: c.bounding_box()[1])  # by min_y

        try:
            # Find outer crossings at top and bottom of entire region
            outer_top_left = self.find_edge_crossing(outer, bridge_left, True, constraint_min=all_inner_max_y)
            outer_top_right = self.find_edge_crossing(outer, bridge_right, True, constraint_min=all_inner_max_y)
            outer_bot_left = self.find_edge_crossing(outer, bridge_left, True, constraint_max=all_inner_min_y)
            outer_bot_right = self.find_edge_crossing(outer, bridge_right, True, constraint_max=all_inner_min_y)

            if not all([outer_top_left, outer_top_right, outer_bot_left, outer_bot_right]):
                return [outer, *inners]

            # Type narrowing: assert all outer values are not None
            assert outer_top_left is not None
            assert outer_top_right is not None
            assert outer_bot_left is not None
            assert outer_bot_right is not None

            # Find crossings for each inner
            inner_crossings = []
            for inner in inners_sorted:
                inner_bbox = inner.bounding_box()
                inner_mid_y = (inner_bbox[1] + inner_bbox[3]) / 2

                # Find crossings on each side of this inner
                left_top = self.find_edge_crossing(inner, bridge_left, True, constraint_min=inner_mid_y - 1)
                left_bot = self.find_edge_crossing(inner, bridge_left, True, constraint_max=inner_mid_y + 1)
                right_top = self.find_edge_crossing(inner, bridge_right, True, constraint_min=inner_mid_y - 1)
                right_bot = self.find_edge_crossing(inner, bridge_right, True, constraint_max=inner_mid_y + 1)

                if not all([left_top, left_bot, right_top, right_bot]):
                    return [outer, *inners]

                inner_crossings.append({
                    'inner': inner,
                    'left_top': left_top,
                    'left_bot': left_bot,
                    'right_top': right_top,
                    'right_bot': right_bot,
                    'min_y': inner_bbox[1],
                    'max_y': inner_bbox[3],
                })

            # Build LEFT piece: outer_left + all inner_lefts connected with bridges
            left_piece = self._build_multi_island_piece(
                outer, inner_crossings,
                outer_bot_left, outer_top_left,
                bridge_left, is_left=True
            )

            # Build RIGHT piece: outer_right + all inner_rights connected with bridges
            right_piece = self._build_multi_island_piece(
                outer, inner_crossings,
                outer_top_right, outer_bot_right,
                bridge_right, is_left=False
            )

            if left_piece and right_piece:
                return [left_piece, right_piece]
            else:
                return [outer, *inners]

        except Exception:
            return [outer, *inners]

    def _build_multi_island_piece(
        self,
        outer: Contour,
        inner_crossings: list[dict],
        outer_start: tuple[int, float, float],
        outer_end: tuple[int, float, float],
        bridge_x: float,
        is_left: bool,
    ) -> Contour | None:
        """Build a merged contour piece with multiple inner cutouts.

        For LEFT piece (is_left=True): traverse outer bottom-to-top on left side,
        splicing in left halves of each inner (bottom to top).

        For RIGHT piece (is_left=False): traverse outer top-to-bottom on right side,
        splicing in right halves of each inner (top to bottom).
        """
        from stencilizer.domain import PointType

        try:
            points = []
            outer_points = outer.points
            n_outer = len(outer_points)

            outer_start_idx, outer_start_y, _ = outer_start
            outer_end_idx, outer_end_y, _ = outer_end

            # Detect traversal direction for outer
            outer_dir = self._detect_traversal_direction(
                outer_points, outer_start_idx, outer_end_idx,
                bridge_x, want_less_than=is_left, is_x=True
            )

            # Sort inner_crossings by Y for proper splice order
            if is_left:
                # For left piece: process bottom to top
                sorted_crossings = sorted(inner_crossings, key=lambda c: c['min_y'])
            else:
                # For right piece: process top to bottom
                sorted_crossings = sorted(inner_crossings, key=lambda c: -c['max_y'])

            # Start point on outer
            points.append(Point(bridge_x, outer_start_y, PointType.ON_CURVE))

            # We'll traverse outer segments and splice in inners at appropriate Y positions
            # For simplicity, traverse the entire outer side first, then add inner segments
            # This creates a contour that goes: outer_left -> bridge -> inner1_left -> bridge -> inner2_left -> etc.

            # Traverse outer from start to end
            if outer_dir == -1:
                idx = outer_start_idx
                count = 0
                while idx != outer_end_idx and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx - 1) % n_outer
                    count += 1
            else:
                idx = (outer_start_idx + 1) % n_outer
                count = 0
                while idx != (outer_end_idx + 1) % n_outer and count < n_outer:
                    p = outer_points[idx]
                    points.append(Point(p.x, p.y, p.point_type))
                    idx = (idx + 1) % n_outer
                    count += 1

            # End point on outer
            points.append(Point(bridge_x, outer_end_y, PointType.ON_CURVE))

            # Now add each inner's left/right segment with bridges
            for crossing in sorted_crossings:
                inner = crossing['inner']
                inner_points = inner.points
                n_inner = len(inner_points)

                if is_left:
                    start_crossing = crossing['left_bot']
                    end_crossing = crossing['left_top']
                else:
                    start_crossing = crossing['right_top']
                    end_crossing = crossing['right_bot']

                start_idx, start_y, _ = start_crossing
                end_idx, end_y, _ = end_crossing

                # Bridge to inner
                points.append(Point(bridge_x, start_y, PointType.ON_CURVE))

                # Detect inner traversal direction
                inner_dir = self._detect_traversal_direction(
                    inner_points, start_idx, end_idx,
                    bridge_x, want_less_than=is_left, is_x=True
                )

                # Traverse inner
                if inner_dir == -1:
                    idx = start_idx
                    count = 0
                    while idx != end_idx and count < n_inner:
                        p = inner_points[idx]
                        points.append(Point(p.x, p.y, p.point_type))
                        idx = (idx - 1) % n_inner
                        count += 1
                else:
                    idx = (start_idx + 1) % n_inner
                    count = 0
                    while idx != (end_idx + 1) % n_inner and count < n_inner:
                        p = inner_points[idx]
                        points.append(Point(p.x, p.y, p.point_type))
                        idx = (idx + 1) % n_inner
                        count += 1

                # Bridge back
                points.append(Point(bridge_x, end_y, PointType.ON_CURVE))

            if len(points) < 3:
                return None

            # Compute winding direction from actual geometry
            direction = self._compute_winding_direction(points)
            return Contour(points=points, direction=direction)
        except Exception:
            return None


# Keep the old class name as an alias for compatibility
BridgeHoleCreator = ContourMerger


class GlyphTransformer:
    """Transforms glyphs by merging contours with bridge gaps.

    For each island (inner contour), merges it with its parent outer contour
    to create new contours with bridge gaps built in.
    """

    def __init__(
        self,
        analyzer: GlyphAnalyzer,
        placer: BridgePlacer,
        generator: BridgeGenerator,
    ) -> None:
        """Initialize glyph transformer with required services."""
        self.analyzer = analyzer
        self.placer = placer
        self.generator = generator
        self.merger = ContourMerger()

    def transform(self, glyph: Glyph, upm: int = 1000) -> Glyph:
        """Transform a glyph by merging contours with bridge gaps.

        Args:
            glyph: Glyph to transform
            upm: Units per em for the font

        Returns:
            New glyph with bridge gaps built into contours
        """
        hierarchy = self.analyzer.analyze(glyph)

        if not hierarchy.islands:
            return glyph

        # Calculate bridge width from config
        reference_stroke = upm * 0.1
        bridge_width = (self.placer.config.width_percent / 100.0) * reference_stroke

        # Group islands by their parent outer contour
        islands_by_parent: dict[int, list[int]] = {}
        for island_idx in hierarchy.islands:
            parent_idx = hierarchy.containment.get(island_idx)
            if parent_idx is not None:
                if parent_idx not in islands_by_parent:
                    islands_by_parent[parent_idx] = []
                islands_by_parent[parent_idx].append(island_idx)

        # Track which contours have been processed
        processed_indices: set[int] = set()
        new_contours: list[Contour] = []

        # Process each parent's islands sequentially
        for parent_idx, island_indices in islands_by_parent.items():
            if parent_idx in processed_indices:
                continue

            # Sort islands by Y position (top-to-bottom) for consistent merge order
            island_indices_sorted = sorted(
                island_indices,
                key=lambda idx: -glyph.contours[idx].bounding_box()[3],  # -max_y (top first)
            )

            # Check if islands are vertically stacked (distinct Y ranges) or
            # horizontally arranged (distinct X ranges, similar Y)
            is_vertically_stacked = False
            is_horizontally_arranged = False
            if len(island_indices_sorted) > 1:
                bboxes = [glyph.contours[idx].bounding_box() for idx in island_indices_sorted]

                # Check for vertical stacking (one above another)
                for i in range(len(bboxes) - 1):
                    upper_center_y = (bboxes[i][1] + bboxes[i][3]) / 2
                    lower_max_y = bboxes[i + 1][3]
                    if lower_max_y < upper_center_y:
                        is_vertically_stacked = True
                        break

                # Check for horizontal arrangement (side by side at similar Y)
                if not is_vertically_stacked:
                    # Sort by X to check horizontal arrangement
                    bboxes_by_x = sorted(bboxes, key=lambda b: b[0])  # sort by min_x
                    for i in range(len(bboxes_by_x) - 1):
                        left_center_x = (bboxes_by_x[i][0] + bboxes_by_x[i][2]) / 2
                        right_min_x = bboxes_by_x[i + 1][0]
                        # Check if they're horizontally separated
                        if right_min_x > left_center_x:
                            # Also check Y overlap (similar Y range)
                            y_overlap = (
                                min(bboxes_by_x[i][3], bboxes_by_x[i + 1][3]) -
                                max(bboxes_by_x[i][1], bboxes_by_x[i + 1][1])
                            )
                            if y_overlap > 0:
                                is_horizontally_arranged = True
                                break

            # For vertically-stacked multi-island glyphs (like 8, B),
            # use horizontal bridges (TOP/BOTTOM pieces)
            if is_vertically_stacked and len(island_indices_sorted) > 1:
                current_pieces: list[Contour] = [glyph.contours[parent_idx]]

                for island_idx in island_indices_sorted:
                    inner = glyph.contours[island_idx]
                    inner_bbox = inner.bounding_box()
                    inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                    inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                    containing_piece_idx = None
                    for i, piece in enumerate(current_pieces):
                        if piece.contains_point(inner_center_x, inner_center_y):
                            containing_piece_idx = i
                            break

                    if containing_piece_idx is not None:
                        piece = current_pieces[containing_piece_idx]
                        # Force horizontal bridges for multi-island to create TOP/BOTTOM pieces
                        result = self.merger.merge_contours_with_bridges(
                            inner, piece, bridge_width, force_horizontal=True,
                            all_contours=glyph.contours
                        )
                        if len(result) >= 1 and result != [piece, inner]:
                            current_pieces = (
                                current_pieces[:containing_piece_idx]
                                + result
                                + current_pieces[containing_piece_idx + 1:]
                            )
                            processed_indices.add(island_idx)

                new_contours.extend(current_pieces)
                processed_indices.add(parent_idx)
            elif is_horizontally_arranged and len(island_indices_sorted) > 1:
                # For horizontally-arranged multi-island glyphs (like Phi),
                # use vertical bridges (LEFT/RIGHT pieces) so each island
                # ends up in a separate piece
                current_pieces: list[Contour] = [glyph.contours[parent_idx]]

                # Sort by X for left-to-right processing
                island_indices_by_x = sorted(
                    island_indices,
                    key=lambda idx: glyph.contours[idx].bounding_box()[0],  # min_x
                )

                for island_idx in island_indices_by_x:
                    inner = glyph.contours[island_idx]
                    inner_bbox = inner.bounding_box()
                    inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                    inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                    containing_piece_idx = None
                    for i, piece in enumerate(current_pieces):
                        if piece.contains_point(inner_center_x, inner_center_y):
                            containing_piece_idx = i
                            break

                    if containing_piece_idx is not None:
                        piece = current_pieces[containing_piece_idx]
                        # Force vertical bridges for horizontal arrangement
                        # This creates LEFT/RIGHT pieces, keeping each island separate
                        result = self.merger.merge_contours_with_bridges(
                            inner, piece, bridge_width, force_vertical=True,
                            all_contours=glyph.contours
                        )
                        if len(result) >= 1 and result != [piece, inner]:
                            current_pieces = (
                                current_pieces[:containing_piece_idx]
                                + result
                                + current_pieces[containing_piece_idx + 1:]
                            )
                            processed_indices.add(island_idx)

                new_contours.extend(current_pieces)
                processed_indices.add(parent_idx)
            else:
                # Single island - process normally
                current_pieces: list[Contour] = [glyph.contours[parent_idx]]

                for island_idx in island_indices_sorted:
                    if island_idx in processed_indices:
                        continue

                    inner = glyph.contours[island_idx]
                    inner_bbox = inner.bounding_box()
                    inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                    inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                    # Find which current piece contains this island
                    containing_piece_idx: int | None = None
                    for i, piece in enumerate(current_pieces):
                        if piece.contains_point(inner_center_x, inner_center_y):
                            containing_piece_idx = i
                            break

                    if containing_piece_idx is None:
                        continue

                    # Merge this island with the containing piece
                    outer = current_pieces[containing_piece_idx]
                    merged = self.merger.merge_contours_with_bridges(
                        inner=inner,
                        outer=outer,
                        bridge_width=bridge_width,
                        all_contours=glyph.contours,
                    )

                    if len(merged) >= 1 and merged != [outer, inner]:
                        current_pieces = (
                            current_pieces[:containing_piece_idx]
                            + merged
                            + current_pieces[containing_piece_idx + 1:]
                        )
                        processed_indices.add(island_idx)

                new_contours.extend(current_pieces)
                processed_indices.add(parent_idx)

        # Add any unprocessed contours (non-islands, failed merges, orphans)
        for i, contour in enumerate(glyph.contours):
            if i not in processed_indices:
                new_contours.append(contour)

        return Glyph(metadata=glyph.metadata, contours=new_contours)
