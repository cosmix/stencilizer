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

            # Keep track of best - we want the CLOSEST crossing to the constraint boundary
            # For constraint_min: want the MINIMUM other that satisfies constraint (closest to inner)
            # For constraint_max: want the MAXIMUM other that satisfies constraint (closest to inner)
            if best is None:
                best = (i, other, t)
                best_other = other
            else:
                if constraint_min is not None and best_other is not None and other < best_other:
                    # Want smallest Y that's still > constraint_min (closest to inner_max_y)
                    best = (i, other, t)
                    best_other = other
                elif constraint_max is not None and best_other is not None and other > best_other:
                    # Want largest Y that's still < constraint_max (closest to inner_min_y)
                    best = (i, other, t)
                    best_other = other

        return best

    def merge_contours_with_bridges(
        self,
        inner: Contour,
        outer: Contour,
        bridge_width: float,
        force_horizontal: bool = False,
    ) -> list[Contour]:
        """Merge outer and inner contours with bridge gaps.

        Creates new contours that have bridge gaps built in by
        connecting segments of outer and inner contours.

        Args:
            inner: Inner contour (island/hole)
            outer: Outer contour (parent)
            bridge_width: Width of each bridge
            force_horizontal: If True, only use horizontal bridges (for multi-island cases)

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

        if not can_horizontal and not can_vertical:
            # No valid bridges, return original contours unchanged
            return [outer, inner]

        # For multi-island cases, force horizontal to separate vertically-stacked islands
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

            return Contour(points=points, direction=WindingDirection.CLOCKWISE)
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

            return Contour(points=points, direction=WindingDirection.CLOCKWISE)
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
        _outer_bbox = self.find_contour_bounds(outer)
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

            return Contour(points=points, direction=WindingDirection.CLOCKWISE)
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

            # Check if islands are vertically stacked (distinct Y ranges)
            is_vertically_stacked = False
            if len(island_indices_sorted) > 1:
                bboxes = [glyph.contours[idx].bounding_box() for idx in island_indices_sorted]
                for i in range(len(bboxes) - 1):
                    upper_center_y = (bboxes[i][1] + bboxes[i][3]) / 2
                    lower_max_y = bboxes[i + 1][3]
                    if lower_max_y < upper_center_y:
                        is_vertically_stacked = True
                        break

            # For vertically-stacked multi-island glyphs (like 8, B),
            # use horizontal bridges (TOP/BOTTOM pieces) to avoid issues where
            # islands span both left/right halves
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
                            inner, piece, bridge_width, force_horizontal=True
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
                # Single island or horizontally arranged - process normally
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
