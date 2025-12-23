"""Contour merger for creating stencil bridges.

This module provides the ContourMerger class which orchestrates bridge creation
by delegating to specialized vertical and horizontal bridge modules.
"""

from stencilizer.core.geometry import (
    find_edge_crossing,
    is_bridge_path_clear,
)
from stencilizer.core.horizontal_bridge import create_horizontal_bridge_contours
from stencilizer.core.multi_island import merge_multi_island_vertical
from stencilizer.core.vertical_bridge import create_vertical_bridge_contours
from stencilizer.domain import Contour


class ContourMerger:
    """Creates bridge gaps by merging outer and inner contours.

    Instead of adding CCW hole contours (which create black artifacts in
    empty space), this class modifies the outer contour to have notches
    that reach the inner contour, creating clean bridge gaps.
    """

    def merge_contours_with_bridges(
        self,
        inner: Contour,
        outer: Contour,
        bridge_width: float,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        all_contours: list[Contour] | None = None,
        processed_nested: list[Contour] | None = None,
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
            processed_nested: Optional list to track nested contours that were processed

        Returns:
            List of new contours (replacing both outer and inner)
        """
        inner_min_x, inner_min_y, inner_max_x, inner_max_y = inner.bounding_box()
        outer_min_x, outer_min_y, outer_max_x, outer_max_y = outer.bounding_box()

        min_stroke = 20
        inner_width = inner_max_x - inner_min_x
        inner_height = inner_max_y - inner_min_y
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

        # Variables to store actual crossing points
        h_outer_left_x: float | None = None
        h_outer_right_x: float | None = None
        v_outer_top_y: float | None = None
        v_outer_bottom_y: float | None = None

        # Validate horizontal bridges using actual crossings
        if can_horizontal:
            outer_left_crossing = find_edge_crossing(
                outer, center_y, False, constraint_max=inner_min_x
            )
            outer_right_crossing = find_edge_crossing(
                outer, center_y, False, constraint_min=inner_max_x
            )

            if outer_left_crossing and outer_right_crossing:
                h_outer_left_x = outer_left_crossing[1]
                h_outer_right_x = outer_right_crossing[1]

                left_bridge_len = inner_min_x - h_outer_left_x
                right_bridge_len = h_outer_right_x - inner_max_x

                if left_bridge_len > max_bridge_length or right_bridge_len > max_bridge_length or left_bridge_len < 0 or right_bridge_len < 0:
                    can_horizontal = False
            else:
                can_horizontal = False

        # Validate vertical bridges using actual crossings
        if can_vertical:
            outer_top_crossing = find_edge_crossing(
                outer, center_x, True, constraint_min=inner_max_y
            )
            outer_bottom_crossing = find_edge_crossing(
                outer, center_x, True, constraint_max=inner_min_y
            )

            if outer_top_crossing and outer_bottom_crossing:
                v_outer_top_y = outer_top_crossing[1]
                v_outer_bottom_y = outer_bottom_crossing[1]

                top_bridge_len = v_outer_top_y - inner_max_y
                bottom_bridge_len = inner_min_y - v_outer_bottom_y

                if top_bridge_len > max_bridge_length or bottom_bridge_len > max_bridge_length or top_bridge_len < 0 or bottom_bridge_len < 0:
                    can_vertical = False
            else:
                can_vertical = False

        # Check for obstructions
        if all_contours and (can_horizontal or can_vertical):
            if can_horizontal and h_outer_left_x is not None and h_outer_right_x is not None:
                left_clear = is_bridge_path_clear(
                    inner_min_x, center_y, h_outer_left_x, center_y,
                    inner, outer, all_contours
                )
                right_clear = is_bridge_path_clear(
                    inner_max_x, center_y, h_outer_right_x, center_y,
                    inner, outer, all_contours
                )

                if not left_clear or not right_clear:
                    can_horizontal = False

            if can_vertical and v_outer_top_y is not None and v_outer_bottom_y is not None:
                top_clear = is_bridge_path_clear(
                    center_x, inner_max_y, center_x, v_outer_top_y,
                    inner, outer, all_contours
                )
                bottom_clear = is_bridge_path_clear(
                    center_x, inner_min_y, center_x, v_outer_bottom_y,
                    inner, outer, all_contours
                )

                if not top_clear or not bottom_clear:
                    can_vertical = False

        if not can_horizontal and not can_vertical:
            return [outer, inner]

        # For multi-island cases, force specific orientation
        if force_horizontal:
            if can_horizontal:
                result = create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x,
                    all_contours=all_contours, processed_nested=processed_nested
                )
                if result != [outer, inner]:
                    return result
            if can_vertical:
                return create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            return [outer, inner]

        if force_vertical:
            if can_vertical:
                result = create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y,
                    all_contours=all_contours, processed_nested=processed_nested
                )
                if result != [outer, inner]:
                    return result
            if can_horizontal:
                return create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            return [outer, inner]

        # Check for asymmetric shapes where vertical bridges would cut off parts
        vertical_asymmetry = max(stroke_top, stroke_bottom) / max(min(stroke_top, stroke_bottom), 1)
        horizontal_asymmetry = max(stroke_left, stroke_right) / max(min(stroke_left, stroke_right), 1)

        force_horizontal_for_asymmetry = (
            can_horizontal
            and vertical_asymmetry > 2.5
            and vertical_asymmetry > horizontal_asymmetry
        )

        # Check if inner is horizontally off-center from outer
        # NOTE: For off-center islands, vertical bridges are actually BETTER
        # because they only affect the side where the hole is, not the entire width.
        # So we do NOT force horizontal for off-center - we let vertical handle it.
        force_horizontal_for_off_center = False

        # Prefer the orientation with smaller stroke
        prefer_vertical = (
            can_vertical
            and not force_horizontal_for_asymmetry
            and not force_horizontal_for_off_center
            and (not can_horizontal or vertical_stroke <= horizontal_stroke)
        )

        if prefer_vertical:
            result = create_vertical_bridge_contours(
                inner, outer, center_x, half_width, inner_min_y, inner_max_y,
                all_contours=all_contours, processed_nested=processed_nested
            )
            if result == [outer, inner] and can_horizontal:
                result = create_horizontal_bridge_contours(
                    inner, outer, center_y, half_width, inner_min_x, inner_max_x,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            if result == [outer, inner] and can_horizontal:
                bottom_center_y = inner_min_y + half_width + 10
                result = create_horizontal_bridge_contours(
                    inner, outer, bottom_center_y, half_width, inner_min_x, inner_max_x,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            return result
        else:
            result = create_horizontal_bridge_contours(
                inner, outer, center_y, half_width, inner_min_x, inner_max_x,
                all_contours=all_contours, processed_nested=processed_nested
            )
            if result == [outer, inner] and can_vertical:
                result = create_vertical_bridge_contours(
                    inner, outer, center_x, half_width, inner_min_y, inner_max_y,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            if result == [outer, inner] and can_horizontal:
                bottom_center_y = inner_min_y + half_width + 10
                result = create_horizontal_bridge_contours(
                    inner, outer, bottom_center_y, half_width, inner_min_x, inner_max_x,
                    all_contours=all_contours, processed_nested=processed_nested
                )
            return result
