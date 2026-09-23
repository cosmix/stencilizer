"""Geometry checks used before merging an island with its parent contour."""

from dataclasses import dataclass

from stencilizer.config.settings import GeometryConfig
from stencilizer.core.geometry import find_edge_crossing, is_bridge_path_clear
from stencilizer.domain import Contour


@dataclass
class MergeMeasure:
    inner_min_x: float
    inner_min_y: float
    inner_max_x: float
    inner_max_y: float
    center_x: float
    center_y: float
    half_width: float
    stroke_left: float
    stroke_right: float
    stroke_top: float
    stroke_bottom: float
    horizontal_stroke: float
    vertical_stroke: float
    max_bridge_length: float
    can_horizontal: bool
    can_vertical: bool
    h_outer_left_x: float | None = None
    h_outer_right_x: float | None = None
    v_outer_top_y: float | None = None
    v_outer_bottom_y: float | None = None


def _orientations(
    left: float,
    right: float,
    top: float,
    bottom: float,
    horizontal: float,
    vertical: float,
    max_h: float,
    max_v: float,
    minimum: float,
) -> tuple[bool, bool]:
    can_vertical = (
        vertical >= minimum and top >= minimum and bottom >= minimum and vertical <= max_v
    )
    can_horizontal = (
        horizontal >= minimum and left >= minimum and right >= minimum and horizontal <= max_h
    )
    return can_horizontal, can_vertical


def _length_limit(
    left: float,
    right: float,
    top: float,
    bottom: float,
    geometry: GeometryConfig,
    upm: int,
) -> float:
    return max(
        min(left, right, top, bottom) * 3,
        geometry.scaled("bridge_length_floor", upm),
    )


def measure(
    inner: Contour, outer: Contour, bridge_width: float, geometry: GeometryConfig, upm: int
) -> MergeMeasure:
    """Measure bbox strokes, retaining the original calculation order."""
    inner_bbox = inner.bounding_box()
    inner_min_x, inner_min_y, inner_max_x, inner_max_y = inner_bbox
    outer_min_x, outer_min_y, outer_max_x, outer_max_y = outer.bounding_box()
    min_stroke = geometry.scaled("min_stroke", upm)
    inner_width = inner_max_x - inner_min_x
    inner_height = inner_max_y - inner_min_y
    max_stroke_h = max(inner_width * 1.5, geometry.scaled("stroke_search_floor", upm))
    max_stroke_v = max(inner_height * 1.5, geometry.scaled("stroke_search_floor", upm))
    half_width = bridge_width / 2.0
    center_x = (inner_min_x + inner_max_x) / 2.0
    center_y = (inner_min_y + inner_max_y) / 2.0
    stroke_left = inner_min_x - outer_min_x
    stroke_right = outer_max_x - inner_max_x
    stroke_top = outer_max_y - inner_max_y
    stroke_bottom = inner_min_y - outer_min_y
    horizontal_stroke = min(stroke_left, stroke_right)
    vertical_stroke = min(stroke_top, stroke_bottom)
    max_bridge_length = _length_limit(
        stroke_left, stroke_right, stroke_top, stroke_bottom, geometry, upm
    )
    can_horizontal, can_vertical = _orientations(
        stroke_left,
        stroke_right,
        stroke_top,
        stroke_bottom,
        horizontal_stroke,
        vertical_stroke,
        max_stroke_h,
        max_stroke_v,
        min_stroke,
    )
    return MergeMeasure(
        *inner_bbox,
        center_x,
        center_y,
        half_width,
        stroke_left,
        stroke_right,
        stroke_top,
        stroke_bottom,
        horizontal_stroke,
        vertical_stroke,
        max_bridge_length,
        can_horizontal,
        can_vertical,
    )


def check_crossings(m: MergeMeasure, outer: Contour, epsilon: float) -> None:
    """Reject orientations whose actual edge crossings are too distant."""
    if m.can_horizontal:
        left = find_edge_crossing(
            outer, m.center_y, False, constraint_max=m.inner_min_x, epsilon=epsilon
        )
        right = find_edge_crossing(
            outer, m.center_y, False, constraint_min=m.inner_max_x, epsilon=epsilon
        )
        if left and right:
            m.h_outer_left_x, m.h_outer_right_x = left[1], right[1]
            left_length = m.inner_min_x - left[1]
            right_length = right[1] - m.inner_max_x
            if (
                left_length > m.max_bridge_length
                or right_length > m.max_bridge_length
                or left_length < 0
                or right_length < 0
            ):
                m.can_horizontal = False
        else:
            m.can_horizontal = False
    if m.can_vertical:
        top = find_edge_crossing(
            outer, m.center_x, True, constraint_min=m.inner_max_y, epsilon=epsilon
        )
        bottom = find_edge_crossing(
            outer, m.center_x, True, constraint_max=m.inner_min_y, epsilon=epsilon
        )
        if top and bottom:
            m.v_outer_top_y, m.v_outer_bottom_y = top[1], bottom[1]
            top_length = top[1] - m.inner_max_y
            bottom_length = m.inner_min_y - bottom[1]
            if (
                top_length > m.max_bridge_length
                or bottom_length > m.max_bridge_length
                or top_length < 0
                or bottom_length < 0
            ):
                m.can_vertical = False
        else:
            m.can_vertical = False


def check_obstructions(
    m: MergeMeasure, inner: Contour, outer: Contour, all_contours: list[Contour] | None
) -> None:
    """Reject bridge paths blocked by another filled contour."""
    if not all_contours or not (m.can_horizontal or m.can_vertical):
        return
    if m.can_horizontal and m.h_outer_left_x is not None and m.h_outer_right_x is not None:
        left_clear = is_bridge_path_clear(
            m.inner_min_x,
            m.center_y,
            m.h_outer_left_x,
            m.center_y,
            inner,
            outer,
            all_contours,
        )
        right_clear = is_bridge_path_clear(
            m.inner_max_x,
            m.center_y,
            m.h_outer_right_x,
            m.center_y,
            inner,
            outer,
            all_contours,
        )
        if not left_clear or not right_clear:
            m.can_horizontal = False
    if m.can_vertical and m.v_outer_top_y is not None and m.v_outer_bottom_y is not None:
        top_clear = is_bridge_path_clear(
            m.center_x,
            m.inner_max_y,
            m.center_x,
            m.v_outer_top_y,
            inner,
            outer,
            all_contours,
        )
        bottom_clear = is_bridge_path_clear(
            m.center_x,
            m.inner_min_y,
            m.center_x,
            m.v_outer_bottom_y,
            inner,
            outer,
            all_contours,
        )
        if not top_clear or not bottom_clear:
            m.can_vertical = False
