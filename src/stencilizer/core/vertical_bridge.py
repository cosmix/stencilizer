"""Vertical bridge contour operations."""

from stencilizer.core.axis import VERTICAL
from stencilizer.core.bridge_contours import BridgeRequest, create_bridge_contours_for_axis
from stencilizer.core.bridge_portions import build_inner_portion, build_outer_portion
from stencilizer.domain import Contour, WindingDirection


def create_vertical_bridge_contours(
    inner: Contour,
    outer: Contour,
    center_x: float,
    half_width: float,
    inner_min_y: float,
    inner_max_y: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> list[Contour]:
    """Create top/bottom bridges and left/right contour pieces."""
    request = BridgeRequest(
        center=center_x,
        half_width=half_width,
        inner_min=inner_min_y,
        inner_max=inner_max_y,
        axis=VERTICAL,
        all_contours=all_contours,
        processed_nested=processed_nested,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
    return create_bridge_contours_for_axis(inner, outer, request)


def build_outer_portion_vertical(
    outer: Contour,
    bridge_x: float,
    outer_crossings: list[tuple[int, float, float]],
    inner_min_y: float,
    inner_max_y: float,
    is_left: bool,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> Contour | None:
    """Build the left or right filled portion."""
    portion, _ = build_outer_portion(
        outer,
        bridge_x,
        outer_crossings,
        inner_min_y,
        inner_max_y,
        is_left,
        VERTICAL,
        detect_internal_holes=False,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
    return portion


def build_inner_portion_vertical(
    inner: Contour,
    bridge_x: float,
    inner_crossings: list[tuple[int, float, float]],
    is_left: bool,
    target_winding: WindingDirection | None = None,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> Contour | None:
    """Build the left or right hole portion."""
    return build_inner_portion(
        inner,
        bridge_x,
        inner_crossings,
        is_left,
        VERTICAL,
        target_winding,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
