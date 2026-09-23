"""Horizontal bridge contour operations."""

from stencilizer.core.axis import HORIZONTAL
from stencilizer.core.bridge_contours import BridgeRequest, create_bridge_contours_for_axis
from stencilizer.domain import Contour


def create_horizontal_bridge_contours(
    inner: Contour,
    outer: Contour,
    center_y: float,
    half_width: float,
    inner_min_x: float,
    inner_max_x: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> list[Contour]:
    """Create left/right bridges and top/bottom contour pieces."""
    request = BridgeRequest(
        center=center_y,
        half_width=half_width,
        inner_min=inner_min_x,
        inner_max=inner_max_x,
        axis=HORIZONTAL,
        all_contours=all_contours,
        processed_nested=processed_nested,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
    return create_bridge_contours_for_axis(inner, outer, request)
