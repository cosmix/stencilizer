"""Public vertical multi-island bridge operations."""

from stencilizer.core.axis import VERTICAL
from stencilizer.core.multi_island_merge import merge_multi_island_axis
from stencilizer.core.multi_island_obstruction import has_spanning_obstruction_axis
from stencilizer.domain import Contour


def has_spanning_obstruction(
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    bridge_left_x: float,
    bridge_right_x: float,
    *,
    edge_margin: float = 20.0,
) -> bool:
    """Check the vertical spanning bridge for non-structural contours."""
    return has_spanning_obstruction_axis(
        outer,
        inners,
        all_contours,
        bridge_left_x,
        bridge_right_x,
        VERTICAL,
        edge_margin=edge_margin,
    )


def merge_multi_island_vertical(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    edge_margin: float = 20.0,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> list[Contour]:
    """Merge outer and inner contours with a vertical spanning cut."""
    return merge_multi_island_axis(
        outer,
        inners,
        bridge_width,
        VERTICAL,
        all_contours,
        processed_nested,
        edge_margin=edge_margin,
        epsilon=epsilon,
        connection_tolerance=connection_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
