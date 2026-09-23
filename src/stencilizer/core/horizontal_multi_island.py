"""Public horizontal multi-island bridge operations."""

from stencilizer.core.axis import HORIZONTAL
from stencilizer.core.multi_island_merge import merge_multi_island_axis
from stencilizer.domain import Contour


def merge_multi_island_horizontal(
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
    """Merge outer and inner contours with a horizontal spanning cut."""
    return merge_multi_island_axis(
        outer,
        inners,
        bridge_width,
        HORIZONTAL,
        all_contours,
        processed_nested,
        edge_margin=edge_margin,
        epsilon=epsilon,
        connection_tolerance=connection_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
