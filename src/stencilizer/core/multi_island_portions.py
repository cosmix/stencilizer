"""Side builders shared by the two multi-island merge orientations."""

from stencilizer.core.axis import Axis
from stencilizer.core.bridge_portions import build_inner_portion
from stencilizer.core.multi_island_outer import build_outer_portion_multi_island_axis
from stencilizer.core.vertical_bridge import build_outer_portion_vertical
from stencilizer.domain import Contour, WindingDirection

Crossings = list[tuple[int, float, float]]


def build_inner_axis(
    contour: Contour,
    bridge: float,
    crossings: Crossings,
    first_side: bool,
    axis: Axis,
    target_winding: WindingDirection | None = None,
    *,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> Contour | None:
    """Build the low X side or high Y side when first_side is true."""
    lower = first_side if axis.is_x else not first_side
    return build_inner_portion(
        contour,
        bridge,
        crossings,
        lower,
        axis,
        target_winding,
        epsilon=epsilon,
        bridge_tolerance=connection_tolerance,
        point_tolerance=duplicate_tolerance,
    )


def build_outer_axis(
    outer: Contour,
    bridge: float,
    crossings: Crossings,
    cross_min: float,
    cross_max: float,
    first_side: bool,
    axis: Axis,
    *,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> Contour | None:
    """Use each orientation's existing outer contour construction."""
    if axis.is_x:
        return build_outer_portion_vertical(
            outer,
            bridge,
            crossings,
            cross_min,
            cross_max,
            is_left=first_side,
            epsilon=epsilon,
            bridge_tolerance=connection_tolerance,
            point_tolerance=duplicate_tolerance,
        )
    return build_outer_portion_multi_island_axis(
        outer,
        bridge,
        is_top=first_side,
        epsilon=epsilon,
        connection_tolerance=connection_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
