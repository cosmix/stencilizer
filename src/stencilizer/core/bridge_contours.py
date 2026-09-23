"""Create bridge contours with one implementation for both fixed axes."""

import logging
from dataclasses import dataclass

from stencilizer.core.axis import Axis
from stencilizer.core.bridge_nested import append_nested_contours
from stencilizer.core.bridge_portions import build_inner_portion, build_outer_portion
from stencilizer.core.geometry import find_all_edge_crossings
from stencilizer.domain import Contour

logger = logging.getLogger("stencilizer.core.vertical_bridge")
Crossing = tuple[int, float, float]
BridgeSide = tuple[float, list[Crossing], list[Crossing], bool]


@dataclass(frozen=True, slots=True)
class BridgeRequest:
    center: float
    half_width: float
    inner_min: float
    inner_max: float
    axis: Axis
    all_contours: list[Contour] | None
    processed_nested: list[Contour] | None
    epsilon: float
    bridge_tolerance: float
    point_tolerance: float


def create_bridge_contours_for_axis(
    inner: Contour, outer: Contour, request: BridgeRequest
) -> list[Contour]:
    return create_bridge_contours(
        inner,
        outer,
        request.center,
        request.half_width,
        request.inner_min,
        request.inner_max,
        request.axis,
        request.all_contours,
        request.processed_nested,
        detect_internal_holes=not request.axis.is_x,
        epsilon=request.epsilon,
        bridge_tolerance=request.bridge_tolerance,
        point_tolerance=request.point_tolerance,
    )


def _crossings(
    contour: Contour,
    line: float,
    axis: Axis,
    epsilon: float,
) -> list[Crossing]:
    return find_all_edge_crossings(contour, line, axis.is_x, epsilon=epsilon)


def _valid_crossings(
    first_outer: list[Crossing],
    second_outer: list[Crossing],
    first_inner: list[Crossing],
    second_inner: list[Crossing],
    inner_min: float,
    inner_max: float,
) -> bool:
    return all(
        [item for item in crossings if item[2] < inner_min]
        and [item for item in crossings if item[2] > inner_max]
        for crossings in (first_outer, second_outer)
    ) and bool(first_inner and second_inner)


def _missing_vertical(
    first_outer: list[Crossing],
    second_outer: list[Crossing],
    first_inner: list[Crossing],
    second_inner: list[Crossing],
    inner_min: float,
    inner_max: float,
    center: float,
) -> None:
    missing: list[str] = []
    for crossings, side in ((first_outer, "left"), (second_outer, "right")):
        if not [item for item in crossings if item[2] > inner_max]:
            missing.append(f"outer_{side}_above")
        if not [item for item in crossings if item[2] < inner_min]:
            missing.append(f"outer_{side}_below")
    if not first_inner:
        missing.append("inner_left_crossings")
    if not second_inner:
        missing.append("inner_right_crossings")
    logger.debug(
        "Vertical bridge creation failed: missing crossings %s at center_x=%.1f, "
        "inner_y_range=[%.1f, %.1f]",
        missing,
        center,
        inner_min,
        inner_max,
    )


def _side_contours(
    inner: Contour,
    outer: Contour,
    line: float,
    outer_crossings: list[Crossing],
    inner_crossings: list[Crossing],
    inner_min: float,
    inner_max: float,
    lower: bool,
    axis: Axis,
    detect_internal_holes: bool,
    epsilon: float,
    bridge_tolerance: float,
    point_tolerance: float,
) -> list[Contour]:
    portion, holes = build_outer_portion(
        outer,
        line,
        outer_crossings,
        inner_min,
        inner_max,
        lower,
        axis,
        detect_internal_holes=detect_internal_holes,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
    hole = build_inner_portion(
        inner,
        line,
        inner_crossings,
        lower,
        axis,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
    )
    result: list[Contour] = []
    if portion:
        result.append(portion)
    result.extend(holes)
    if hole:
        result.append(hole)
    return result


def _prepare_sides(
    inner: Contour,
    outer: Contour,
    center: float,
    half_width: float,
    inner_min: float,
    inner_max: float,
    axis: Axis,
    epsilon: float,
) -> tuple[BridgeSide, BridgeSide] | None:
    lower_line = center - half_width
    upper_line = center + half_width
    first_line, second_line = (lower_line, upper_line) if axis.is_x else (upper_line, lower_line)
    first_outer = _crossings(outer, first_line, axis, epsilon)
    second_outer = _crossings(outer, second_line, axis, epsilon)
    first_inner = _crossings(inner, first_line, axis, epsilon)
    second_inner = _crossings(inner, second_line, axis, epsilon)
    if not _valid_crossings(
        first_outer, second_outer, first_inner, second_inner, inner_min, inner_max
    ):
        if axis.is_x:
            _missing_vertical(
                first_outer, second_outer, first_inner, second_inner, inner_min, inner_max, center
            )
        return None
    return (
        (first_line, first_outer, first_inner, axis.is_x),
        (second_line, second_outer, second_inner, not axis.is_x),
    )


def _bridge_sides(
    inner: Contour,
    outer: Contour,
    sides: tuple[BridgeSide, BridgeSide],
    inner_min: float,
    inner_max: float,
    axis: Axis,
    detect_internal_holes: bool,
    epsilon: float,
    bridge_tolerance: float,
    point_tolerance: float,
) -> list[Contour]:
    result: list[Contour] = []
    for line, outer_crossings, inner_crossings, lower in sides:
        result.extend(
            _side_contours(
                inner,
                outer,
                line,
                outer_crossings,
                inner_crossings,
                inner_min,
                inner_max,
                lower,
                axis,
                detect_internal_holes,
                epsilon,
                bridge_tolerance,
                point_tolerance,
            )
        )
    return result


def create_bridge_contours(
    inner: Contour,
    outer: Contour,
    center: float,
    half_width: float,
    inner_min: float,
    inner_max: float,
    axis: Axis,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    detect_internal_holes: bool = False,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> list[Contour]:
    """Split a filled contour and its hole along two parallel bridge lines."""
    sides = _prepare_sides(inner, outer, center, half_width, inner_min, inner_max, axis, epsilon)
    if sides is None:
        return [outer, inner]
    result = _bridge_sides(
        inner,
        outer,
        sides,
        inner_min,
        inner_max,
        axis,
        detect_internal_holes,
        epsilon,
        bridge_tolerance,
        point_tolerance,
    )
    if all_contours:
        append_nested_contours(
            result,
            inner,
            outer,
            all_contours,
            processed_nested,
            sides[0][0],
            sides[1][0],
            axis.is_x,
            axis,
            epsilon=epsilon,
            bridge_tolerance=bridge_tolerance,
            point_tolerance=point_tolerance,
        )
    return result if result else [outer, inner]
