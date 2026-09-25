"""Create bridge contours with one implementation for both fixed axes."""

import logging
from dataclasses import dataclass, replace

from stencilizer.core.axis import Axis
from stencilizer.core.bridge_nested import append_nested_contours
from stencilizer.core.bridge_portions import build_inner_portion, build_outer_portion
from stencilizer.core.curve import flatten_contour
from stencilizer.core.geometry import find_all_edge_crossings
from stencilizer.domain import Contour

logger = logging.getLogger("stencilizer.core.vertical_bridge")
Crossing = tuple[int, float, float]


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
    detect_internal_holes: bool = False


@dataclass(frozen=True, slots=True)
class BridgeSide:
    line: float
    outer_crossings: list[Crossing]
    inner_crossings: list[Crossing]
    lower: bool


def create_bridge_contours_for_axis(
    inner: Contour, outer: Contour, request: BridgeRequest
) -> list[Contour]:
    return _create_bridge_contours(
        inner, outer, replace(request, detect_internal_holes=not request.axis.is_x)
    )


def _crossings(
    contour: Contour,
    line: float,
    axis: Axis,
    epsilon: float,
) -> list[Crossing]:
    return find_all_edge_crossings(contour, line, axis.is_x, epsilon=epsilon)


def _valid_crossings(sides: tuple[BridgeSide, BridgeSide], request: BridgeRequest) -> bool:
    return all(
        any(item[2] < request.inner_min for item in side.outer_crossings)
        and any(item[2] > request.inner_max for item in side.outer_crossings)
        and bool(side.inner_crossings)
        for side in sides
    )


def _missing_vertical(sides: tuple[BridgeSide, BridgeSide], request: BridgeRequest) -> None:
    missing: list[str] = []
    for side, name in zip(sides, ("left", "right"), strict=True):
        if not any(item[2] > request.inner_max for item in side.outer_crossings):
            missing.append(f"outer_{name}_above")
        if not any(item[2] < request.inner_min for item in side.outer_crossings):
            missing.append(f"outer_{name}_below")
    if not sides[0].inner_crossings:
        missing.append("inner_left_crossings")
    if not sides[1].inner_crossings:
        missing.append("inner_right_crossings")
    logger.debug(
        "Vertical bridge creation failed: missing crossings %s at center_x=%.1f, "
        "inner_y_range=[%.1f, %.1f]",
        missing,
        request.center,
        request.inner_min,
        request.inner_max,
    )


def _side_contours(
    inner: Contour,
    outer: Contour,
    side: BridgeSide,
    request: BridgeRequest,
) -> list[Contour]:
    portion, holes = build_outer_portion(
        outer,
        side.line,
        side.outer_crossings,
        request.inner_min,
        request.inner_max,
        side.lower,
        request.axis,
        detect_internal_holes=request.detect_internal_holes,
        epsilon=request.epsilon,
        bridge_tolerance=request.bridge_tolerance,
        point_tolerance=request.point_tolerance,
    )
    hole = build_inner_portion(
        inner,
        side.line,
        side.inner_crossings,
        side.lower,
        request.axis,
        epsilon=request.epsilon,
        bridge_tolerance=request.bridge_tolerance,
        point_tolerance=request.point_tolerance,
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
    request: BridgeRequest,
) -> tuple[BridgeSide, BridgeSide] | None:
    lower_line = request.center - request.half_width
    upper_line = request.center + request.half_width
    axis = request.axis
    epsilon = request.epsilon
    first_line, second_line = (lower_line, upper_line) if axis.is_x else (upper_line, lower_line)
    first_outer = _crossings(outer, first_line, axis, epsilon)
    second_outer = _crossings(outer, second_line, axis, epsilon)
    first_inner = _crossings(inner, first_line, axis, epsilon)
    second_inner = _crossings(inner, second_line, axis, epsilon)
    sides = (
        BridgeSide(first_line, first_outer, first_inner, axis.is_x),
        BridgeSide(second_line, second_outer, second_inner, not axis.is_x),
    )
    if not _valid_crossings(sides, request):
        if axis.is_x:
            _missing_vertical(sides, request)
        return None
    return sides


def _bridge_sides(
    inner: Contour,
    outer: Contour,
    sides: tuple[BridgeSide, BridgeSide],
    request: BridgeRequest,
) -> list[Contour]:
    result: list[Contour] = []
    for side in sides:
        result.extend(_side_contours(inner, outer, side, request))
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
    request = BridgeRequest(
        center=center,
        half_width=half_width,
        inner_min=inner_min,
        inner_max=inner_max,
        axis=axis,
        all_contours=all_contours,
        processed_nested=processed_nested,
        epsilon=epsilon,
        bridge_tolerance=bridge_tolerance,
        point_tolerance=point_tolerance,
        detect_internal_holes=detect_internal_holes,
    )
    return _create_bridge_contours(inner, outer, request)


def _create_bridge_contours(
    inner: Contour, outer: Contour, request: BridgeRequest
) -> list[Contour]:
    original_inner, original_outer = inner, outer
    inner = flatten_contour(inner, request.epsilon * 250)
    outer = flatten_contour(outer, request.epsilon * 250)
    if request.all_contours is not None and (
        inner is not original_inner or outer is not original_outer
    ):
        contours = [
            inner if contour is original_inner else outer if contour is original_outer else contour
            for contour in request.all_contours
        ]
        request = replace(request, all_contours=contours)
    sides = _prepare_sides(inner, outer, request)
    if sides is None:
        return [original_outer, original_inner]
    result = _bridge_sides(inner, outer, sides, request)
    if request.all_contours:
        append_nested_contours(
            result,
            inner,
            outer,
            request.all_contours,
            request.processed_nested,
            sides[0].line,
            sides[1].line,
            request.axis.is_x,
            request.axis,
            epsilon=request.epsilon,
            bridge_tolerance=request.bridge_tolerance,
            point_tolerance=request.point_tolerance,
        )
    return result if result else [original_outer, original_inner]
