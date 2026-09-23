"""Nested contour handling for multi-island merges."""

from dataclasses import dataclass

from stencilizer.core.axis import Axis
from stencilizer.core.geometry import find_all_edge_crossings, signed_area
from stencilizer.core.multi_island_obstruction import classify_obstruction_axis
from stencilizer.core.multi_island_portions import build_inner_axis
from stencilizer.domain import Contour, WindingDirection


@dataclass(frozen=True, slots=True)
class _NestedCuts:
    lower: float
    upper: float
    axis: Axis
    epsilon: float
    connection_tolerance: float
    duplicate_tolerance: float


def _inside_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]


def _is_grandchild(
    contour: Contour,
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    center_x: float,
    center_y: float,
) -> bool:
    for other in all_contours:
        if other is contour or other is outer or other in inners:
            continue
        other_area = signed_area(other.points)
        if other_area < 0:
            other_bbox = other.bounding_box()
            if _inside_bbox(center_x, center_y, other_bbox):
                other_center_x = (other_bbox[0] + other_bbox[2]) / 2
                other_center_y = (other_bbox[1] + other_bbox[3]) / 2
                for inner in inners:
                    if _inside_bbox(other_center_x, other_center_y, inner.bounding_box()):
                        return True
    return False


def _inside_any_hole(bbox: tuple[float, float, float, float], inners: list[Contour]) -> bool:
    for inner in inners:
        inner_bbox = inner.bounding_box()
        if (
            bbox[0] >= inner_bbox[0]
            and bbox[2] <= inner_bbox[2]
            and bbox[1] >= inner_bbox[1]
            and bbox[3] <= inner_bbox[3]
        ):
            return True
    return False


def _has_own_holes(
    contour: Contour,
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    bbox: tuple[float, float, float, float],
) -> bool:
    for other in all_contours:
        if other is contour or other is outer or other in inners:
            continue
        other_area = signed_area(other.points)
        if other_area > 0:
            other_bbox = other.bounding_box()
            if (
                bbox[0] < other_bbox[0]
                and bbox[2] > other_bbox[2]
                and bbox[1] < other_bbox[1]
                and bbox[3] > other_bbox[3]
            ):
                return True
    return False


def _preserve_structural(
    contour: Contour,
    outer: Contour,
    inners: list[Contour],
    bbox: tuple[float, float, float, float],
    center_x: float,
    center_y: float,
    lower: float,
    upper: float,
    axis: Axis,
    edge_margin: float,
) -> bool:
    if not axis.is_x:
        return (
            classify_obstruction_axis(contour, outer, axis, edge_margin=edge_margin) == "structural"
        )
    is_filled = signed_area(contour.points) < 0
    spans_bridge = bbox[0] < lower and bbox[2] > upper
    in_gap = True
    for inner in inners:
        if _inside_bbox(center_x, center_y, inner.bounding_box()):
            in_gap = False
            break
    return is_filled and spans_bridge and in_gap


def _append_portion(
    result: list[Contour],
    contour: Contour,
    bridge: float,
    crossings: list[tuple[int, float, float]],
    first_side: bool,
    target_winding: WindingDirection,
    cuts: _NestedCuts,
) -> None:
    portion = build_inner_axis(
        contour,
        bridge,
        crossings,
        first_side,
        cuts.axis,
        target_winding,
        epsilon=cuts.epsilon,
        connection_tolerance=cuts.connection_tolerance,
        duplicate_tolerance=cuts.duplicate_tolerance,
    )
    if portion:
        result.append(portion)


def _append_split(
    result: list[Contour],
    contour: Contour,
    bbox: tuple[float, float, float, float],
    cuts: _NestedCuts,
) -> None:
    first_bridge = cuts.lower if cuts.axis.is_x else cuts.upper
    second_bridge = cuts.upper if cuts.axis.is_x else cuts.lower
    first_crossings = find_all_edge_crossings(
        contour, first_bridge, cuts.axis.is_x, epsilon=cuts.epsilon
    )
    second_crossings = find_all_edge_crossings(
        contour, second_bridge, cuts.axis.is_x, epsilon=cuts.epsilon
    )
    target_winding = (
        WindingDirection.CLOCKWISE
        if signed_area(contour.points) < 0
        else WindingDirection.COUNTER_CLOCKWISE
    )
    if not first_crossings and not second_crossings:
        if cuts.axis.bbox_hi(bbox) <= cuts.lower or cuts.axis.bbox_lo(bbox) >= cuts.upper:
            result.append(contour)
        return
    if first_crossings and not second_crossings:
        if cuts.axis.bbox_lo(bbox) >= cuts.upper:
            result.append(contour)
        else:
            _append_portion(
                result, contour, first_bridge, first_crossings, True, target_winding, cuts
            )
        return
    if second_crossings and not first_crossings:
        if cuts.axis.bbox_hi(bbox) <= cuts.lower:
            result.append(contour)
        else:
            _append_portion(
                result, contour, second_bridge, second_crossings, False, target_winding, cuts
            )
        return
    _append_portion(result, contour, first_bridge, first_crossings, True, target_winding, cuts)
    _append_portion(result, contour, second_bridge, second_crossings, False, target_winding, cuts)


def append_nested_contours(
    result: list[Contour],
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    processed_nested: list[Contour] | None,
    lower: float,
    upper: float,
    axis: Axis,
    *,
    edge_margin: float = 20.0,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> None:
    """Append unchanged or split nested contours in their original order."""
    processed_set = {id(outer)} | {id(inner) for inner in inners}
    outer_bbox = outer.bounding_box()
    cuts = _NestedCuts(lower, upper, axis, epsilon, connection_tolerance, duplicate_tolerance)
    for contour in all_contours:
        if id(contour) in processed_set:
            continue
        bbox = contour.bounding_box()
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        if not _inside_bbox(center_x, center_y, outer_bbox):
            continue
        if _is_grandchild(contour, outer, inners, all_contours, center_x, center_y):
            continue
        is_filled = signed_area(contour.points) < 0
        if is_filled and _inside_any_hole(bbox, inners):
            if _has_own_holes(contour, outer, inners, all_contours, bbox):
                continue
            result.append(contour)
            if processed_nested is not None:
                processed_nested.append(contour)
            continue
        if _preserve_structural(
            contour, outer, inners, bbox, center_x, center_y, lower, upper, axis, edge_margin
        ):
            result.append(contour)
            if processed_nested is not None:
                processed_nested.append(contour)
            continue
        _append_split(result, contour, bbox, cuts)
