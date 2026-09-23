"""Nested contour handling shared by both bridge directions."""

from stencilizer.core.axis import Axis
from stencilizer.core.bridge_portions import build_inner_portion
from stencilizer.core.geometry import find_all_edge_crossings, signed_area
from stencilizer.domain import Contour, WindingDirection


def _inside_box(
    x: float,
    y: float,
    box: tuple[float, float, float, float],
) -> bool:
    return box[0] < x < box[2] and box[1] < y < box[3]


def _grandchild(
    contour: Contour,
    inner: Contour,
    outer: Contour,
    all_contours: list[Contour],
    center_x: float,
    center_y: float,
    inner_box: tuple[float, float, float, float],
) -> bool:
    for other in all_contours:
        if other is contour or other is inner or other is outer:
            continue
        if signed_area(other.points) < 0:
            box = other.bounding_box()
            if _inside_box(center_x, center_y, box):
                other_x = (box[0] + box[2]) / 2
                other_y = (box[1] + box[3]) / 2
                if _inside_box(other_x, other_y, inner_box):
                    return True
    return False


def _own_holes(
    contour: Contour,
    inner: Contour,
    outer: Contour,
    all_contours: list[Contour],
    box: tuple[float, float, float, float],
) -> bool:
    for other in all_contours:
        if other is contour or other is inner or other is outer:
            continue
        if signed_area(other.points) > 0:
            other_box = other.bounding_box()
            if (
                box[0] < other_box[0]
                and box[2] > other_box[2]
                and box[1] < other_box[1]
                and box[3] > other_box[3]
            ):
                return True
    return False


def _nested_status(
    contour: Contour,
    inner: Contour,
    outer: Contour,
    all_contours: list[Contour],
    inner_box: tuple[float, float, float, float],
) -> tuple[int, tuple[float, float, float, float]]:
    box = contour.bounding_box()
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    if not _inside_box(center_x, center_y, inner_box):
        return 0, box
    if _grandchild(contour, inner, outer, all_contours, center_x, center_y, inner_box):
        return 0, box
    inside = (
        box[0] >= inner_box[0]
        and box[2] <= inner_box[2]
        and box[1] >= inner_box[1]
        and box[3] <= inner_box[3]
    )
    if signed_area(contour.points) < 0 and inside:
        return (0 if _own_holes(contour, inner, outer, all_contours, box) else 1), box
    return 2, box


def _nested_winding(contour: Contour) -> WindingDirection:
    return (
        WindingDirection.CLOCKWISE
        if signed_area(contour.points) < 0
        else WindingDirection.COUNTER_CLOCKWISE
    )


def _split_nested(
    contour: Contour,
    first_line: float,
    second_line: float,
    first_crossings: list[tuple[int, float, float]],
    second_crossings: list[tuple[int, float, float]],
    first_lower: bool,
    axis: Axis,
    box: tuple[float, float, float, float],
    lower_line: float,
    upper_line: float,
    epsilon: float,
    bridge_tolerance: float,
    point_tolerance: float,
) -> list[Contour]:
    if not first_crossings and not second_crossings:
        if axis.bbox_hi(box) <= lower_line or axis.bbox_lo(box) >= upper_line:
            return [contour]
        return []
    winding = _nested_winding(contour)

    def portion(
        line: float, crossings: list[tuple[int, float, float]], lower: bool
    ) -> Contour | None:
        return build_inner_portion(
            contour,
            line,
            crossings,
            lower,
            axis,
            winding,
            epsilon=epsilon,
            bridge_tolerance=bridge_tolerance,
            point_tolerance=point_tolerance,
        )

    if first_crossings and not second_crossings:
        if axis.bbox_lo(box) >= upper_line:
            return [contour]
        piece = portion(first_line, first_crossings, first_lower)
        return [piece] if piece else []
    if second_crossings and not first_crossings:
        if axis.bbox_hi(box) <= lower_line:
            return [contour]
        piece = portion(second_line, second_crossings, not first_lower)
        return [piece] if piece else []
    first = portion(first_line, first_crossings, first_lower)
    second = portion(second_line, second_crossings, not first_lower)
    return [piece for piece in (first, second) if piece]


def append_nested_contours(
    result: list[Contour],
    inner: Contour,
    outer: Contour,
    all_contours: list[Contour],
    processed_nested: list[Contour] | None,
    first_line: float,
    second_line: float,
    first_lower: bool,
    axis: Axis,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> None:
    """Append eligible descendants in their original input and split order."""
    processed_ids = {id(outer), id(inner)}
    inner_box = inner.bounding_box()
    lower_line, upper_line = sorted((first_line, second_line))
    for contour in all_contours:
        if id(contour) in processed_ids:
            continue
        status, box = _nested_status(contour, inner, outer, all_contours, inner_box)
        if status == 0:
            continue
        if status == 1:
            result.append(contour)
            if processed_nested is not None:
                processed_nested.append(contour)
            continue
        first = find_all_edge_crossings(contour, first_line, axis.is_x, epsilon=epsilon)
        second = find_all_edge_crossings(contour, second_line, axis.is_x, epsilon=epsilon)
        pieces = _split_nested(
            contour,
            first_line,
            second_line,
            first,
            second,
            first_lower,
            axis,
            box,
            lower_line,
            upper_line,
            epsilon,
            bridge_tolerance,
            point_tolerance,
        )
        result.extend(pieces)
        if processed_nested is not None and (first or second or pieces):
            processed_nested.append(contour)
