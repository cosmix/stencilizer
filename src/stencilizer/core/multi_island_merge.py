"""Axis-parameterized multi-island bridge merge."""

import logging
from dataclasses import dataclass

from stencilizer.core.axis import Axis
from stencilizer.core.curve import flatten_contour
from stencilizer.core.geometry import find_all_edge_crossings, find_edge_crossing
from stencilizer.core.multi_island_nested import append_nested_contours
from stencilizer.core.multi_island_obstruction import has_spanning_obstruction_axis
from stencilizer.core.multi_island_portions import build_inner_axis, build_outer_axis
from stencilizer.domain import Contour

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _MergeBounds:
    lower: float
    upper: float
    cross_min: float
    cross_max: float


def _bridge_interval(
    inners: list[Contour], bridge_width: float, axis: Axis
) -> tuple[float, float] | None:
    common_min = max(axis.bbox_lo(inner.bounding_box()) for inner in inners)
    common_max = min(axis.bbox_hi(inner.bounding_box()) for inner in inners)
    if common_max <= common_min:
        if axis.is_x:
            logger.debug(
                "Multi-island merge failed: no common X overlap (common_min_x=%.1f, common_max_x=%.1f)",
                common_min,
                common_max,
            )
        return None
    center = (common_min + common_max) / 2.0
    available = common_max - common_min
    if bridge_width > available:
        if axis.is_x:
            logger.debug(
                "Bridge width (%.1f) exceeds available width (%.1f), clamping to %.1f",
                bridge_width,
                available,
                available * 0.9,
            )
        bridge_width = available * 0.9
    half_width = bridge_width / 2.0
    return center - half_width, center + half_width


def _outer_crossings_exist(
    outer: Contour, bounds: _MergeBounds, axis: Axis, epsilon: float
) -> bool:
    if axis.is_x:
        crossings = [
            find_edge_crossing(
                outer, bounds.lower, True, constraint_min=bounds.cross_max, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.upper, True, constraint_min=bounds.cross_max, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.lower, True, constraint_max=bounds.cross_min, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.upper, True, constraint_max=bounds.cross_min, epsilon=epsilon
            ),
        ]
        names = ["outer_top_left", "outer_top_right", "outer_bot_left", "outer_bot_right"]
    else:
        crossings = [
            find_edge_crossing(
                outer, bounds.upper, False, constraint_min=bounds.cross_max, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.upper, False, constraint_max=bounds.cross_min, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.lower, False, constraint_min=bounds.cross_max, epsilon=epsilon
            ),
            find_edge_crossing(
                outer, bounds.lower, False, constraint_max=bounds.cross_min, epsilon=epsilon
            ),
        ]
        names = []
    if all(crossings):
        return True
    if axis.is_x:
        missing = [name for name, crossing in zip(names, crossings, strict=True) if not crossing]
        logger.debug("Multi-island merge failed: missing outer crossings %s", missing)
    return False


def _inner_crossings_exist(
    inners_sorted: list[Contour], bounds: _MergeBounds, axis: Axis, epsilon: float
) -> bool:
    first_bridge = bounds.lower if axis.is_x else bounds.upper
    second_bridge = bounds.upper if axis.is_x else bounds.lower
    for inner in inners_sorted:
        first_all = find_all_edge_crossings(inner, first_bridge, axis.is_x, epsilon=epsilon)
        second_all = find_all_edge_crossings(inner, second_bridge, axis.is_x, epsilon=epsilon)
        if len(first_all) < 2 or len(second_all) < 2:
            if axis.is_x:
                bbox = inner.bounding_box()
                logger.debug(
                    "Multi-island merge failed: insufficient crossings for island at y=[%.1f, %.1f] "
                    "(left=%d, right=%d)",
                    bbox[1],
                    bbox[3],
                    len(first_all),
                    len(second_all),
                )
            return False
    return True


def _append_side(
    result: list[Contour],
    outer: Contour,
    inners_sorted: list[Contour],
    bounds: _MergeBounds,
    axis: Axis,
    first_side: bool,
    epsilon: float,
    connection_tolerance: float,
    duplicate_tolerance: float,
) -> tuple[Contour | None, int]:
    bridge = (
        (bounds.lower if first_side else bounds.upper)
        if axis.is_x
        else (bounds.upper if first_side else bounds.lower)
    )
    outer_crossings = (
        find_all_edge_crossings(outer, bridge, axis.is_x, epsilon=epsilon) if axis.is_x else []
    )
    outer_portion = build_outer_axis(
        outer,
        bridge,
        outer_crossings,
        bounds.cross_min,
        bounds.cross_max,
        first_side,
        axis,
        epsilon=epsilon,
        connection_tolerance=connection_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
    if outer_portion:
        result.append(outer_portion)
    built = 0
    for inner in inners_sorted:
        crossings = find_all_edge_crossings(inner, bridge, axis.is_x, epsilon=epsilon)
        portion = build_inner_axis(
            inner,
            bridge,
            crossings,
            first_side,
            axis,
            epsilon=epsilon,
            connection_tolerance=connection_tolerance,
            duplicate_tolerance=duplicate_tolerance,
        )
        if portion:
            result.append(portion)
            built += 1
    return outer_portion, built


def _build_core(
    outer: Contour,
    inners_sorted: list[Contour],
    bounds: _MergeBounds,
    axis: Axis,
    epsilon: float,
    connection_tolerance: float,
    duplicate_tolerance: float,
) -> list[Contour] | None:
    result: list[Contour] = []
    first_outer, first_count = _append_side(
        result,
        outer,
        inners_sorted,
        bounds,
        axis,
        True,
        epsilon,
        connection_tolerance,
        duplicate_tolerance,
    )
    second_outer, second_count = _append_side(
        result,
        outer,
        inners_sorted,
        bounds,
        axis,
        False,
        epsilon,
        connection_tolerance,
        duplicate_tolerance,
    )
    if axis.is_x and not _core_complete(
        first_outer, second_outer, first_count, second_count, len(inners_sorted)
    ):
        return None
    return result


def _core_complete(
    first_outer: Contour | None,
    second_outer: Contour | None,
    first_count: int,
    second_count: int,
    count: int,
) -> bool:
    if not first_outer or not second_outer:
        logger.debug(
            "Multi-island merge failed: outer portion build failed (left=%s, right=%s)",
            first_outer is not None,
            second_outer is not None,
        )
        return False
    if first_count != count or second_count != count:
        logger.debug(
            "Multi-island merge failed: inner portions incomplete (left=%d/%d, right=%d/%d)",
            first_count,
            count,
            second_count,
            count,
        )
        return False
    return True


def _obstructed(
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour] | None,
    lower: float,
    upper: float,
    axis: Axis,
    edge_margin: float,
) -> bool:
    if not all_contours or not has_spanning_obstruction_axis(
        outer, inners, all_contours, lower, upper, axis, edge_margin=edge_margin
    ):
        return False
    if axis.is_x:
        logger.debug(
            "Multi-island merge aborted: spanning obstruction detected at x=[%.1f, %.1f]",
            lower,
            upper,
        )
    return True


def merge_multi_island_axis(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    axis: Axis,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    edge_margin: float = 20.0,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> list[Contour]:
    """Flatten once so crossing indexes and traversal refer to the same edges."""
    originals = [outer, *inners, *(all_contours or [])]
    flattened = {id(c): flatten_contour(c, epsilon * 250) for c in originals}
    reverse = {id(flattened[id(c)]): c for c in originals}
    flat_outer = flattened[id(outer)]
    flat_inners = [flattened[id(c)] for c in inners]
    flat_all = [flattened[id(c)] for c in all_contours] if all_contours is not None else None
    nested_start = len(processed_nested) if processed_nested is not None else 0
    result = _merge_multi_island_flat(
        flat_outer,
        flat_inners,
        bridge_width,
        axis,
        flat_all,
        processed_nested,
        edge_margin=edge_margin,
        epsilon=epsilon,
        connection_tolerance=connection_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
    if processed_nested is not None:
        processed_nested[nested_start:] = [
            reverse.get(id(c), c) for c in processed_nested[nested_start:]
        ]
    return [outer, *inners] if result == [flat_outer, *flat_inners] else result


def _merge_multi_island_flat(
    outer: Contour,
    inners: list[Contour],
    bridge_width: float,
    axis: Axis,
    all_contours: list[Contour] | None = None,
    processed_nested: list[Contour] | None = None,
    *,
    edge_margin: float = 20.0,
    epsilon: float = 0.001,
    connection_tolerance: float = 1.0,
    duplicate_tolerance: float = 0.5,
) -> list[Contour]:
    """Merge islands through one axis, retaining orientation-specific side order."""
    if not inners:
        return [outer]
    interval = _bridge_interval(inners, bridge_width, axis)
    if interval is None:
        return [outer, *inners]
    lower, upper = interval
    if _obstructed(outer, inners, all_contours, lower, upper, axis, edge_margin):
        return [outer, *inners]
    cross_min = min(axis.cross_lo(inner.bounding_box()) for inner in inners)
    cross_max = max(axis.cross_hi(inner.bounding_box()) for inner in inners)
    bounds = _MergeBounds(lower, upper, cross_min, cross_max)
    inners_sorted = sorted(inners, key=lambda c: axis.cross_lo(c.bounding_box()))
    try:
        return _execute_merge(
            outer,
            inners,
            inners_sorted,
            all_contours,
            processed_nested,
            bounds,
            axis,
            edge_margin,
            epsilon,
            connection_tolerance,
            duplicate_tolerance,
        )
    except Exception as exc:
        if axis.is_x:
            logger.debug("Multi-island merge failed with exception: %s", str(exc))
        return [outer, *inners]


def _execute_merge(
    outer: Contour,
    inners: list[Contour],
    inners_sorted: list[Contour],
    all_contours: list[Contour] | None,
    processed_nested: list[Contour] | None,
    bounds: _MergeBounds,
    axis: Axis,
    edge_margin: float,
    epsilon: float,
    connection_tolerance: float,
    duplicate_tolerance: float,
) -> list[Contour]:
    if not _outer_crossings_exist(outer, bounds, axis, epsilon):
        return [outer, *inners]
    if not _inner_crossings_exist(inners_sorted, bounds, axis, epsilon):
        return [outer, *inners]
    result = _build_core(
        outer, inners_sorted, bounds, axis, epsilon, connection_tolerance, duplicate_tolerance
    )
    if result is None:
        return [outer, *inners]
    if all_contours:
        append_nested_contours(
            result,
            outer,
            inners,
            all_contours,
            processed_nested,
            bounds.lower,
            bounds.upper,
            axis,
            edge_margin=edge_margin,
            epsilon=epsilon,
            connection_tolerance=connection_tolerance,
            duplicate_tolerance=duplicate_tolerance,
        )
    return result if len(result) >= 4 else [outer, *inners]
