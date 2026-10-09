"""Fallback bridge line positions for islands the bbox-centre line cannot bridge."""

from stencilizer.core.axis import HORIZONTAL, VERTICAL, Axis
from stencilizer.core.geometry import find_all_edge_crossings, find_edge_crossing
from stencilizer.core.merger_dispatch import MergeDispatch
from stencilizer.domain import Contour

Extent = tuple[float, float]
_OFF_CENTER_FRACTIONS = tuple(k / 8 for k in range(1, 8) if k != 4)


def _crossings(dispatch: MergeDispatch, axis: Axis, line: float) -> list[float]:
    epsilon = dispatch.geometry.get_line_epsilon(dispatch.upm)
    found = find_all_edge_crossings(dispatch.inner, line, axis.is_x, epsilon=epsilon)
    return [crossing[2] for crossing in found]


def _band_extent(dispatch: MergeDispatch, axis: Axis, line: float) -> Extent | None:
    """Return the inner contour's extent along the bridge band, or None if a line misses it.

    The builder classifies outer crossings against one extent for both band lines, so the
    extent spans the lower band line, the centre line, and the upper band line.
    """
    half_width = dispatch.measure.half_width
    values: list[float] = []
    for coord in (line - half_width, line, line + half_width):
        crossings = _crossings(dispatch, axis, coord)
        if len(crossings) < 2:
            return None
        values.extend(crossings)
    return min(values), max(values)


def _candidate_lines(dispatch: MergeDispatch, axis: Axis) -> list[float]:
    """Return the bbox centre, then k/8 positions ordered by the inner contour's full extent, widest first."""
    bbox = dispatch.inner.bounding_box()
    low, high = axis.bbox_lo(bbox), axis.bbox_hi(bbox)
    centre = (low + high) / 2.0
    extents: dict[float, float] = {}
    for fraction in _OFF_CENTER_FRACTIONS:
        line = low + (high - low) * fraction
        crossings = _crossings(dispatch, axis, line)
        if len(crossings) >= 2:
            extents[line] = max(crossings) - min(crossings)
    return [centre, *sorted(extents, key=lambda line: -extents[line])]


def _outer_ends(dispatch: MergeDispatch, axis: Axis, line: float, extent: Extent) -> Extent | None:
    """Return the nearest outer crossings below and above ``extent`` on one line."""
    epsilon = dispatch.geometry.get_line_epsilon(dispatch.upm)
    low, high = extent
    near = find_edge_crossing(dispatch.outer, line, axis.is_x, constraint_max=low, epsilon=epsilon)
    far = find_edge_crossing(dispatch.outer, line, axis.is_x, constraint_min=high, epsilon=epsilon)
    return (near[1], far[1]) if near and far else None


def _band_line_clear(
    dispatch: MergeDispatch, axis: Axis, line: float, ends: Extent, inner: list[float]
) -> bool:
    """Return whether both cuts on one band line miss every contour but the pair being split.

    ``ends`` are the nearest outer crossings around the extent and ``inner`` the inner
    contour's crossings on ``line``. Unlike the bbox-centre obstruction test, holes count:
    the builder splits only the bridged pair, so a sibling counter the band crossed would
    stay whole across the gap.
    """
    epsilon = dispatch.geometry.get_line_epsilon(dispatch.upm)
    margin = dispatch.geometry.get_point_dedup_tolerance(dispatch.upm)
    cuts = ((ends[0] + margin, min(inner) - margin), (max(inner) + margin, ends[1] - margin))
    for contour in dispatch.all_contours or ():
        if contour is dispatch.inner or contour is dispatch.outer:
            continue
        found = find_all_edge_crossings(contour, line, axis.is_x, epsilon=epsilon)
        if any(low < crossing[2] < high for crossing in found for low, high in cuts):
            return False
    return True


def _reachable(dispatch: MergeDispatch, axis: Axis, line: float, extent: Extent) -> bool:
    """Apply the bbox-centre crossing and length tests, then keep all three band lines clear."""
    low, high = extent
    limit = dispatch.measure.max_bridge_length
    half_width = dispatch.measure.half_width
    for coord in (line - half_width, line, line + half_width):
        ends = _outer_ends(dispatch, axis, coord, extent)
        if ends is None:
            return False
        if coord == line and not (0 <= low - ends[0] <= limit and 0 <= ends[1] - high <= limit):
            return False
        if not _band_line_clear(dispatch, axis, coord, ends, _crossings(dispatch, axis, coord)):
            return False
    return True


def _build(dispatch: MergeDispatch, axis: Axis, line: float, extent: Extent) -> list[Contour]:
    if axis is HORIZONTAL:
        return dispatch.horizontal(line, extent)
    return dispatch.vertical(line, extent)


def merge_at_candidates(
    dispatch: MergeDispatch, orientations: tuple[bool, bool], vertical_first: bool
) -> list[Contour]:
    """Try further bridge line positions; return ``[outer, inner]`` when none builds.

    ``orientations`` are the bbox gates (horizontal, vertical) measured before the
    crossing checks; only an orientation whose gate passed is tried.
    """
    failed = [dispatch.outer, dispatch.inner]
    nested = dispatch.processed_nested
    nested_start = len(nested) if nested is not None else 0
    axes = [(HORIZONTAL, orientations[0]), (VERTICAL, orientations[1])]
    for axis, eligible in axes[::-1] if vertical_first else axes:
        if not eligible:
            continue
        for line in _candidate_lines(dispatch, axis):
            extent = _band_extent(dispatch, axis, line)
            if extent is None or not _reachable(dispatch, axis, line, extent):
                continue
            result = _build(dispatch, axis, line, extent)
            if result != failed:
                return result
            if nested is not None:
                del nested[nested_start:]
    return failed
