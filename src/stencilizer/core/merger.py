"""Contour merger for creating stencil bridges."""

from stencilizer.config.settings import GeometryConfig
from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.core.merger_candidates import merge_at_candidates
from stencilizer.core.merger_checks import check_crossings, check_obstructions, measure
from stencilizer.core.merger_dispatch import MergeDispatch
from stencilizer.domain import Contour


def _flatten_inputs(
    inner: Contour, outer: Contour, all_contours: list[Contour] | None, upm: int
) -> tuple[Contour, Contour, list[Contour] | None, dict[int, Contour]]:
    tolerance = curve_tolerance(upm)
    originals = list(all_contours or [])
    originals.extend((inner, outer))
    flattened = {id(c): flatten_contour(c, tolerance) for c in originals}
    reverse = {id(flattened[id(c)]): c for c in originals}
    contours = [flattened[id(c)] for c in all_contours] if all_contours is not None else None
    return flattened[id(inner)], flattened[id(outer)], contours, reverse


def _bbox_centre_merge(
    dispatch: MergeDispatch, force_horizontal: bool, force_vertical: bool
) -> list[Contour]:
    """Bridge at the bbox-centre line, or return ``[outer, inner]`` when neither axis passes."""
    m = dispatch.measure
    if not m.can_horizontal and not m.can_vertical:
        return [dispatch.outer, dispatch.inner]
    if force_horizontal:
        return dispatch.forced_horizontal()
    if force_vertical:
        return dispatch.forced_vertical()
    return dispatch.preferred()


def _merge_fallback(
    dispatch: MergeDispatch,
    nested: list[Contour] | None,
    start: int,
    orientations: tuple[bool, bool],
    vertical_first: bool,
) -> list[Contour]:
    """Run the candidate fallback after a failed bbox-centre merge.

    The failed attempt may already have appended nested pieces past ``start``; they are set
    aside so a successful fallback does not emit them twice, and restored if it fails too.
    """
    stale = nested[start:] if nested is not None else []
    if nested is not None:
        del nested[start:]
    result = merge_at_candidates(dispatch, orientations, vertical_first)
    if nested is not None and result == [dispatch.outer, dispatch.inner]:
        nested[start:] = stale
    return result


class ContourMerger:
    """Merge an inner and outer contour by cutting bridge gaps."""

    def merge_contours_with_bridges(
        self,
        inner: Contour,
        outer: Contour,
        bridge_width: float,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        all_contours: list[Contour] | None = None,
        processed_nested: list[Contour] | None = None,
        *,
        upm: int = 1000,
        geometry: GeometryConfig | None = None,
    ) -> list[Contour]:
        """Merge contours while preserving the original orientation preferences."""
        config = geometry if geometry is not None else GeometryConfig()
        original_inner, original_outer = inner, outer
        inner, outer, all_contours, reverse = _flatten_inputs(inner, outer, all_contours, upm)
        nested_start = len(processed_nested) if processed_nested is not None else 0
        measured = measure(inner, outer, bridge_width, config, upm)
        orientations = (measured.can_horizontal, measured.can_vertical)
        check_crossings(measured, outer, config.get_line_epsilon(upm))
        check_obstructions(measured, inner, outer, all_contours)
        dispatch = MergeDispatch(
            inner, outer, measured, all_contours, processed_nested, config, upm
        )
        result = _bbox_centre_merge(dispatch, force_horizontal, force_vertical)
        if result == [outer, inner]:
            result = _merge_fallback(
                dispatch,
                processed_nested,
                nested_start,
                orientations,
                vertical_first=force_vertical and not force_horizontal,
            )
        if processed_nested is not None:
            processed_nested[nested_start:] = [
                reverse.get(id(contour), contour) for contour in processed_nested[nested_start:]
            ]
        return [original_outer, original_inner] if result == [outer, inner] else result
