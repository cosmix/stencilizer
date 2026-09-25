"""Contour merger for creating stencil bridges."""

from stencilizer.config.settings import GeometryConfig
from stencilizer.core.curve import curve_tolerance, flatten_contour
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
        check_crossings(measured, outer, config.get_line_epsilon(upm))
        check_obstructions(measured, inner, outer, all_contours)
        if not measured.can_horizontal and not measured.can_vertical:
            return [original_outer, original_inner]
        dispatch = MergeDispatch(
            inner, outer, measured, all_contours, processed_nested, config, upm
        )
        if force_horizontal:
            result = dispatch.forced_horizontal()
        elif force_vertical:
            result = dispatch.forced_vertical()
        else:
            result = dispatch.preferred()
        if processed_nested is not None:
            processed_nested[nested_start:] = [
                reverse.get(id(contour), contour) for contour in processed_nested[nested_start:]
            ]
        return [original_outer, original_inner] if result == [outer, inner] else result
