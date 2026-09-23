"""Contour merger for creating stencil bridges."""

from stencilizer.config.settings import GeometryConfig
from stencilizer.core.merger_checks import check_crossings, check_obstructions, measure
from stencilizer.core.merger_dispatch import MergeDispatch
from stencilizer.domain import Contour


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
        measured = measure(inner, outer, bridge_width, config, upm)
        check_crossings(measured, outer, config.get_line_epsilon(upm))
        check_obstructions(measured, inner, outer, all_contours)
        if not measured.can_horizontal and not measured.can_vertical:
            return [outer, inner]
        dispatch = MergeDispatch(
            inner, outer, measured, all_contours, processed_nested, config, upm
        )
        if force_horizontal:
            return dispatch.forced_horizontal()
        if force_vertical:
            return dispatch.forced_vertical()
        return dispatch.preferred()
