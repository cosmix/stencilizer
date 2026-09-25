"""Transform glyph contours by merging islands with their parent contours."""

from dataclasses import dataclass

from stencilizer.config.settings import BridgeConfig, GeometryConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.core.merger import ContourMerger
from stencilizer.core.surgery_context import SurgeryContext
from stencilizer.core.surgery_groups import process_groups
from stencilizer.core.surgery_nested import find_containing_hole, process_nested
from stencilizer.domain import Glyph

__all__ = ["ContourMerger", "GlyphTransformer", "TransformOutcome"]

_find_containing_hole = find_containing_hole


@dataclass(frozen=True, slots=True)
class TransformOutcome:
    glyph: Glyph
    bridge_count: int
    unbridged_count: int


class GlyphTransformer:
    """Transform glyph islands into bridge gaps."""

    def __init__(
        self,
        analyzer: GlyphAnalyzer,
        bridge_config: BridgeConfig | None = None,
        merger: ContourMerger | None = None,
        geometry_config: GeometryConfig | None = None,
    ) -> None:
        """Initialize the transformer with its geometry services."""
        self.analyzer = analyzer
        self.merger = merger if merger is not None else ContourMerger()
        self.bridge_config = bridge_config if bridge_config is not None else BridgeConfig()
        self.geometry_config = geometry_config if geometry_config is not None else GeometryConfig()

    def transform(self, glyph: Glyph, upm: int = 1000) -> Glyph:
        """Return a glyph with bridge gaps built into its contours."""
        return self.transform_with_outcome(glyph, upm).glyph

    def transform_with_outcome(self, glyph: Glyph, upm: int = 1000) -> TransformOutcome:
        """Return the transformed glyph and confirmed bridge results."""
        tolerance = curve_tolerance(upm)
        working = Glyph(
            metadata=glyph.metadata,
            contours=[flatten_contour(contour, tolerance) for contour in glyph.contours],
        )
        hierarchy = self.analyzer.analyze(working)
        if not hierarchy.islands:
            return TransformOutcome(glyph, 0, 0)
        reference_stroke = upm * 0.1
        bridge_width = (self.bridge_config.width_percent / 100.0) * reference_stroke
        use_spanning = self.bridge_config.use_spanning_bridges
        ctx = SurgeryContext(
            working,
            hierarchy,
            bridge_width,
            self.merger,
            self.geometry_config,
            upm,
            use_spanning,
            direction=self.bridge_config.direction,
        )
        process_groups(ctx)
        process_nested(ctx)
        if ctx.bridge_count == 0:
            return TransformOutcome(glyph, 0, len(hierarchy.islands))
        for i, contour in enumerate(glyph.contours):
            if i not in ctx.processed:
                ctx.contours.append(contour)
        result = Glyph(metadata=glyph.metadata, contours=ctx.contours)
        unresolved = len(set(hierarchy.islands) - ctx.bridged)
        return TransformOutcome(result, ctx.bridge_count, unresolved)
