"""Transform glyph contours by merging islands with their parent contours."""

from stencilizer.config.settings import BridgeConfig, GeometryConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.merger import ContourMerger
from stencilizer.core.surgery_context import SurgeryContext
from stencilizer.core.surgery_groups import process_groups
from stencilizer.core.surgery_nested import find_containing_hole, process_nested
from stencilizer.domain import Glyph

__all__ = ["ContourMerger", "GlyphTransformer"]

_find_containing_hole = find_containing_hole


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
        hierarchy = self.analyzer.analyze(glyph)
        if not hierarchy.islands:
            return glyph
        reference_stroke = upm * 0.1
        bridge_width = (self.bridge_config.width_percent / 100.0) * reference_stroke
        use_spanning = self.bridge_config.use_spanning_bridges
        ctx = SurgeryContext(
            glyph,
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
        for i, contour in enumerate(glyph.contours):
            if i not in ctx.processed:
                ctx.contours.append(contour)
        return Glyph(metadata=glyph.metadata, contours=ctx.contours)
