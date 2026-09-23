"""Core processing algorithms for stencilizer.

This module contains the core algorithms for:

- Geometry operations (signed area, point-in-polygon, intersections)
- Glyph analysis (island detection, contour classification)
- Glyph transformation (contour surgery, bridge insertion)

All services are designed to be:
- Stateless (safe for use in worker processes)
- Pure (no side effects)
- Well-tested with property-based testing

Key functions:
- signed_area: Calculate polygon area using shoelace formula
- point_in_polygon: Test if point is inside polygon
- line_intersection: Find intersection of two line segments

Key classes:
- GlyphAnalyzer: Analyzes glyphs to detect islands
- GlyphTransformer: Transforms glyphs by inserting bridges
"""

from stencilizer.core.analyzer import (
    ContourHierarchy,
    GlyphAnalyzer,
    get_island_glyphs,
)
from stencilizer.core.geometry import (
    line_intersection,
    point_in_polygon,
    signed_area,
)
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.core.surgery import ContourMerger, GlyphTransformer

__all__ = [
    "ContourHierarchy",
    "ContourMerger",
    "FontProcessor",
    "GlyphAnalyzer",
    "GlyphTransformer",
    "get_island_glyphs",
    "line_intersection",
    "point_in_polygon",
    "process_glyph",
    "signed_area",
]
