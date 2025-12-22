"""Core processing algorithms for stencilizer.

This module contains the core algorithms for:

- Geometry operations (signed area, point-in-polygon, intersections)
- Glyph analysis (island detection, contour classification)
- Bridge placement (candidate generation, scoring, selection)
- Glyph transformation (contour surgery, bridge insertion)

All services are designed to be:
- Stateless (safe for use in worker processes)
- Pure (no side effects)
- Well-tested with property-based testing

Key functions:
- signed_area: Calculate polygon area using shoelace formula
- point_in_polygon: Test if point is inside polygon
- line_intersection: Find intersection of two line segments
- bezier_flatten: Convert Bezier curves to line segments
- nearest_point_on_segment: Find closest point on line segment
- nearest_point_on_contour: Find closest point on contour
- perpendicular_direction: Calculate perpendicular unit vector

Key classes:
- GlyphAnalyzer: Analyzes glyphs to detect islands
- BridgePlacer: Determines optimal bridge placements
- BridgeGenerator: Creates bridge geometry from specifications
- GlyphTransformer: Transforms glyphs by inserting bridges
"""

from stencilizer.core.analyzer import (
    ContourHierarchy,
    GlyphAnalyzer,
    get_island_glyphs,
)
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.geometry import (
    bezier_flatten,
    line_intersection,
    nearest_point_on_contour,
    nearest_point_on_segment,
    perpendicular_direction,
    point_in_polygon,
    signed_area,
)
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.core.surgery import BridgeHoleCreator, GlyphTransformer

__all__ = [
    "BridgeGenerator",
    # Surgery classes
    "BridgeHoleCreator",
    # Bridge classes
    "BridgePlacer",
    # Analyzer classes
    "ContourHierarchy",
    # Processor classes
    "FontProcessor",
    "GlyphAnalyzer",
    "GlyphTransformer",
    # Geometry functions
    "bezier_flatten",
    "get_island_glyphs",
    "line_intersection",
    "nearest_point_on_contour",
    "nearest_point_on_segment",
    "perpendicular_direction",
    "point_in_polygon",
    "process_glyph",
    "signed_area",
]
