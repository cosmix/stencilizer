"""Domain models for stencilizer.

This module contains the core domain models representing fonts, glyphs, contours,
and bridges. All models are designed to be:

- Immutable where possible (using frozen dataclasses)
- Serializable for inter-process communication (parallel processing)
- Independent of fonttools implementation details

Key classes:
- Point: A 2D point with curve metadata
- Contour: A closed contour representing a shape boundary
- Glyph: A single glyph with its contours
- BridgeSpec: Specification for a bridge to be created
- BridgeGeometry: Actual geometric shape of a bridge
"""

from stencilizer.domain.bridge import BridgeGeometry, BridgeSpec
from stencilizer.domain.contour import Contour, Point, PointType, WindingDirection
from stencilizer.domain.glyph import Glyph, GlyphMetadata

__all__: list[str] = [
    # Enums
    "WindingDirection",
    "PointType",
    # Core types
    "Point",
    "Contour",
    "GlyphMetadata",
    "Glyph",
    "BridgeSpec",
    "BridgeGeometry",
]
