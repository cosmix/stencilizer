"""Glyph representation and metadata.

This module defines the glyph domain model, which represents a single
glyph (character) in a font with its outline contours and metadata.
"""

from dataclasses import dataclass
from typing import Any

from stencilizer.domain.contour import Contour, WindingDirection


@dataclass
class GlyphMetadata:
    """Metadata about a glyph.

    Attributes:
        name: Glyph name (e.g., "A", "B", "exclam")
        unicode: Unicode code point (None for unencoded glyphs)
        advance_width: Horizontal advance width in font units
        left_side_bearing: Left side bearing in font units
    """

    name: str
    unicode: int | None
    advance_width: int
    left_side_bearing: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary representation of metadata
        """
        return {
            "name": self.name,
            "unicode": self.unicode,
            "advance_width": self.advance_width,
            "lsb": self.left_side_bearing
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlyphMetadata":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of metadata

        Returns:
            GlyphMetadata instance
        """
        return cls(
            name=data["name"],
            unicode=data["unicode"],
            advance_width=data["advance_width"],
            left_side_bearing=data["lsb"]
        )


@dataclass
class Glyph:
    """Represents a single glyph with its contours.

    Designed for efficient serialization for parallel processing.

    Attributes:
        metadata: Glyph metadata (name, unicode, metrics)
        contours: List of contours forming the glyph outline
        _is_composite: Internal flag indicating if glyph uses component references
    """

    metadata: GlyphMetadata
    contours: list[Contour]
    _is_composite: bool = False

    @property
    def name(self) -> str:
        """Get glyph name from metadata.

        Returns:
            Glyph name
        """
        return self.metadata.name

    def is_empty(self) -> bool:
        """Check if glyph has no outlines.

        Empty glyphs include spaces and other non-printing characters.

        Returns:
            True if glyph has no contours, False otherwise
        """
        return len(self.contours) == 0

    def is_composite(self) -> bool:
        """Check if glyph is made of component references.

        Composite glyphs reference other glyphs rather than having
        their own outlines. This flag is set during font loading.

        Returns:
            True if glyph is composite, False otherwise
        """
        return self._is_composite

    def has_islands(self) -> bool:
        """Check if glyph has inner contours (islands/holes).

        Islands are contours with clockwise winding direction (in TrueType
        convention) that represent holes in the glyph shape.

        Returns:
            True if glyph has at least one inner contour, False otherwise
        """
        return any(contour.direction == WindingDirection.CLOCKWISE for contour in self.contours)

    def get_islands(self) -> list[Contour]:
        """Get list of inner contours (islands/holes).

        Returns:
            List of contours with clockwise winding direction
        """
        return [
            contour for contour in self.contours
            if contour.direction == WindingDirection.CLOCKWISE
        ]

    def get_outer_contours(self) -> list[Contour]:
        """Get list of outer contours.

        Returns:
            List of contours with counter-clockwise winding direction
        """
        return [
            contour for contour in self.contours
            if contour.direction == WindingDirection.COUNTER_CLOCKWISE
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary representation of the glyph
        """
        return {
            "metadata": self.metadata.to_dict(),
            "contours": [c.to_dict() for c in self.contours],
            "is_composite": self.is_composite()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Glyph":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of a glyph

        Returns:
            Glyph instance
        """
        metadata = GlyphMetadata.from_dict(data["metadata"])
        contours = [Contour.from_dict(c) for c in data["contours"]]
        is_composite = data.get("is_composite", False)
        return cls(metadata=metadata, contours=contours, _is_composite=is_composite)
