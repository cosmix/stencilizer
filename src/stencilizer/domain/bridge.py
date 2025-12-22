"""Bridge types for connecting inner and outer contours.

This module defines the bridge domain models used for connecting islands
(inner contours) to outer contours in stencil fonts.
"""

from dataclasses import dataclass
from typing import Any

from stencilizer.domain.contour import Contour, Point, WindingDirection


@dataclass
class BridgeSpec:
    """Specification for a bridge to be created.

    A bridge connects an inner contour (island) to an outer contour,
    preventing the island from falling off during stencil cutting.

    Attributes:
        inner_contour_idx: Index of inner contour in glyph's contour list
        outer_contour_idx: Index of outer contour in glyph's contour list
        inner_point: Connection point on inner contour
        outer_point: Connection point on outer contour
        width: Bridge width in font units
        score: Quality score for this bridge position (higher is better)
    """

    inner_contour_idx: int
    outer_contour_idx: int
    inner_point: Point
    outer_point: Point
    width: float
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary representation of the bridge spec
        """
        return {
            "inner_idx": self.inner_contour_idx,
            "outer_idx": self.outer_contour_idx,
            "inner_point": self.inner_point.to_dict(),
            "outer_point": self.outer_point.to_dict(),
            "width": self.width,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeSpec":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of a bridge spec

        Returns:
            BridgeSpec instance
        """
        return cls(
            inner_contour_idx=data["inner_idx"],
            outer_contour_idx=data["outer_idx"],
            inner_point=Point.from_dict(data["inner_point"]),
            outer_point=Point.from_dict(data["outer_point"]),
            width=data["width"],
            score=data["score"]
        )


@dataclass
class BridgeGeometry:
    """Actual geometric shape of a bridge.

    A bridge is represented as a rectangle with four vertices in
    counter-clockwise order, connecting an inner and outer contour.

    Attributes:
        vertices: Tuple of 4 points forming rectangle corners (CCW order)
        spec: Original bridge specification
    """

    vertices: tuple[Point, Point, Point, Point]
    spec: BridgeSpec

    def as_contour(self) -> Contour:
        """Convert bridge to a contour for Boolean operations.

        Returns:
            Contour with bridge vertices in counter-clockwise order
        """
        return Contour(
            points=list(self.vertices),
            direction=WindingDirection.COUNTER_CLOCKWISE
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary representation of the bridge geometry
        """
        return {
            "vertices": [v.to_dict() for v in self.vertices],
            "spec": self.spec.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeGeometry":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of bridge geometry

        Returns:
            BridgeGeometry instance
        """
        vertices_list = [Point.from_dict(v) for v in data["vertices"]]
        vertices: tuple[Point, Point, Point, Point] = (
            vertices_list[0], vertices_list[1], vertices_list[2], vertices_list[3]
        )
        spec = BridgeSpec.from_dict(data["spec"])
        return cls(vertices=vertices, spec=spec)
