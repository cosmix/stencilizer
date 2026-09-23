"""Accessors for horizontal and vertical bridge coordinates."""

from dataclasses import dataclass

from stencilizer.domain import Point, PointType


@dataclass(frozen=True, slots=True)
class Axis:
    """A bridge line's fixed coordinate and its crossing coordinate."""

    is_x: bool
    name: str

    def coord(self, p: Point) -> float:
        """Return the coordinate held fixed by the bridge line."""
        return p.x if self.is_x else p.y

    def cross(self, p: Point) -> float:
        """Return the coordinate along the bridge line."""
        return p.y if self.is_x else p.x

    def point(self, coord: float, cross: float, point_type: PointType) -> Point:
        """Build a point from fixed and crossing coordinates."""
        return Point(coord, cross, point_type) if self.is_x else Point(cross, coord, point_type)

    def bbox_lo(self, bbox: tuple[float, float, float, float]) -> float:
        """Return the lower bound of the fixed coordinate."""
        return bbox[0] if self.is_x else bbox[1]

    def bbox_hi(self, bbox: tuple[float, float, float, float]) -> float:
        """Return the upper bound of the fixed coordinate."""
        return bbox[2] if self.is_x else bbox[3]

    def cross_lo(self, bbox: tuple[float, float, float, float]) -> float:
        """Return the lower bound of the crossing coordinate."""
        return bbox[1] if self.is_x else bbox[0]

    def cross_hi(self, bbox: tuple[float, float, float, float]) -> float:
        """Return the upper bound of the crossing coordinate."""
        return bbox[3] if self.is_x else bbox[2]


VERTICAL = Axis(True, "vertical")
HORIZONTAL = Axis(False, "horizontal")
