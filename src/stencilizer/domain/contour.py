"""Core geometric types for contour representation.

This module defines the fundamental geometric types used throughout the stencilizer:
- Point: A 2D point with curve type information
- Contour: A closed contour representing a shape boundary
- WindingDirection: Enum for contour winding direction
- PointType: Enum for point type on a curve
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class WindingDirection(Enum):
    """Contour winding direction.

    In TrueType/OpenType convention:
    - Outer contours typically wind counter-clockwise
    - Inner contours (holes) typically wind clockwise

    Note: PostScript/CFF fonts may use opposite convention.
    """

    CLOCKWISE = auto()
    COUNTER_CLOCKWISE = auto()


class PointType(Enum):
    """Point type on a contour.

    Points can be:
    - ON_CURVE: Point on the actual curve
    - OFF_CURVE_QUAD: Quadratic Bezier control point (TrueType)
    - OFF_CURVE_CUBIC: Cubic Bezier control point (PostScript/CFF)
    """

    ON_CURVE = auto()
    OFF_CURVE_QUAD = auto()
    OFF_CURVE_CUBIC = auto()


@dataclass(frozen=True, slots=True)
class Point:
    """A point in 2D space with curve metadata.

    Immutable and hashable for use in sets/dicts.
    Uses slots for memory efficiency in parallel processing.

    Attributes:
        x: X coordinate in font units
        y: Y coordinate in font units
        point_type: Type of point (on-curve or control point)
    """

    x: float
    y: float
    point_type: PointType = PointType.ON_CURVE

    def to_tuple(self) -> tuple[float, float]:
        """Convert to simple (x, y) tuple.

        Returns:
            Tuple of (x, y) coordinates
        """
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary with x, y, and type fields
        """
        return {
            "x": self.x,
            "y": self.y,
            "type": self.point_type.value
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Point":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with x, y, and type fields

        Returns:
            Point instance
        """
        return cls(
            x=data["x"],
            y=data["y"],
            point_type=PointType(data["type"])
        )


@dataclass
class Contour:
    """A closed contour representing a shape boundary.

    A contour is a sequence of points that form a closed shape.
    Contours can be outer (shape boundary) or inner (holes).

    Attributes:
        points: List of points forming the contour
        direction: Winding direction (None until calculated)
    """

    points: list[Point]
    direction: WindingDirection | None = field(default=None)
    _cached_area: float | None = field(default=None, repr=False, init=False)
    _cached_bbox: tuple[float, float, float, float] | None = field(
        default=None, repr=False, init=False
    )

    def signed_area(self) -> float:
        """Calculate signed area using shoelace formula.

        The sign of the area indicates winding direction:
        - Positive area: counter-clockwise winding
        - Negative area: clockwise winding

        Result is cached for efficiency.

        Returns:
            Signed area of the contour
        """
        if self._cached_area is not None:
            return self._cached_area

        n = len(self.points)
        if n < 3:
            self._cached_area = 0.0
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y

        self._cached_area = area / 2.0
        return self._cached_area

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Calculate bounding box of the contour.

        Result is cached for efficiency.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if self._cached_bbox is not None:
            return self._cached_bbox

        if not self.points:
            self._cached_bbox = (0.0, 0.0, 0.0, 0.0)
            return self._cached_bbox

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]

        self._cached_bbox = (min(xs), min(ys), max(xs), max(ys))
        return self._cached_bbox

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside contour using ray casting algorithm.

        Casts a ray from the point to the right and counts intersections
        with contour edges. Odd count means inside, even means outside.

        Args:
            x: X coordinate of point to test
            y: Y coordinate of point to test

        Returns:
            True if point is inside contour, False otherwise
        """
        n = len(self.points)
        if n < 3:
            return False

        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = self.points[i].x, self.points[i].y
            xj, yj = self.points[j].x, self.points[j].y

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def sample_points(self, n: int) -> list[Point]:
        """Sample n evenly-spaced points along contour.

        This is a simplified implementation that samples from the existing
        points. For more accurate sampling along curves, Bezier curve
        evaluation would be needed.

        Args:
            n: Number of points to sample

        Returns:
            List of sampled points
        """
        if n <= 0:
            return []

        if n >= len(self.points):
            return list(self.points)

        # Sample evenly from existing points
        step = len(self.points) / n
        indices = [int(i * step) for i in range(n)]
        return [self.points[i] for i in indices]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for IPC.

        Returns:
            Dictionary representation of the contour
        """
        return {
            "points": [p.to_dict() for p in self.points],
            "direction": self.direction.value if self.direction else None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Contour":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of a contour

        Returns:
            Contour instance
        """
        points = [Point.from_dict(p) for p in data["points"]]
        direction = (
            WindingDirection(data["direction"])
            if data["direction"] is not None
            else None
        )
        return cls(points=points, direction=direction)
