"""Geometric operations for contour and bridge calculations.

This module provides core mathematical utilities for:
- Signed area calculation (shoelace formula)
- Point-in-polygon testing (ray casting algorithm)
- Line segment intersection
- Bezier curve flattening
- Nearest point calculations
- Perpendicular vector computation

All functions are pure, stateless, and designed for use in parallel processing.
"""

import math

from stencilizer.core._bezier import flatten_cubic as _flatten_cubic
from stencilizer.core._bezier import flatten_quadratic as _flatten_quadratic
from stencilizer.domain import Contour, Point, PointType


def signed_area(points: list[Point]) -> float:
    """Calculate signed area of a polygon using the shoelace formula.

    The sign of the area indicates winding direction:
    - Positive area: counter-clockwise winding
    - Negative area: clockwise winding

    Args:
        points: List of points forming the polygon boundary

    Returns:
        Signed area in square units. Returns 0.0 for degenerate polygons.

    Examples:
        >>> p1 = Point(0.0, 0.0)
        >>> p2 = Point(1.0, 0.0)
        >>> p3 = Point(1.0, 1.0)
        >>> p4 = Point(0.0, 1.0)
        >>> signed_area([p1, p2, p3, p4])  # CCW square
        1.0
        >>> signed_area([p1, p4, p3, p2])  # CW square
        -1.0
    """
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y

    return area / 2.0


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Determine if a point is inside a polygon using ray casting algorithm.

    Casts a horizontal ray from the point to the right and counts intersections
    with polygon edges. Odd number of intersections = inside, even = outside.

    Args:
        point: The point to test
        polygon: List of points forming the polygon boundary

    Returns:
        True if point is inside polygon, False otherwise

    Examples:
        >>> p1 = Point(0.0, 0.0)
        >>> p2 = Point(2.0, 0.0)
        >>> p3 = Point(2.0, 2.0)
        >>> p4 = Point(0.0, 2.0)
        >>> square = [p1, p2, p3, p4]
        >>> point_in_polygon(Point(1.0, 1.0), square)  # Center
        True
        >>> point_in_polygon(Point(3.0, 3.0), square)  # Outside
        False
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x, y = point.x, point.y
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y

        # Check if ray from point intersects edge (j, i)
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point | None:
    """Find intersection point of two line segments.

    Uses parametric line equations to find intersection. Returns None if lines
    are parallel or if intersection is outside both segments.

    Args:
        p1: First endpoint of segment 1
        p2: Second endpoint of segment 1
        p3: First endpoint of segment 2
        p4: Second endpoint of segment 2

    Returns:
        Point at intersection if segments intersect, None otherwise

    Examples:
        >>> p1 = Point(0.0, 0.0)
        >>> p2 = Point(2.0, 2.0)
        >>> p3 = Point(0.0, 2.0)
        >>> p4 = Point(2.0, 0.0)
        >>> intersection = line_intersection(p1, p2, p3, p4)
        >>> # Should be at (1.0, 1.0)
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    # Calculate denominator for parametric equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Lines are parallel or coincident
    if abs(denom) < 1e-10:
        return None

    # Calculate parametric values for intersection
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return Point(x, y)

    return None  # Intersection outside segments


def bezier_flatten(points: list[Point], tolerance: float = 1.0) -> list[Point]:
    """Convert Bezier curve to line segments using recursive subdivision.

    Handles both quadratic (3 points) and cubic (4 points) Bezier curves.
    Uses recursive subdivision until the curve is flat enough (within tolerance).

    Args:
        points: Control points of the Bezier curve (3 for quadratic, 4 for cubic)
        tolerance: Maximum distance from true curve (in font units)

    Returns:
        List of points forming line segments that approximate the curve

    Raises:
        ValueError: If points list is not of length 3 or 4
    """
    if len(points) == 2:
        # Already a line segment
        return points
    elif len(points) == 3:
        # Quadratic Bezier curve
        return _flatten_quadratic(points, tolerance)
    elif len(points) == 4:
        # Cubic Bezier curve
        return _flatten_cubic(points, tolerance)
    else:
        raise ValueError(f"Expected 2-4 points for Bezier curve, got {len(points)}")


def nearest_point_on_segment(point: Point, seg_start: Point, seg_end: Point) -> tuple[Point, float]:
    """Find the closest point on a line segment to a given point.

    Projects the point onto the infinite line, then clamps to the segment endpoints.

    Args:
        point: The point to project
        seg_start: Start point of line segment
        seg_end: End point of line segment

    Returns:
        Tuple of (nearest_point, distance) where nearest_point is the closest
        point on the segment and distance is the Euclidean distance to it

    Examples:
        >>> p = Point(1.0, 1.0)
        >>> seg_start = Point(0.0, 0.0)
        >>> seg_end = Point(2.0, 0.0)
        >>> nearest, dist = nearest_point_on_segment(p, seg_start, seg_end)
        >>> # nearest should be (1.0, 0.0), dist should be 1.0
    """
    # Vector from start to end
    dx = seg_end.x - seg_start.x
    dy = seg_end.y - seg_start.y

    # Handle zero-length segment
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq < 1e-10:
        distance = math.hypot(point.x - seg_start.x, point.y - seg_start.y)
        return seg_start, distance

    # Project point onto infinite line
    # t = dot(point - start, end - start) / ||end - start||^2
    t = ((point.x - seg_start.x) * dx + (point.y - seg_start.y) * dy) / segment_length_sq

    # Clamp t to [0, 1] to stay within segment
    t = max(0.0, min(1.0, t))

    # Calculate nearest point
    nearest_x = seg_start.x + t * dx
    nearest_y = seg_start.y + t * dy
    nearest = Point(nearest_x, nearest_y)

    # Calculate distance
    distance = math.hypot(point.x - nearest_x, point.y - nearest_y)

    return nearest, distance


def nearest_point_on_contour(point: Point, contour: Contour) -> tuple[Point, float]:
    """Find the closest point on a contour to a given point.

    Checks all segments in the contour and returns the globally nearest point.
    For contours with Bezier curves, this flattens the curves first.

    Args:
        point: The point to find the nearest point to
        contour: The contour to search

    Returns:
        Tuple of (nearest_point, distance) where nearest_point is the closest
        point on the contour and distance is the Euclidean distance to it

    Raises:
        ValueError: If contour has fewer than 2 points
    """
    points = contour.points
    n = len(points)

    if n < 2:
        raise ValueError("Contour must have at least 2 points")

    nearest_point = points[0]
    min_distance = math.hypot(point.x - points[0].x, point.y - points[0].y)

    # First, expand implied on-curve points for TrueType contours
    # TrueType uses implicit on-curve points between consecutive off-curve points
    expanded_points: list[Point] = []
    for i in range(n):
        curr = points[i]
        next_pt = points[(i + 1) % n]

        expanded_points.append(curr)

        # If current and next are both off-curve, insert implied on-curve midpoint
        if (
            curr.point_type == PointType.OFF_CURVE_QUAD
            and next_pt.point_type == PointType.OFF_CURVE_QUAD
        ):
            mid_x = (curr.x + next_pt.x) / 2
            mid_y = (curr.y + next_pt.y) / 2
            expanded_points.append(Point(mid_x, mid_y, PointType.ON_CURVE))

    # Now process segments from expanded points
    exp_n = len(expanded_points)
    i = 0
    while i < exp_n:
        p0 = expanded_points[i]

        # Find next on-curve point
        j = i + 1
        while j < exp_n + i and expanded_points[j % exp_n].point_type != PointType.ON_CURVE:
            j += 1

        # Collect points for this segment
        segment_points = [p0]
        for k in range(i + 1, j + 1):
            segment_points.append(expanded_points[k % exp_n])

        # Flatten if Bezier, otherwise use as line
        if len(segment_points) == 2:
            flattened = segment_points
        elif len(segment_points) == 3:
            # Quadratic Bezier
            flattened = bezier_flatten(segment_points)
        elif len(segment_points) == 4:
            # Cubic Bezier
            flattened = bezier_flatten(segment_points)
        else:
            # Fallback: just use endpoints
            flattened = [segment_points[0], segment_points[-1]]

        # Check each flattened segment
        for k in range(len(flattened) - 1):
            seg_start = flattened[k]
            seg_end = flattened[k + 1]

            nearest, distance = nearest_point_on_segment(point, seg_start, seg_end)

            if distance < min_distance:
                min_distance = distance
                nearest_point = nearest

        # Move to next on-curve point
        i = j if j <= exp_n else i + 1

    return nearest_point, min_distance


def perpendicular_direction(p1: Point, p2: Point) -> tuple[float, float]:
    """Calculate the unit perpendicular vector to a line from p1 to p2.

    The perpendicular is rotated 90 degrees counter-clockwise from the
    direction vector (p2 - p1).

    Args:
        p1: Start point of line
        p2: End point of line

    Returns:
        Tuple (px, py) representing the unit perpendicular vector

    Raises:
        ValueError: If p1 and p2 are the same point (zero-length line)

    Examples:
        >>> p1 = Point(0.0, 0.0)
        >>> p2 = Point(1.0, 0.0)  # Horizontal line to the right
        >>> px, py = perpendicular_direction(p1, p2)
        >>> # Should return (0.0, 1.0) - perpendicular points up
    """
    # Direction vector
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    # Calculate length
    length = math.hypot(dx, dy)

    # Handle zero-length line
    if length < 1e-10:
        raise ValueError("Cannot calculate perpendicular of zero-length line")

    # Normalize
    dx /= length
    dy /= length

    # Rotate 90 degrees counter-clockwise: (x, y) -> (-y, x)
    px = -dy
    py = dx

    return px, py
