"""Polygon area, containment, and winding calculations."""

from stencilizer.domain import Point, WindingDirection


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


def compute_winding_direction(points: list[Point]) -> WindingDirection:
    """Compute winding direction from signed area of points.

    Uses the shoelace formula to calculate signed area.
    Standard convention: CCW traversal gives positive area, CW gives negative.

    Args:
        points: List of points forming a closed contour

    Returns:
        WindingDirection.CLOCKWISE or WindingDirection.COUNTER_CLOCKWISE
    """
    if len(points) < 3:
        return WindingDirection.CLOCKWISE

    area = signed_area(points)

    # Standard convention: positive area = CCW, negative area = CW
    if area < 0:
        return WindingDirection.CLOCKWISE
    else:
        return WindingDirection.COUNTER_CLOCKWISE
