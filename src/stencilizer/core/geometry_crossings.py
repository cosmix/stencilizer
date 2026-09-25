"""Line, contour, and bridge crossing calculations."""

from stencilizer.core.curve import flatten_contour
from stencilizer.core.geometry_polygon import signed_area
from stencilizer.domain import Contour, Point


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


def segments_intersect(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> bool:
    """Check if two line segments intersect using cross product method."""

    def cross(o_x: float, o_y: float, a_x: float, a_y: float, b_x: float, b_y: float) -> float:
        return (a_x - o_x) * (b_y - o_y) - (a_y - o_y) * (b_x - o_x)

    d1 = cross(bx1, by1, bx2, by2, ax1, ay1)
    d2 = cross(bx1, by1, bx2, by2, ax2, ay2)
    d3 = cross(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = cross(ax1, ay1, ax2, ay2, bx2, by2)

    return d1 * d2 < 0 and d3 * d4 < 0


def line_intersects_contour(x1: float, y1: float, x2: float, y2: float, contour: Contour) -> bool:
    """Check if a line segment intersects a contour."""
    points = flatten_contour(contour, 0.25).points
    n = len(points)

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        if segments_intersect(x1, y1, x2, y2, p1.x, p1.y, p2.x, p2.y):
            return True

    return False


def is_bridge_path_clear(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    inner: Contour,
    outer: Contour,
    all_contours: list[Contour] | None = None,
) -> bool:
    """Check if a bridge path is clear of obstructions.

    Only FILLED contours (CW, negative area) are considered obstructions.
    Holes (CCW, positive area) can be split by the bridge function and
    are not obstructions.
    """
    if all_contours is None:
        return True

    for contour in all_contours:
        if contour is inner or contour is outer:
            continue
        # Only filled contours (CW = negative signed area) are obstructions.
        # Holes (CCW = positive signed area) can be split by bridge functions.
        if signed_area(flatten_contour(contour, 0.25).points) >= 0:
            continue  # This is a hole, not an obstruction
        if line_intersects_contour(start_x, start_y, end_x, end_y, contour):
            return False

    return True


def find_edge_crossing(
    contour: Contour,
    coord: float,
    is_x: bool,
    constraint_min: float | None = None,
    constraint_max: float | None = None,
    pick_extreme: bool = False,
    *,
    epsilon: float = 0.001,
) -> tuple[int, float, float] | None:
    """Find a crossing on the flattened outline within exclusive coordinate bounds.

    Return (edge_index, crossing_coord, t_param), choosing the nearest eligible
    crossing or the farthest when pick_extreme is true. epsilon is in font units."""
    points = flatten_contour(contour, epsilon * 250).points
    n = len(points)
    best = None
    best_other = None

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        crossing = _edge_crossing_coordinates(p1, p2, coord, is_x, epsilon)
        if crossing is None:
            continue
        t, other = crossing

        if constraint_min is not None and other <= constraint_min:
            continue
        if constraint_max is not None and other >= constraint_max:
            continue

        if best is None:
            best = (i, other, t)
            best_other = other
        else:
            if pick_extreme:
                if (
                    constraint_min is not None and best_other is not None and other > best_other
                ) or (constraint_max is not None and best_other is not None and other < best_other):
                    best = (i, other, t)
                    best_other = other
            else:
                if (
                    constraint_min is not None and best_other is not None and other < best_other
                ) or (constraint_max is not None and best_other is not None and other > best_other):
                    best = (i, other, t)
                    best_other = other

    return best


def _edge_crossing_coordinates(
    p1: Point, p2: Point, coord: float, is_x: bool, epsilon: float
) -> tuple[float, float] | None:
    if is_x:
        c1, c2 = p1.x, p2.x
        o1, o2 = p1.y, p2.y
    else:
        c1, c2 = p1.y, p2.y
        o1, o2 = p1.x, p2.x

    if not ((c1 <= coord <= c2) or (c2 <= coord <= c1)):
        return None

    if abs(c2 - c1) < epsilon:
        t = 0.5
        other = (o1 + o2) / 2
    else:
        t = (coord - c1) / (c2 - c1)
        other = o1 + t * (o2 - o1)

    return t, other


def find_all_edge_crossings(
    contour: Contour, coord: float, is_x: bool, *, epsilon: float = 0.001
) -> list[tuple[int, float, float]]:
    """Find ALL points where contour edges cross a coordinate.

    The epsilon parameter is the degenerate-segment threshold in font units.

    Returns:
        List of (edge_index, crossing_coord, other_coord) sorted by other_coord descending
    """
    points = flatten_contour(contour, epsilon * 250).points
    n = len(points)
    crossings = []

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        if is_x:
            c1, c2 = p1.x, p2.x
            o1, o2 = p1.y, p2.y
        else:
            c1, c2 = p1.y, p2.y
            o1, o2 = p1.x, p2.x

        if not ((c1 <= coord <= c2) or (c2 <= coord <= c1)):
            continue

        if abs(c2 - c1) < epsilon:
            other = (o1 + o2) / 2
        else:
            t = (coord - c1) / (c2 - c1)
            other = o1 + t * (o2 - o1)

        crossings.append((i, coord, other))

    crossings.sort(key=lambda c: -c[2])
    return crossings
