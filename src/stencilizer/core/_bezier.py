"""Internal Bezier curve flattening algorithms.

This is an internal module containing helper functions for bezier_flatten.
Not intended for public use.
"""

import math

from stencilizer.domain import Point


def flatten_quadratic(points: list[Point], tolerance: float) -> list[Point]:
    """Flatten a quadratic Bezier curve using recursive subdivision.

    Args:
        points: List of 3 control points [p0, p1, p2]
        tolerance: Maximum distance from true curve

    Returns:
        List of points approximating the curve
    """
    p0, p1, p2 = points

    # Calculate midpoint of control polygon
    mid_x = (p0.x + 2 * p1.x + p2.x) / 4
    mid_y = (p0.y + 2 * p1.y + p2.y) / 4

    # Calculate actual curve midpoint (at t=0.5)
    curve_mid_x = 0.25 * p0.x + 0.5 * p1.x + 0.25 * p2.x
    curve_mid_y = 0.25 * p0.y + 0.5 * p1.y + 0.25 * p2.y

    # Check flatness
    distance = math.hypot(mid_x - curve_mid_x, mid_y - curve_mid_y)

    if distance <= tolerance:
        # Flat enough, return endpoints
        return [p0, p2]

    # Subdivide at t=0.5
    # Left half control points
    q0 = p0
    q1 = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
    q2 = Point(curve_mid_x, curve_mid_y)

    # Right half control points
    r0 = q2
    r1 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    r2 = p2

    # Recursively flatten both halves
    left = flatten_quadratic([q0, q1, q2], tolerance)
    right = flatten_quadratic([r0, r1, r2], tolerance)

    # Combine, avoiding duplicate midpoint
    return left[:-1] + right


def flatten_cubic(points: list[Point], tolerance: float) -> list[Point]:
    """Flatten a cubic Bezier curve using recursive subdivision.

    Uses De Casteljau's algorithm for subdivision.

    Args:
        points: List of 4 control points [p0, p1, p2, p3]
        tolerance: Maximum distance from true curve

    Returns:
        List of points approximating the curve
    """
    p0, p1, p2, p3 = points

    # Calculate curve midpoint (at t=0.5)
    curve_mid_x = 0.125 * (p0.x + 3 * p1.x + 3 * p2.x + p3.x)
    curve_mid_y = 0.125 * (p0.y + 3 * p1.y + 3 * p2.y + p3.y)

    # Approximate with line segment midpoint
    line_mid_x = (p0.x + p3.x) / 2
    line_mid_y = (p0.y + p3.y) / 2

    # Check flatness
    distance = math.hypot(curve_mid_x - line_mid_x, curve_mid_y - line_mid_y)

    if distance <= tolerance:
        # Flat enough, return endpoints
        return [p0, p3]

    # Subdivide at t=0.5 using De Casteljau's algorithm
    # First level
    q0 = p0
    q1 = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
    q2 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    q3 = Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2)

    # Second level
    r0 = q0
    r1 = Point((q1.x + q2.x) / 2, (q1.y + q2.y) / 2)
    r2 = Point((q2.x + q3.x) / 2, (q2.y + q3.y) / 2)

    # Third level (midpoint)
    mid = Point((r1.x + r2.x) / 2, (r1.y + r2.y) / 2)

    # Left half: p0, q1, r1, mid
    left_points = [r0, Point((q0.x + q1.x) / 2, (q0.y + q1.y) / 2), r1, mid]
    # Right half: mid, r2, q3, p3
    right_points = [mid, r2, Point((q2.x + q3.x) / 2, (q2.y + q3.y) / 2), p3]

    # Recursively flatten both halves
    left = flatten_cubic(left_points, tolerance)
    right = flatten_cubic(right_points, tolerance)

    # Combine, avoiding duplicate midpoint
    return left[:-1] + right
