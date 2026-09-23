"""Geometric operations for contour and bridge calculations.

Functions remain available here while their implementations live in focused modules.
"""

import math

from stencilizer.core.geometry_crossings import (
    find_all_edge_crossings,
    find_edge_crossing,
    is_bridge_path_clear,
    line_intersection,
    line_intersects_contour,
    segments_intersect,
)
from stencilizer.core.geometry_polygon import (
    compute_winding_direction,
    point_in_polygon,
    signed_area,
)
from stencilizer.core.geometry_traversal import (
    _compute_side_percentages,
    detect_traversal_direction,
    detect_traversal_direction_robust,
)
from stencilizer.domain import Contour, Point, PointType, WindingDirection

__all__ = [
    "Contour",
    "Point",
    "PointType",
    "WindingDirection",
    "_compute_side_percentages",
    "compute_winding_direction",
    "detect_traversal_direction",
    "detect_traversal_direction_robust",
    "find_all_edge_crossings",
    "find_edge_crossing",
    "is_bridge_path_clear",
    "line_intersection",
    "line_intersects_contour",
    "math",
    "point_in_polygon",
    "segments_intersect",
    "signed_area",
]
