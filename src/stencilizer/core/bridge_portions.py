"""Axis-parameterized outer and inner bridge portions."""

import logging

from stencilizer.core.axis import Axis
from stencilizer.core.bridge_segments import (
    build_contour_from_segments,
    clean_points,
    collect_segments,
)
from stencilizer.core.geometry import compute_winding_direction, signed_area
from stencilizer.domain import Contour, Point, PointType, WindingDirection

logger = logging.getLogger("stencilizer.core.vertical_bridge")
Crossing = tuple[int, float, float]


def _outer_contour(points: list[Point], point_tolerance: float) -> Contour | None:
    cleaned = clean_points(points, point_tolerance)
    if len(cleaned) < 3:
        return None
    direction = compute_winding_direction(cleaned)
    if direction != WindingDirection.CLOCKWISE:
        cleaned = list(reversed(cleaned))
        direction = WindingDirection.CLOCKWISE
    return Contour(points=cleaned, direction=direction)


def _internal_holes(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    bridge_tolerance: float,
) -> tuple[list[list[Point]], list[Contour]]:
    ranges = [
        (min(axis.cross(p) for p in seg), max(axis.cross(p) for p in seg)) for seg in segments
    ]
    internal: set[int] = set()
    for i in range(len(segments)):
        for j in range(len(segments)):
            if i != j and ranges[i][0] > ranges[j][0] and ranges[i][1] < ranges[j][1]:
                internal.add(i)
                break
    holes: list[Contour] = []
    for index in internal:
        segment = segments[index]
        if len(segment) < 2:
            continue
        points = list(segment)
        first, last = points[0], points[-1]
        if not (
            abs(axis.coord(first) - line) < bridge_tolerance
            and abs(axis.coord(last) - line) < bridge_tolerance
        ):
            if abs(axis.coord(last) - line) > bridge_tolerance:
                points.append(axis.point(line, axis.cross(last), PointType.ON_CURVE))
            if abs(axis.coord(first) - line) > bridge_tolerance:
                points.append(axis.point(line, axis.cross(first), PointType.ON_CURVE))
        if len(points) >= 3:
            if compute_winding_direction(points) != WindingDirection.COUNTER_CLOCKWISE:
                points = list(reversed(points))
            holes.append(Contour(points=points, direction=WindingDirection.COUNTER_CLOCKWISE))
    main = [seg for i, seg in enumerate(segments) if i not in internal]
    return (main if main else segments), holes


def build_outer_portion(
    outer: Contour,
    line: float,
    crossings: list[Crossing],
    inner_min: float,
    inner_max: float,
    lower: bool,
    axis: Axis,
    *,
    detect_internal_holes: bool,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> tuple[Contour | None, list[Contour]]:
    """Build the filled outer portion; horizontal bridges may retain holes."""
    try:
        below = [crossing for crossing in crossings if crossing[2] < inner_min]
        above = [crossing for crossing in crossings if crossing[2] > inner_max]
        if not below or not above:
            return None, []
        segments = collect_segments(
            outer.points,
            line,
            axis,
            lower,
            epsilon,
            require_existing_wrap=axis.is_x,
        )
        if not segments:
            return None, []
        if lower == axis.is_x:
            segments.sort(key=lambda seg: min(axis.cross(p) for p in seg))
        else:
            segments.sort(key=lambda seg: max(axis.cross(p) for p in seg), reverse=True)
        holes: list[Contour] = []
        if detect_internal_holes:
            segments, holes = _internal_holes(segments, line, axis, bridge_tolerance)
        points = build_contour_from_segments(segments, line, axis, bridge_tolerance)
        if points is None:
            return None, holes
        return _outer_contour(points, point_tolerance), holes
    except Exception as exc:
        if axis.is_x:
            logger.debug(
                "build_outer_portion_vertical failed: %s at bridge_x=%.1f, is_left=%s",
                str(exc),
                line,
                lower,
            )
        return None, []


def _candidate(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    bridge_tolerance: float,
    point_tolerance: float,
) -> list[Point] | None:
    points = build_contour_from_segments(segments, line, axis, bridge_tolerance)
    if not points:
        return None
    cleaned = clean_points(points, point_tolerance)
    return cleaned if len(cleaned) >= 3 else None


def _single_inner(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    target: WindingDirection,
    bridge_tolerance: float,
    point_tolerance: float,
) -> Contour | None:
    cleaned = _candidate(segments, line, axis, bridge_tolerance, point_tolerance)
    if cleaned is None:
        return None
    direction = compute_winding_direction(cleaned)
    if direction != target:
        cleaned = list(reversed(cleaned))
        direction = target
    return Contour(points=cleaned, direction=direction)


def _best_ordering(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    target: WindingDirection,
    bridge_tolerance: float,
    point_tolerance: float,
) -> list[Point] | None:
    orderings = [
        segments[:],
        segments[::-1],
        sorted(segments, key=lambda seg: min(axis.cross(p) for p in seg)),
        sorted(segments, key=lambda seg: max(axis.cross(p) for p in seg), reverse=True),
    ]
    best: list[Point] | None = None
    best_reversal = True
    best_area = 0.0
    for ordered in orderings:
        cleaned = _candidate(ordered, line, axis, bridge_tolerance, point_tolerance)
        if cleaned is None:
            continue
        needs_reversal = compute_winding_direction(cleaned) != target
        area = abs(signed_area(cleaned))
        if (
            best is None
            or (not needs_reversal and best_reversal)
            or (needs_reversal == best_reversal and area > best_area)
        ):
            best = cleaned
            best_reversal = needs_reversal
            best_area = area
    if best is not None and best_reversal:
        return list(reversed(best))
    return best


def _inner_from_segments(
    segments: list[list[Point]],
    line: float,
    axis: Axis,
    target_winding: WindingDirection,
    bridge_tolerance: float,
    point_tolerance: float,
) -> Contour | None:
    if len(segments) == 1:
        return _single_inner(
            segments,
            line,
            axis,
            target_winding,
            bridge_tolerance,
            point_tolerance,
        )
    best = _best_ordering(
        segments,
        line,
        axis,
        target_winding,
        bridge_tolerance,
        point_tolerance,
    )
    return Contour(points=best, direction=target_winding) if best is not None else None


def build_inner_portion(
    inner: Contour,
    line: float,
    crossings: list[Crossing],
    lower: bool,
    axis: Axis,
    target_winding: WindingDirection | None = None,
    *,
    epsilon: float = 0.001,
    bridge_tolerance: float = 1.0,
    point_tolerance: float = 0.5,
) -> Contour | None:
    """Build one hole or nested contour portion with original ordering rules."""
    if target_winding is None:
        target_winding = WindingDirection.COUNTER_CLOCKWISE
    try:
        if len(crossings) < 2:
            return None
        segments = collect_segments(
            inner.points,
            line,
            axis,
            lower,
            epsilon,
            require_existing_wrap=True,
        )
        if not segments:
            return None
        return _inner_from_segments(
            segments,
            line,
            axis,
            target_winding,
            bridge_tolerance,
            point_tolerance,
        )
    except Exception as exc:
        if axis.is_x:
            logger.debug(
                "build_inner_portion_vertical failed: %s at bridge_x=%.1f, is_left=%s",
                str(exc),
                line,
                lower,
            )
        return None
