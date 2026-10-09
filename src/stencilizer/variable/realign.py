"""Put the points of a bridge line on that line in one master.

Each line point moves to the nearest master edge crossing the line's target, searched by
edge count from the input edges it lies on; the vertices next to it that end up on the
wrong side of the line are snapped onto it.
"""

import math
from collections.abc import Sequence
from statistics import fmean
from typing import TYPE_CHECKING

from stencilizer.domain.contour import Point
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.solver import Coords

if TYPE_CHECKING:
    from stencilizer.variable.replay import BridgeLine, LineMember, Slot

_SEARCH_EDGES = 12
_MIN_EDGE_SPAN = 1e-12

_Pt = tuple[float, float]
Span = tuple[int, int]
"""(start, length) of the contour a point belongs to, in the concatenated point list."""
Placed = list[list[list[float]]]
"""Mutable [x, y] per output point, per output contour."""


def axis_value(point: Point, axis: int) -> float:
    """``point.x`` for axis 0, ``point.y`` for axis 1."""
    return point.x if axis == 0 else point.y


def contour_spans(glyph: Glyph) -> list[Span]:
    """For each point of the concatenated point list, (start, length) of its contour."""
    spans: list[Span] = []
    for contour in glyph.contours:
        start = len(spans)
        spans += [(start, len(contour.points))] * len(contour.points)
    return spans


def lerp(a: _Pt, b: _Pt, t: float) -> list[float]:
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]


def mean_target(line: "BridgeLine", placed: Placed) -> float:
    """The mean ``line.axis`` coordinate of the line's cut points in ``placed``."""
    cuts = [member.slot for member in line.members if member.cut]
    return fmean(placed[ci][pi][line.axis] for ci, pi in cuts)


def _cross(a: _Pt, b: _Pt, axis: int, target: float) -> list[float] | None:
    """Where edge a -> b crosses the line ``axis == target``; exact on that axis."""
    low, high = a[axis], b[axis]
    if (low - target) * (high - target) > 0 or abs(high - low) <= _MIN_EDGE_SPAN:
        return None
    point = lerp(a, b, (target - low) / (high - low))
    point[axis] = target
    return point


def _crossing(
    member: "LineMember",
    axis: int,
    target: float,
    master: Coords,
    spans: Sequence[Span],
    fixed: list[float],
) -> list[float] | None:
    """Nearest master edge crossing the line, by edge count from the member's edges.

    At equal edge count, the crossing closest to the fixed-parameter position wins.
    """
    first, last = member.edges
    start, length = spans[first]
    low = first - start
    high = low + (last - first) % length
    for distance in range(_SEARCH_EDGES + 1):
        offsets = range(low, high + 1) if distance == 0 else (low - distance, high + distance)
        found: list[list[float]] = []
        for offset in offsets:
            a, b = master[start + offset % length], master[start + (offset + 1) % length]
            point = _cross(a, b, axis, target)
            if point is not None:
                found.append(point)
        if found:
            return min(found, key=lambda p: math.dist(p, fixed))
    return None


def realign_line(
    line: "BridgeLine", target: float, placed: Placed, master: Coords, spans: Sequence[Span]
) -> bool:
    """Put every point of ``line`` on ``axis == target``; False when one finds no crossing."""
    for member in line.members:
        ci, pi = member.slot
        point = _crossing(member, line.axis, target, master, spans, placed[ci][pi])
        if point is None:
            return False
        placed[ci][pi] = point
    return True


def project(
    slot: "Slot",
    line: "BridgeLine",
    target: float,
    placed: Placed,
    reference: Sequence[Point],
    stops: frozenset["Slot"],
) -> None:
    """Snap the vertices next to a line point that crossed to the wrong side of the line.

    ``reference`` is the slot's contour in the default output; the walk ends at any slot in
    ``stops`` (line points and cut points).
    """
    ci, pi = slot
    count = len(placed[ci])
    for step in (1, -1):
        k = pi
        for _ in range(count - 1):
            k = (k + step) % count
            if (ci, k) in stops:
                break
            default_side = axis_value(reference[k], line.axis) - line.coordinate
            if default_side * (placed[ci][k][line.axis] - target) >= 0:
                break
            placed[ci][k][line.axis] = target
