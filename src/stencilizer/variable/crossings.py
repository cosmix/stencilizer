"""The steps of an overlap-union plan, and where its crossings land in one master.

A crossing replays as the intersection of the lines through its two input edges, so it
slides along flattened curves from master to master. When it slides past a neighbouring
union vertex, that vertex lies beyond the crossing and the outline folds back on itself:
a spike, a bow-tie or a sliver hole the master's true union does not have. Such vertices
collapse onto the crossing, re-intersected on the input edges where the contours now
cross, so the structure stays the same in every master.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from stencilizer.core.geometry import signed_area
from stencilizer.domain.contour import Point
from stencilizer.variable.solver import Coords

PARALLEL_EPSILON = 1e-12
# Most union vertices along one crossing edge that a master may collapse onto it.
SETTLE_LIMIT = 8

Pt = tuple[float, float]
Edge = tuple[int, int]


@dataclass(frozen=True)
class Vertex:
    """An output vertex that is input vertex ``index`` (flat index over all contours)."""

    index: int


@dataclass(frozen=True)
class Crossing:
    """An output vertex where input edges ``first`` and ``second`` intersect."""

    first: Edge
    second: Edge


Step = Vertex | Crossing
Plan = list[list[Step]]
_Slot = tuple[int, int]
# Input vertices walking away from a crossing, and the union slots of the leading ones.
_Walk = tuple[list[int], list[_Slot]]
# Each input vertex's predecessor and successor along its contour.
_Links = tuple[list[int], list[int]]


def polygon_area(polygon: Sequence[Pt]) -> float:
    """Signed area of ``polygon``: positive counter-clockwise, negative clockwise."""
    return signed_area([Point(x, y) for x, y in polygon])


def intersect(points: Coords, first: Edge, second: Edge) -> Pt | None:
    """Intersection of the lines through both edges; None when they are parallel.

    The point may lie past an edge end. ``settle_crossings`` handles a slide past the
    union's own vertices; the overlap replay's fidelity check rejects the rest, such as
    strokes that no longer overlap.
    """
    (ax, ay), (bx, by) = points[first[0]], points[first[1]]
    (cx, cy), (dx, dy) = points[second[0]], points[second[1]]
    denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denominator) < PARALLEL_EPSILON:
        return None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _walk(plan: Plan, slot: _Slot, direction: int, edge: Edge, links: _Links) -> _Walk:
    """The input contour from the crossing at ``slot`` past its neighbour on ``edge``.

    Empty when that union neighbour is not an end of ``edge``. Slots cover the leading
    vertices the union keeps as consecutive steps.
    """
    steps = plan[slot[0]]
    neighbour = steps[(slot[1] + direction) % len(steps)]
    if not isinstance(neighbour, Vertex) or neighbour.index not in edge:
        return [], []
    onward = links[1] if neighbour.index == edge[1] else links[0]
    indices = [neighbour.index]
    while len(indices) <= SETTLE_LIMIT:
        indices.append(onward[indices[-1]])
    slots: list[_Slot] = []
    for distance, index in enumerate(indices[:SETTLE_LIMIT], start=1):
        position = (slot[1] + direction * distance) % len(steps)
        step = steps[position]
        if not isinstance(step, Vertex) or step.index != index:
            break
        slots.append((slot[0], position))
    return indices, slots


def _behind(index: int, anchor: Pt, point: Pt, default: Coords, points: Coords) -> bool:
    """True when input vertex ``index`` lies on the other side of the crossing than by default."""
    (dx, dy), (mx, my) = default[index], points[index]
    return (dx - anchor[0]) * (mx - point[0]) + (dy - anchor[1]) * (my - point[1]) < 0.0


def _settled(
    crossing: Crossing, walks: list[list[_Walk]], default: Coords, points: Coords
) -> tuple[Pt, list[_Slot]] | None:
    """The settled crossing and the union slots collapsing onto it; None when none move.

    Each crossing edge follows at most one walk, the one whose vertices fall behind the
    crossing. Each collapse re-intersects the crossing on the input edge from the walk's
    nearest kept vertex to its last collapsed one.
    """
    edges = (crossing.first, crossing.second)
    anchor = intersect(default, *edges)
    point = intersect(points, *edges)
    if anchor is None or point is None:
        return None
    active: list[_Walk] = [([], []), ([], [])]
    counts = [0, 0]
    while True:
        moved = False
        for side in (0, 1):
            pending = [active[side]] if counts[side] else walks[side]
            behind = [
                walk
                for walk in pending
                if counts[side] < len(walk[1])
                and _behind(walk[0][counts[side]], anchor, point, default, points)
            ]
            if len(behind) > 1:
                return None
            if behind:
                active[side], counts[side], moved = behind[0], counts[side] + 1, True
        if not moved:
            break
        lines = [
            (walk[0][count], walk[0][count - 1]) if count else edge
            for walk, count, edge in zip(active, counts, edges, strict=True)
        ]
        settled = intersect(points, lines[0], lines[1])
        if settled is None:
            return None
        point = settled
    if not any(counts):
        return None
    return point, [s for walk, count in zip(active, counts, strict=True) for s in walk[1][:count]]


def _walks(plan: Plan, slots: list[_Slot], edge: Edge, links: _Links) -> list[_Walk]:
    """Every walk leaving one of a crossing's ``slots`` along its input ``edge``."""
    walks = [_walk(plan, slot, direction, edge, links) for slot in slots for direction in (-1, 1)]
    return [walk for walk in walks if walk[0]]


def settle_crossings(
    plan: Plan, edges: Sequence[Edge], default: Coords, points: Coords, polygons: list[list[Pt]]
) -> None:
    """Collapse onto each crossing of ``polygons`` the union vertices it slid past.

    ``polygons`` is ``plan`` replayed on ``points`` and is updated in place; ``edges[i]``
    is input edge ``(i, next vertex along its contour)``. A union may pass one crossing
    more than once; every pass lands on the same settled point.
    """
    following = [b for _, b in edges]
    preceding = list(following)
    for a, b in edges:
        preceding[b] = a
    places: dict[Crossing, list[_Slot]] = defaultdict(list)
    for p, steps in enumerate(plan):
        for k, step in enumerate(steps):
            if isinstance(step, Crossing):
                places[step].append((p, k))
    for crossing, slots in places.items():
        walks = [
            _walks(plan, slots, edge, (preceding, following))
            for edge in (crossing.first, crossing.second)
        ]
        settled = _settled(crossing, walks, default, points)
        if settled is None:
            continue
        point, collapsed = settled
        for p, k in [*slots, *collapsed]:
            polygons[p][k] = point
