"""Map default-master surgery back onto its input and replay it on every master.

Vertices follow their master positions; bridge-line points are re-placed on lines
recomputed per master, because a cut point kept at its default edge parameter drifts off
its line and leaves islands in most masters.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from stencilizer.domain.contour import Contour
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.bridge_width import WidthRule, pair_targets
from stencilizer.variable.crossings import Pt
from stencilizer.variable.model import glyph_coordinates, with_coordinates
from stencilizer.variable.realign import (
    Placed,
    Span,
    axis_value,
    contour_spans,
    lerp,
    mean_target,
    project,
    realign_line,
)
from stencilizer.variable.solver import Coords

_VERTEX_DECIMALS = 6
_EDGE_DISTANCE = 1e-4
_EDGE_T_SLACK = 1e-9
_LINE_DECIMALS = 5
# Core surgery's point dedup at 1000 UPM; callers pass their geometry's snap distance.
_SNAP_DISTANCE = 0.5

Slot = tuple[int, int]
"""An output point: (contour index, point index within that contour)."""


@dataclass(frozen=True, slots=True)
class Vertex:
    """An output point equal to input point ``index`` (and to its unresolved ``ties``)."""

    index: int
    ties: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class EdgePoint:
    """An output point at parameter ``t`` on the input edge ``a`` -> ``b``."""

    a: int
    b: int
    t: float


Source = Vertex | EdgePoint
_Candidate = list[int] | EdgePoint


@dataclass(frozen=True, slots=True)
class LineMember:
    """An output point on a bridge line and the input edges it lies on.

    ``edges`` are the first and last input edge by start index: one for a cut point, two
    around a vertex.
    """

    slot: Slot
    edges: tuple[int, int]
    cut: bool


@dataclass(frozen=True, slots=True)
class BridgeLine:
    """Points on ``x = coordinate`` (axis 0) or ``y = coordinate`` (axis 1) in the default."""

    axis: int
    coordinate: float
    members: tuple[LineMember, ...]

    @property
    def slots(self) -> list[Slot]:
        """The output point of every member."""
        return [member.slot for member in self.members]


@dataclass(frozen=True, slots=True)
class SurgeryMap:
    """The source of every output point, per output contour, and the bridge lines.

    Without ``widths`` every line goes to the mean of its cut points in each master. With
    it, each bridge's two lines go to the bridge's centre -/+ half its gap in that master.
    """

    sources: tuple[tuple[Source, ...], ...]
    lines: tuple[BridgeLine, ...]
    widths: WidthRule | None = None


def slot_values(glyph: Glyph, slots: Sequence[Slot], axis: int) -> list[float]:
    """The ``axis`` coordinate of each slot's point in ``glyph``."""
    return [axis_value(glyph.contours[ci].points[pi], axis) for ci, pi in slots]


def _key(point: Pt) -> Pt:
    return (round(point[0], _VERTEX_DECIMALS), round(point[1], _VERTEX_DECIMALS))


def _edge_point(point: Pt, coords: Coords, spans: list[Span]) -> EdgePoint | None:
    """The input edge nearest to ``point`` within the mapping distance, if any."""
    best: EdgePoint | None = None
    best_distance = _EDGE_DISTANCE
    px, py = point
    for a, (start, length) in enumerate(spans):
        b = start + (a - start + 1) % length
        (ax, ay), (bx, by) = coords[a], coords[b]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 == 0.0:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / length2
        if not -_EDGE_T_SLACK <= t <= 1.0 + _EDGE_T_SLACK:
            continue
        t = min(1.0, max(0.0, t))
        distance = math.hypot(ax + t * dx - px, ay + t * dy - py)
        if distance < best_distance:
            best, best_distance = EdgePoint(a, b, t), distance
    return best


def _neighbour_owners(owners: list[frozenset[int]], k: int) -> frozenset[int]:
    """Contours of the nearest unambiguous output points before and after point ``k``."""
    found: frozenset[int] = frozenset()
    count = len(owners)
    for step in (1, -1):
        for distance in range(1, count):
            other = owners[(k + step * distance) % count]
            if len(other) == 1:
                found |= other
                break
    return found


def _resolve(row: list[_Candidate], spans: list[Span]) -> tuple[Source, ...]:
    """Pick one input vertex per ambiguous point, preferring its neighbours' contour."""
    owners = [
        frozenset({spans[c.a][0]} if isinstance(c, EdgePoint) else {spans[i][0] for i in c})
        for c in row
    ]
    sources: list[Source] = []
    for k, candidate in enumerate(row):
        if isinstance(candidate, EdgePoint):
            sources.append(candidate)
            continue
        if len(owners[k]) > 1:
            nearby = _neighbour_owners(owners, k)
            candidate = [i for i in candidate if spans[i][0] in nearby] or candidate
        sources.append(Vertex(candidate[0], tuple(candidate[1:])))
    return tuple(sources)


def _edges_on(source: Source, spans: list[Span]) -> tuple[int, int]:
    """First and last input edge (by start index) that an output point lies on."""
    if isinstance(source, EdgePoint):
        return (source.a, source.a)
    start, length = spans[source.index]
    return (start + (source.index - start - 1) % length, source.index)


def _bridge_axes(contour: Contour, row: tuple[Source, ...], spans: list[Span]) -> list[set[int]]:
    """Per output point, the axes of its adjacent segments that surgery created.

    A segment is inherited when both ends lie on one input edge. Any other segment runs
    along a bridge line: mostly vertical is an x line (axis 0), else a y line (axis 1).
    """
    on = [set(_edges_on(source, spans)) for source in row]
    axes: list[set[int]] = [set() for _ in row]
    count = len(row)
    for k in range(count if count > 1 else 0):
        n = (k + 1) % count
        if on[k] & on[n]:
            continue
        p, q = contour.points[k], contour.points[n]
        axis = 0 if abs(q.x - p.x) <= abs(q.y - p.y) else 1
        axes[k].add(axis)
        axes[n].add(axis)
    return axes


def _cut_axis(source: EdgePoint, axes: set[int], coords: Coords) -> int:
    """The axis of a cut point's bridge segment; without exactly one, its edge decides.

    A cut on a mostly horizontal edge then lies on an x line, and the other way round,
    so cuts on one vertical stem never share an x line.
    """
    if len(axes) == 1:
        return next(iter(axes))
    (ax, ay), (bx, by) = coords[source.a], coords[source.b]
    return 0 if abs(bx - ax) >= abs(by - ay) else 1


_Groups = dict[tuple[int, float], list[LineMember]]
_Loose = list[tuple[Slot, int, Vertex, float]]


def _collect(
    output_glyph: Glyph, sources: list[tuple[Source, ...]], coords: Coords, spans: list[Span]
) -> tuple[_Groups, _Loose]:
    """Cut points keyed by bridge line, and vertices at the ends of bridge segments."""
    groups: _Groups = defaultdict(list)
    loose: _Loose = []
    for ci, (contour, row) in enumerate(zip(output_glyph.contours, sources, strict=True)):
        axes = _bridge_axes(contour, row, spans)
        for pi, (point, source) in enumerate(zip(contour.points, row, strict=True)):
            if isinstance(source, Vertex):
                loose += [((ci, pi), axis, source, axis_value(point, axis)) for axis in axes[pi]]
                continue
            axis = _cut_axis(source, axes[pi], coords)
            key = (axis, round(axis_value(point, axis), _LINE_DECIMALS))
            groups[key].append(LineMember((ci, pi), (source.a, source.a), cut=True))
    return groups, loose


def _bridge_lines(
    output_glyph: Glyph,
    sources: list[tuple[Source, ...]],
    coords: Coords,
    spans: list[Span],
    snap: float,
) -> tuple[BridgeLine, ...]:
    """Every bridge line with at least two points.

    A vertex ending a bridge segment joins the nearest line of that axis within ``snap``:
    surgery kept it in place of a cut point.
    """
    groups, loose = _collect(output_glyph, sources, coords, spans)
    centres = {
        key: fmean(slot_values(output_glyph, [member.slot for member in members], key[0]))
        for key, members in groups.items()
    }
    for slot, axis, source, value in loose:
        near = [k for k in groups if k[0] == axis and abs(centres[k] - value) <= snap]
        if not near:
            continue
        key = min(near, key=lambda k: abs(centres[k] - value))
        groups[key].append(LineMember(slot, _edges_on(source, spans), cut=False))
    return tuple(
        BridgeLine(key[0], centres[key], tuple(members))
        for key, members in groups.items()
        if len(members) >= 2
    )


def map_surgery(
    input_glyph: Glyph, output_glyph: Glyph, snap: float = _SNAP_DISTANCE
) -> SurgeryMap | None:
    """Source of every output point in the (polygon) input, or None when one has none."""
    coords = glyph_coordinates(input_glyph)
    spans = contour_spans(input_glyph)
    vertices: dict[Pt, list[int]] = defaultdict(list)
    for index, coord in enumerate(coords):
        vertices[_key(coord)].append(index)
    sources: list[tuple[Source, ...]] = []
    for contour in output_glyph.contours:
        row: list[_Candidate] = []
        for point in contour.points:
            position = (point.x, point.y)
            candidate = vertices.get(_key(position)) or _edge_point(position, coords, spans)
            if candidate is None:
                return None
            row.append(candidate)
        sources.append(_resolve(row, spans))
    return SurgeryMap(tuple(sources), _bridge_lines(output_glyph, sources, coords, spans, snap))


def _place(smap: SurgeryMap, master: Coords) -> Placed | None:
    """Vertices at their master positions, cut points at their default edge parameter."""
    placed: Placed = []
    for sources in smap.sources:
        row: list[list[float]] = []
        for source in sources:
            if isinstance(source, EdgePoint):
                row.append(lerp(master[source.a], master[source.b], source.t))
                continue
            position = master[source.index]
            if any(_key(master[i]) != _key(position) for i in source.ties):
                return None
            row.append(list(position))
        placed.append(row)
    return placed


def _stops(smap: SurgeryMap) -> frozenset[Slot]:
    """Line points and cut points: the slots projection never moves or walks past."""
    members = {slot for line in smap.lines for slot in line.slots}
    cuts = {
        (ci, pi)
        for ci, sources in enumerate(smap.sources)
        for pi, source in enumerate(sources)
        if isinstance(source, EdgePoint)
    }
    return frozenset(members | cuts)


def replay(
    smap: SurgeryMap, input_default: Glyph, output_default: Glyph, master_input: Glyph
) -> Glyph | None:
    """The default surgery output rebuilt from one master of its input, or None.

    Point types and directions come from ``output_default``. None when a line point finds
    no crossing master edge, or an ambiguous vertex's candidates differ in the master.
    """
    master = glyph_coordinates(master_input)
    spans = contour_spans(input_default)
    placed = _place(smap, master)
    if placed is None:
        return None
    targets = [mean_target(line, placed) for line in smap.lines]
    if smap.widths is not None:
        targets = pair_targets(smap.widths, smap.lines, placed, master_input, targets)
    stops = _stops(smap)
    for line, target in zip(smap.lines, targets, strict=True):
        if not realign_line(line, target, placed, master, spans):
            return None
        for member in line.members:
            reference = output_default.contours[member.slot[0]].points
            project(member.slot, line, target, placed, reference, stops)
    return with_coordinates(output_default, [(p[0], p[1]) for row in placed for p in row])
