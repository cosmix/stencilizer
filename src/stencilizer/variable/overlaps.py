"""Remove overlaps once on the default outline and replay the merge in every master.

Merging each master on its own would give incompatible structures, so the union is
computed on the default with skia-pathops and every output vertex is mapped back to
the input: either an input vertex or the crossing of two input edges. The map is then
replayed on each master's own coordinates.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import pathops  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]

from stencilizer.core.geometry import signed_area
from stencilizer.domain.contour import Contour, Point
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.model import VariableGlyph, glyph_coordinates
from stencilizer.variable.solver import Coords

# pathops round-trips coordinates through float32: up to 1.2e-4 off at 2048 UPM.
VERTEX_TOLERANCE = 1e-3
EDGE_TOLERANCE = 2e-3
PARALLEL_EPSILON = 1e-12
# Projection parameter slack when testing whether a union vertex lies on an input edge.
EDGE_PARAMETER_SLACK = 1e-6
# A replayed outline may differ from its location's true union by this share of the
# union's area plus a floor in font units squared. Crossings that slide onto a
# neighbouring chord leave slivers of up to 1.9% (Cantarell `a`); strokes pulling
# apart leave the whole gap between them.
FIDELITY_RELATIVE = 0.03
FIDELITY_FLOOR = 1.0

_Pt = tuple[float, float]
_Polygon = list[_Pt]
_Edge = tuple[int, int]


@dataclass(frozen=True)
class Vertex:
    """An output vertex that is input vertex ``index`` (flat index over all contours)."""

    index: int


@dataclass(frozen=True)
class Crossing:
    """An output vertex where input edges ``first`` and ``second`` intersect."""

    first: _Edge
    second: _Edge


_Step = Vertex | Crossing
_Plan = list[list[_Step]]


def _polygons(glyph: Glyph) -> list[_Polygon]:
    return [[(p.x, p.y) for p in contour.points] for contour in glyph.contours]


def _point(args: tuple[float, float]) -> _Pt:
    return (float(args[0]), float(args[1]))


def _path(polygons: list[_Polygon]) -> pathops.Path:
    path = pathops.Path()
    pen = path.getPen()
    for polygon in polygons:
        if not polygon:
            continue
        pen.moveTo(polygon[0])
        for point in polygon[1:]:
            pen.lineTo(point)
        pen.closePath()
    return path


def _union(polygons: list[_Polygon]) -> list[_Polygon] | None:
    """Overlap-free polygons with clockwise outer contours; None when pathops fails."""
    try:
        result = pathops.simplify(_path(polygons), clockwise=True)
    except pathops.PathOpsError:
        return None
    recording = RecordingPen()
    result.draw(recording)
    out: list[_Polygon] = []
    current: _Polygon | None = None
    for operator, args in recording.value:
        if operator == "moveTo" and current is None:
            current = [_point(args[0])]
        elif operator == "lineTo" and current is not None:
            current.append(_point(args[0]))
        elif operator == "closePath" and current is not None:
            out.append(current)
            current = None
        else:
            return None
    return out if current is None else None


class _VertexGrid:
    """Nearest-input-vertex lookup over buckets of ``VERTEX_TOLERANCE`` size."""

    def __init__(self, points: Coords) -> None:
        self._points = points
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            self._cells[self._cell(point)].append(index)

    @staticmethod
    def _cell(point: _Pt) -> tuple[int, int]:
        return (math.floor(point[0] / VERTEX_TOLERANCE), math.floor(point[1] / VERTEX_TOLERANCE))

    def nearest(self, point: _Pt) -> int | None:
        """Nearest input vertex within tolerance; ties go to the lowest index."""
        cx, cy = self._cell(point)
        best: tuple[float, int] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in self._cells.get((cx + dx, cy + dy), ()):
                    q = self._points[index]
                    candidate = (math.hypot(q[0] - point[0], q[1] - point[1]), index)
                    if candidate[0] <= VERTEX_TOLERANCE and (best is None or candidate < best):
                        best = candidate
        return None if best is None else best[1]


def _edges(polygons: list[_Polygon]) -> list[_Edge]:
    edges: list[_Edge] = []
    start = 0
    for polygon in polygons:
        size = len(polygon)
        edges += [(start + i, start + (i + 1) % size) for i in range(size)]
        start += size
    return edges


def _edges_near(point: _Pt, points: Coords, edges: list[_Edge]) -> list[_Edge]:
    near: list[_Edge] = []
    for a, b in edges:
        (ax, ay), (bx, by) = points[a], points[b]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            continue
        t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq
        if -EDGE_PARAMETER_SLACK <= t <= 1 + EDGE_PARAMETER_SLACK and (
            math.hypot(ax + t * dx - point[0], ay + t * dy - point[1]) < EDGE_TOLERANCE
        ):
            near.append((a, b))
    return near


def _map(polygons: list[_Polygon], union: list[_Polygon]) -> _Plan | None:
    """Express every union vertex as an input vertex or a crossing of two input edges."""
    points = [point for polygon in polygons for point in polygon]
    grid = _VertexGrid(points)
    edges = _edges(polygons)
    plan: _Plan = []
    for polygon in union:
        steps: list[_Step] = []
        for point in polygon:
            index = grid.nearest(point)
            if index is not None:
                steps.append(Vertex(index))
                continue
            near = _edges_near(point, points, edges)
            if len(near) != 2:
                return None
            steps.append(Crossing(near[0], near[1]))
        plan.append(steps)
    return plan


def _intersect(points: Coords, first: _Edge, second: _Edge) -> _Pt | None:
    """Intersection of the lines through both edges; None when they are parallel.

    The point may lie past an edge end: a crossing slides along flattened curves from
    master to master. ``_faithful`` rejects replays where that moves the outline away
    from the location's true union, such as strokes that no longer overlap.
    """
    (ax, ay), (bx, by) = points[first[0]], points[first[1]]
    (cx, cy), (dx, dy) = points[second[0]], points[second[1]]
    denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denominator) < PARALLEL_EPSILON:
        return None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _replay(plan: _Plan, points: Coords) -> list[_Polygon] | None:
    out: list[_Polygon] = []
    for steps in plan:
        polygon: _Polygon = []
        for step in steps:
            if isinstance(step, Vertex):
                polygon.append(points[step.index])
                continue
            crossing = _intersect(points, step.first, step.second)
            if crossing is None:
                return None
            polygon.append(crossing)
        out.append(polygon)
    return out


def _faithful(replayed: list[_Polygon], polygons: list[_Polygon]) -> bool:
    """True when ``replayed`` covers the union of ``polygons`` within the fidelity tolerance.

    Measured as the area of their symmetric difference under non-zero winding, so a
    spike or bowtie built on extended edge lines counts with its full area.
    """
    try:
        reference = pathops.simplify(_path(polygons), clockwise=True)
        difference = pathops.op(_path(replayed), reference, pathops.PathOp.XOR)
    except pathops.PathOpsError:
        return False
    # pathops reports Path.area unsigned (measured: 100.0 for a square in either winding).
    return bool(abs(difference.area) <= FIDELITY_RELATIVE * abs(reference.area) + FIDELITY_FLOOR)


def _replay_faithful(plan: _Plan, glyph: Glyph) -> list[_Polygon] | None:
    """``plan`` replayed on ``glyph``; None when it degenerates or misses the true union."""
    replayed = _replay(plan, glyph_coordinates(glyph))
    if replayed is None or not _faithful(replayed, _polygons(glyph)):
        return None
    return replayed


def _area(polygon: _Polygon) -> float:
    return signed_area([Point(x, y) for x, y in polygon])


def _to_glyph(template: Glyph, polygons: list[_Polygon]) -> Glyph:
    contours = [Contour([Point(x, y) for x, y in polygon]) for polygon in polygons]
    return Glyph(
        metadata=template.metadata, contours=contours, _is_composite=template._is_composite
    )


def _cyclic_index_runs(polygons: list[_Polygon]) -> list[tuple[int, ...]]:
    """Per input contour, flat vertex indices with consecutive duplicate points removed."""
    runs: list[tuple[int, ...]] = []
    start = 0
    for polygon in polygons:
        kept = [start + i for i, point in enumerate(polygon) if i == 0 or point != polygon[i - 1]]
        if len(kept) > 1 and polygon[kept[-1] - start] == polygon[0]:
            kept.pop()
        runs.append(tuple(kept))
        start += len(polygon)
    return runs


def _canonical(indices: tuple[int, ...]) -> tuple[int, ...]:
    """Representative of a cycle up to start point and orientation."""
    if not indices:
        return indices
    pivot = indices.index(min(indices))
    forward = indices[pivot:] + indices[:pivot]
    backward = (forward[0], *reversed(forward[1:]))
    return min(forward, backward)


def _is_unchanged(polygons: list[_Polygon], plan: _Plan) -> bool:
    """True when the union is the input up to contour order, start points and orientation."""
    output: list[tuple[int, ...]] = []
    for steps in plan:
        if not all(isinstance(step, Vertex) for step in steps):
            return False
        output.append(tuple(step.index for step in steps if isinstance(step, Vertex)))
    inputs = [_canonical(run) for run in _cyclic_index_runs(polygons)]
    return sorted(inputs) == sorted(_canonical(run) for run in output)


def remove_overlaps_compatible(vg: VariableGlyph) -> VariableGlyph | None:
    """Overlap-free ``vg`` with one structure in every master, or None when not replayable.

    ``vg`` must already be flattened (``flatten_compatible``). A glyph without overlaps
    comes back as the same object, keeping its contour order and start points.
    """
    polygons = _polygons(vg.default)
    union = _union(polygons)
    if union is None:
        return None
    plan = _map(polygons, union)
    if plan is None:
        return None
    if _is_unchanged(polygons, plan):
        return vg
    merged_default = _replay_faithful(plan, vg.default)
    if merged_default is None:
        return None
    signs = [_area(polygon) for polygon in merged_default]
    merged_masters: list[Glyph] = []
    for master in vg.masters:
        replayed = _replay_faithful(plan, master)
        if replayed is None:
            return None
        if any(_area(p) * sign <= 0 for p, sign in zip(replayed, signs, strict=True)):
            return None
        merged_masters.append(_to_glyph(master, replayed))
    return VariableGlyph(
        _to_glyph(vg.default, merged_default),
        vg.supports,
        tuple(merged_masters),
        vg.axis_tags,
        vg.cff2,
    )
