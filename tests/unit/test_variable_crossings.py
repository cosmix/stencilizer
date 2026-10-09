"""``settle_crossings`` collapses the union vertices a crossing slid past in a master.

Two overlapping boxes, A (vertices 0-3) and B (4-7), both clockwise. Their union has two
crossings: A's top with B's left side at slot 2, and A's right side with B's bottom at
slot 6.
"""

import pytest

from stencilizer.variable.crossings import (
    Crossing,
    Edge,
    Plan,
    Pt,
    Vertex,
    intersect,
    polygon_area,
    settle_crossings,
)
from stencilizer.variable.solver import Coords

DEFAULT: Coords = [
    (0.0, 0.0),
    (0.0, 100.0),
    (100.0, 100.0),
    (100.0, 0.0),
    (50.0, 50.0),
    (50.0, 150.0),
    (150.0, 150.0),
    (150.0, 50.0),
]
EDGES: list[Edge] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
PLAN: Plan = [
    [
        Vertex(0),
        Vertex(1),
        Crossing((1, 2), (4, 5)),
        Vertex(5),
        Vertex(6),
        Vertex(7),
        Crossing((2, 3), (7, 4)),
        Vertex(3),
    ]
]


def _moved(moves: dict[int, Pt]) -> Coords:
    """The default with the given input vertices moved."""
    return [moves.get(index, point) for index, point in enumerate(DEFAULT)]


def _replayed(points: Coords) -> list[list[Pt]]:
    """The union plan replayed on ``points``, crossings unsettled."""
    polygon: list[Pt] = []
    for step in PLAN[0]:
        if isinstance(step, Vertex):
            polygon.append(points[step.index])
            continue
        crossing = intersect(points, step.first, step.second)
        assert crossing is not None
        polygon.append(crossing)
    return [polygon]


def _settle(points: Coords) -> tuple[list[list[Pt]], list[list[Pt]]]:
    """The replay on ``points`` before and after ``settle_crossings``."""
    polygons = _replayed(points)
    before = [list(polygon) for polygon in polygons]
    settle_crossings(PLAN, EDGES, DEFAULT, points, polygons)
    assert [len(polygon) for polygon in polygons] == [len(steps) for steps in PLAN]
    return before, polygons


def _off_line(points: Coords, edge: Edge, point: Pt) -> float:
    """Zero when ``point`` lies on the line through ``edge``."""
    (ax, ay), (bx, by) = points[edge[0]], points[edge[1]]
    return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)


def test_default_and_uniformly_scaled_master_stay_as_replayed() -> None:
    for points in (DEFAULT, [(x * 1.25, y * 1.25) for x, y in DEFAULT]):
        before, after = _settle(points)
        assert after == before


def test_crossing_past_a_vertex_of_the_first_edge_collapses_it() -> None:
    # A's top-left corner moves right of B's left side, so the crossing of A's top with B's
    # left side lies beyond that corner and the outline would fold back on itself.
    points = _moved({1: (60.0, 100.0)})
    before, after = _settle(points)
    settled = after[0][2]
    assert settled == pytest.approx((50.0, 250.0 / 3.0))
    assert after[0][1] == settled
    # The crossing is re-intersected on A's left side (0 -> 1) and B's left side (4 -> 5).
    assert _off_line(points, (0, 1), settled) == pytest.approx(0.0, abs=1e-9)
    assert _off_line(points, (4, 5), settled) == pytest.approx(0.0, abs=1e-9)
    assert after[0][:1] == before[0][:1]
    assert after[0][3:] == before[0][3:]


def test_crossing_past_a_vertex_of_the_second_edge_collapses_it() -> None:
    # B's top-left corner drops below A's top, so the crossing lies beyond that corner on B's
    # left side.
    points = _moved({5: (50.0, 90.0)})
    before, after = _settle(points)
    settled = after[0][2]
    assert settled == pytest.approx((200.0 / 3.0, 100.0))
    assert after[0][3] == settled
    # The crossing is re-intersected on A's top (1 -> 2) and B's top (5 -> 6).
    assert _off_line(points, (1, 2), settled) == pytest.approx(0.0, abs=1e-9)
    assert _off_line(points, (5, 6), settled) == pytest.approx(0.0, abs=1e-9)
    assert after[0][:2] == before[0][:2]
    assert after[0][4:] == before[0][4:]


def test_polygon_area_is_positive_counter_clockwise() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert polygon_area(square) == 100.0
    assert polygon_area(square[::-1]) == -100.0
