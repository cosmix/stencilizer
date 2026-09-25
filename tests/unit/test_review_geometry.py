"""Regression checks for curved outlines, nesting, and bridge outcomes."""

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.axis import VERTICAL
from stencilizer.core.bridge_segments import collect_segments
from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.core.geometry_crossings import find_all_edge_crossings
from stencilizer.core.merger import ContourMerger
from stencilizer.core.multi_island import merge_multi_island_vertical
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point, PointType


def _contour(coords: list[tuple[float, float]]) -> Contour:
    return Contour([Point(x, y) for x, y in coords])


def _glyph(*contours: Contour) -> Glyph:
    return Glyph(GlyphMetadata("review", None, 1000, 0), list(contours))


def _square(lo: float, hi: float, clockwise: bool) -> Contour:
    coords = [(lo, lo), (lo, hi), (hi, hi), (hi, lo)]
    return _contour(coords if clockwise else list(reversed(coords)))


def _curved_hole(left: float, bottom: float, right: float, top: float) -> Contour:
    return Contour(
        [
            Point(left, bottom),
            Point(right, bottom),
            Point(right, top),
            Point((left + right) / 2, top + 30, PointType.OFF_CURVE_QUAD),
            Point(left, top),
        ]
    )


def test_quadratic_crossing_and_clip_follow_curve() -> None:
    contour = Contour(
        [
            Point(0, 0),
            Point(50, 100, PointType.OFF_CURVE_QUAD),
            Point(100, 0),
            Point(100, -20),
            Point(0, -20),
        ]
    )
    crossings = find_all_edge_crossings(contour, 50, True)
    assert max(item[2] for item in crossings) == 50
    segments = collect_segments(contour.points, 50, VERTICAL, True, 0.001)
    assert any(abs(point.y - 50) < 0.001 for run in segments for point in run)
    assert all(point.point_type == PointType.ON_CURVE for run in segments for point in run)


def test_cubic_and_implied_quadratic_flatten_at_scaled_tolerance() -> None:
    cubic = Contour(
        [
            Point(0, 0),
            Point(0, 100, PointType.OFF_CURVE_CUBIC),
            Point(100, 100, PointType.OFF_CURVE_CUBIC),
            Point(100, 0),
            Point(100, -10),
        ]
    )
    for scale in (1, 2):
        scaled = Contour([Point(p.x * scale, p.y * scale, p.point_type) for p in cubic.points])
        flat = flatten_contour(scaled, curve_tolerance(1000 * scale))
        assert max(p.y for p in flat.points) == 75 * scale
        assert all(p.point_type == PointType.ON_CURVE for p in flat.points)
        assert len(flat.points) == len(flatten_contour(cubic, curve_tolerance(1000)).points)
    implied = Contour(
        [
            Point(0, 0),
            Point(25, 100, PointType.OFF_CURVE_QUAD),
            Point(75, 100, PointType.OFF_CURVE_QUAD),
            Point(100, 0),
        ]
    )
    flat = flatten_contour(implied, curve_tolerance(1000))
    assert Point(50, 100) in flat.points


def test_collinear_quadratic_overshoot_is_not_erased() -> None:
    contour = Contour(
        [
            Point(0, 0),
            Point(200, 0, PointType.OFF_CURVE_QUAD),
            Point(100, 0),
            Point(100, -20),
            Point(0, -20),
        ]
    )
    flat = flatten_contour(contour, curve_tolerance(1000))
    assert max(point.x for point in flat.points) > 130


def test_curve_control_outside_outer_does_not_block_island() -> None:
    outer = _contour([(-10, -30), (-10, 60), (110, 60), (110, -30)])
    hole = Contour(
        [
            Point(0, -20),
            Point(100, -20),
            Point(100, 0),
            Point(50, 100, PointType.OFF_CURVE_QUAD),
            Point(0, 0),
        ]
    )
    hierarchy = GlyphAnalyzer().analyze(_glyph(outer, hole))
    assert hierarchy.islands == [1]
    assert hierarchy.containment == {1: 0}


def test_overlapping_contours_have_no_cyclic_parents() -> None:
    a = _contour([(50, 50), (0, 100), (0, 0), (100, 0), (100, 100)])
    b = _contour([(60, 40), (150, -50), (150, 150), (50, 150), (50, -50)])
    tree = GlyphAnalyzer().analyze(_glyph(a, b)).nesting_tree
    assert tree is not None
    assert tree[0].parent is None
    assert tree[1].parent is None
    assert tree[0].depth == 0
    assert tree[1].depth == 0


def test_island_uses_immediate_containing_outer() -> None:
    outer = _square(0, 1000, True)
    nested_outer = _square(200, 800, True)
    hole = _square(300, 700, False)
    hierarchy = GlyphAnalyzer().analyze(_glyph(outer, nested_outer, hole))
    assert hierarchy.containment[2] == 1
    assert hierarchy.nesting_tree is not None
    assert hierarchy.nesting_tree[2].parent == 1


def test_domain_helpers_follow_clockwise_outer_convention_without_direction() -> None:
    outer = _square(0, 1000, True)
    hole = _square(200, 800, False)
    solid = _glyph(outer)
    assert not solid.has_islands()
    assert solid.get_islands() == []
    assert solid.get_outer_contours() == [outer]
    ring = _glyph(outer, hole)
    assert ring.has_islands()
    assert ring.get_islands() == [hole]
    assert ring.get_outer_contours() == [outer]


def test_unbridgeable_island_reports_no_success_and_preserves_outline() -> None:
    outer = _square(0, 1000, True)
    hole = _square(5, 995, False)
    glyph = _glyph(outer, hole)
    transformer = GlyphTransformer(GlyphAnalyzer())
    outcome = transformer.transform_with_outcome(glyph)
    assert outcome.glyph is glyph
    assert outcome.bridge_count == 0
    assert outcome.unbridged_count == 1
    assert transformer.transform(glyph) is glyph


def test_direct_merger_flattens_curved_hole_and_preserves_no_op_identity() -> None:
    outer = _square(0, 1000, True)
    inner = _curved_hole(300, 300, 700, 700)
    merger = ContourMerger()
    merged = merger.merge_contours_with_bridges(inner, outer, 60, all_contours=[outer, inner])
    assert len(merged) >= 4
    assert all(p.point_type == PointType.ON_CURVE for c in merged for p in c.points)

    near_edge = _curved_hole(5, 5, 995, 965)
    unchanged = merger.merge_contours_with_bridges(
        near_edge, outer, 60, all_contours=[outer, near_edge]
    )
    assert unchanged[0] is outer
    assert unchanged[1] is near_edge


def test_direct_multi_island_merge_uses_flattened_crossing_indexes() -> None:
    outer = _square(0, 1000, True)
    bottom = _curved_hole(300, 200, 700, 350)
    top = _curved_hole(300, 650, 700, 800)
    merged = merge_multi_island_vertical(
        outer, [bottom, top], 60, all_contours=[outer, bottom, top]
    )
    assert len(merged) >= 4
    assert all(p.point_type == PointType.ON_CURVE for c in merged for p in c.points)

    remote = _curved_hole(750, 650, 900, 800)
    unchanged = merge_multi_island_vertical(outer, [bottom, remote], 60)
    assert unchanged[0] is outer
    assert unchanged[1] is bottom
    assert unchanged[2] is remote


def test_partial_bridge_reports_remaining_island() -> None:
    glyph = _glyph(
        _square(0, 1000, True),
        _square(5, 100, False),
        _square(300, 700, False),
    )
    transformer = GlyphTransformer(
        GlyphAnalyzer(), bridge_config=BridgeConfig(use_spanning_bridges=False)
    )
    outcome = transformer.transform_with_outcome(glyph)
    assert outcome.bridge_count == 1
    assert outcome.unbridged_count == 1
    assert glyph.contours[1] in outcome.glyph.contours
