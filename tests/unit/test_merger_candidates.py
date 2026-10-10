"""Tests for the fallback bridge line positions tried when the bbox-centre line fails."""

import pytest

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.core import merger as merger_module
from stencilizer.core import merger_candidates
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.axis import HORIZONTAL, VERTICAL, Axis
from stencilizer.core.geometry import find_all_edge_crossings
from stencilizer.core.merger import ContourMerger
from stencilizer.core.merger_candidates import (
    _band_extent,
    _candidate_lines,
    merge_at_candidates,
)
from stencilizer.core.merger_checks import measure
from stencilizer.core.merger_dispatch import MergeDispatch
from stencilizer.core.surgery import GlyphTransformer, TransformOutcome
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point
from stencilizer.variable.transform import transform_variable_glyph
from tests.font_helpers import INTER, island_count
from tests.unit._variable_cases import read_variable

UPM = 2048
BRIDGE_WIDTH = 0.56 * UPM * 0.1

# Inter "four" after flatten and overlap merge: the outer's diagonal puts the counter's left
# edge and top edge out of reach of every bbox-extreme test at the bbox centre.
FOUR_OUTER = [
    (120, 303),
    (120, 458),
    (772, 1490),
    (1003, 1490),
    (1003, 470),
    (1205, 470),
    (1205, 303),
    (1003, 303),
    (1003, 0),
    (821, 0),
    (821, 303),
]
FOUR_COUNTER = [(822, 470), (822, 1252), (810, 1252), (323, 482), (323, 470)]


def _contour(coordinates: list[tuple[int, int]], closed: bool) -> Contour:
    """A contour, optionally repeating the first point at the end as the merged font does."""
    return Contour([Point(float(x), float(y)) for x, y in coordinates + coordinates[:closed]])


def _four(closed: bool = False) -> Glyph:
    return Glyph(
        metadata=GlyphMetadata("four", None, 1200, 0),
        contours=[_contour(FOUR_OUTER, False), _contour(FOUR_COUNTER, closed)],
    )


def _transform(glyph: Glyph, direction: BridgeDirection) -> TransformOutcome:
    transformer = GlyphTransformer(
        analyzer=GlyphAnalyzer(),
        bridge_config=BridgeConfig(direction=direction),
        geometry_config=GeometryConfig(),
    )
    return transformer.transform_with_outcome(glyph, upm=UPM)


def _chord_span(dispatch: MergeDispatch, line: float) -> float:
    crossings = find_all_edge_crossings(dispatch.inner, line, HORIZONTAL.is_x)
    return max(c[2] for c in crossings) - min(c[2] for c in crossings)


def _dispatch(glyph: Glyph) -> MergeDispatch:
    outer, inner, *_ = glyph.contours
    geometry = GeometryConfig()
    measured = measure(inner, outer, BRIDGE_WIDTH, geometry, UPM)
    return MergeDispatch(inner, outer, measured, glyph.contours, [], geometry, UPM)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("direction", list(BridgeDirection))
def test_four_counter_is_bridged(direction: BridgeDirection, closed: bool) -> None:
    outcome = _transform(_four(closed), direction)
    assert outcome.bridge_count >= 1
    assert outcome.unbridged_count == 0
    assert island_count(outcome.glyph, UPM) == 0


def test_merger_returns_split_contours_for_four() -> None:
    outer, inner = _four().contours
    nested: list[Contour] = []
    merged = ContourMerger().merge_contours_with_bridges(
        inner, outer, BRIDGE_WIDTH, all_contours=[outer, inner], processed_nested=nested, upm=UPM
    )
    assert merged != [outer, inner]
    assert len(merged) == 4
    assert nested == []


def test_variable_four_is_bridged_in_every_master() -> None:
    vg, upm = read_variable(INTER, "4")
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count >= 1
    assert outcome.unbridged_count == 0
    assert outcome.glyph is not vg


def test_candidate_lines_start_at_centre_then_widest_chord() -> None:
    dispatch = _dispatch(_four())
    lines = _candidate_lines(dispatch, HORIZONTAL)
    assert lines[0] == pytest.approx((470 + 1252) / 2)
    assert len(lines) == 7
    spans = [_chord_span(dispatch, line) for line in lines[1:]]
    assert spans == sorted(spans, reverse=True)


def test_band_extent_spans_the_whole_band() -> None:
    dispatch = _dispatch(_four())
    half = dispatch.measure.half_width
    extent = _band_extent(dispatch, HORIZONTAL, 700.0)
    upper = _band_extent(dispatch, HORIZONTAL, 700.0 + half)
    assert extent is not None
    assert upper is not None
    # The counter's left edge slopes right going up, so the lower band line sets the minimum;
    # the right edge is the vertical x=822 line.
    assert extent[0] < upper[0]
    assert extent[1] == pytest.approx(822.0)


def test_band_extent_is_none_outside_the_counter() -> None:
    dispatch = _dispatch(_four())
    assert _band_extent(dispatch, HORIZONTAL, 1300.0) is None
    assert _band_extent(dispatch, VERTICAL, 100.0) is None


def test_no_candidate_is_tried_when_the_bbox_gates_failed() -> None:
    dispatch = _dispatch(_four())
    failed = [dispatch.outer, dispatch.inner]
    assert merge_at_candidates(dispatch, (False, False), vertical_first=False) == failed


def test_horizontal_candidates_build_for_four() -> None:
    dispatch = _dispatch(_four())
    result = merge_at_candidates(dispatch, (True, False), vertical_first=False)
    assert result != [dispatch.outer, dispatch.inner]


def test_no_candidate_band_crosses_a_sibling_hole() -> None:
    # A slot in the stem spans every candidate line: each horizontal band would cross it and
    # leave it whole across the gap, so the fallback must refuse them all.
    slot = _contour([(900, 480), (930, 480), (930, 1240), (900, 1240)], False)
    four = _four()
    dispatch = _dispatch(Glyph(metadata=four.metadata, contours=[*four.contours, slot]))
    failed = [dispatch.outer, dispatch.inner]
    assert merge_at_candidates(dispatch, (True, False), vertical_first=False) == failed


def _slot_dispatch() -> MergeDispatch:
    """Four with a horizontal slot under the counter that every vertical band would cross."""
    slot = _contour([(150, 360), (800, 360), (800, 420), (150, 420)], False)
    four = _four()
    return _dispatch(Glyph(metadata=four.metadata, contours=[*four.contours, slot]))


def _record_builds(monkeypatch: pytest.MonkeyPatch) -> list[Axis]:
    """Make every candidate build fail, recording the axis it was attempted on."""
    axes: list[Axis] = []

    def failing_build(
        dispatch: MergeDispatch, axis: Axis, _line: float, _extent: object
    ) -> list[Contour]:
        axes.append(axis)
        return [dispatch.outer, dispatch.inner]

    monkeypatch.setattr(merger_candidates, "_build", failing_build)
    return axes


def test_no_candidate_band_crosses_a_sibling_hole_vertically() -> None:
    dispatch = _dispatch(_four())
    failed = [dispatch.outer, dispatch.inner]
    assert merge_at_candidates(dispatch, (False, True), vertical_first=False) != failed
    blocked = _slot_dispatch()
    failed = [blocked.outer, blocked.inner]
    assert merge_at_candidates(blocked, (False, True), vertical_first=False) == failed


@pytest.mark.parametrize(("vertical_first", "first"), [(False, HORIZONTAL), (True, VERTICAL)])
def test_axis_order_follows_vertical_first(
    monkeypatch: pytest.MonkeyPatch, vertical_first: bool, first: Axis
) -> None:
    axes = _record_builds(monkeypatch)
    merge_at_candidates(_dispatch(_four()), (True, True), vertical_first=vertical_first)
    assert axes[0] is first
    assert {HORIZONTAL, VERTICAL} <= set(axes)


def test_vertical_only_gate_skips_horizontal(monkeypatch: pytest.MonkeyPatch) -> None:
    axes = _record_builds(monkeypatch)
    merge_at_candidates(_dispatch(_four()), (False, True), vertical_first=False)
    assert axes
    assert set(axes) == {VERTICAL}


def test_failed_candidate_builds_roll_back_nested_contours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = _dispatch(_four())
    earlier, stray = _contour(FOUR_COUNTER, False), _contour(FOUR_OUTER, False)
    nested = [earlier]
    dispatch.processed_nested = nested
    axes: list[Axis] = []

    def appending_build(
        d: MergeDispatch, axis: Axis, _line: float, _extent: object
    ) -> list[Contour]:
        axes.append(axis)
        nested.append(stray)
        return [d.outer, d.inner]

    monkeypatch.setattr(merger_candidates, "_build", appending_build)
    result = merge_at_candidates(dispatch, (True, False), vertical_first=False)
    assert result == [dispatch.outer, dispatch.inner]
    assert len(axes) > 1
    assert nested == [earlier]


def _stub_failed_bbox_centre(monkeypatch: pytest.MonkeyPatch, sentinel: Contour) -> None:
    """Make the bbox-centre path append a nested piece, then report failure."""

    def failing(dispatch: MergeDispatch, _force_h: bool, _force_v: bool) -> list[Contour]:
        assert dispatch.processed_nested is not None
        dispatch.processed_nested.append(sentinel)
        return [dispatch.outer, dispatch.inner]

    monkeypatch.setattr(merger_module, "_bbox_centre_merge", failing)


def _merge(glyph: Glyph, nested: list[Contour]) -> list[Contour]:
    outer, inner, *_ = glyph.contours
    return ContourMerger().merge_contours_with_bridges(
        inner, outer, BRIDGE_WIDTH, all_contours=glyph.contours, processed_nested=nested, upm=UPM
    )


def test_nested_pieces_of_a_failed_bbox_centre_merge_are_dropped_on_fallback_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel, earlier = _contour(FOUR_COUNTER, False), _contour(FOUR_OUTER, False)
    _stub_failed_bbox_centre(monkeypatch, sentinel)
    nested = [earlier]
    outer, inner = _four().contours
    assert _merge(_four(), nested) != [outer, inner]
    assert nested == [earlier]


def test_nested_pieces_of_a_failed_bbox_centre_merge_stay_on_total_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel, earlier = _contour(FOUR_COUNTER, False), _contour(FOUR_OUTER, False)
    _stub_failed_bbox_centre(monkeypatch, sentinel)
    stem_slot = _contour([(900, 480), (930, 480), (930, 1240), (900, 1240)], False)
    foot_slot = _contour([(150, 360), (800, 360), (800, 420), (150, 420)], False)
    four = _four()
    glyph = Glyph(metadata=four.metadata, contours=[*four.contours, stem_slot, foot_slot])
    nested = [earlier]
    merged = _merge(glyph, nested)
    assert merged == [glyph.contours[0], glyph.contours[1]]
    assert nested == [earlier, sentinel]
