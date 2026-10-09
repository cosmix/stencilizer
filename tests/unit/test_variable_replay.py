"""Unit tests for surgery mapping, per-master replay, validation and the transform pipeline."""

from pathlib import Path

import pytest

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.geometry import signed_area
from stencilizer.core.surgery import GlyphTransformer, TransformOutcome
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable import transform
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import Support, VariableGlyph, glyph_coordinates, with_coordinates
from stencilizer.variable.replay import EdgePoint, Vertex, map_surgery, replay
from stencilizer.variable.validate import validate, validation_locations
from tests.font_helpers import CANTARELL, UBUNTU, points
from tests.font_helpers import island_count as _islands
from tests.unit._variable_cases import ABOVE, BELOW, STEM, UPM, WGHT, read_variable
from tests.unit._variable_cases import glyph_from_outlines as _glyph

OUTER = [(0.0, 0.0), (0.0, 700.0), (1000.0, 700.0), (1000.0, 0.0)]
LEFT_HOLE = [(100.0, 200.0), (300.0, 200.0), (300.0, 500.0), (100.0, 500.0)]
RIGHT_HOLE = [(700.0, 200.0), (900.0, 200.0), (900.0, 500.0), (700.0, 500.0)]


def _read(path: Path, char: str) -> tuple[VariableGlyph, int]:
    vg, upm = read_variable(path, char)
    assert vg is not None
    return vg, upm


def _scaled(glyph: Glyph, factor: float) -> Glyph:
    return with_coordinates(glyph, [(x * factor, y * factor) for x, y in glyph_coordinates(glyph)])


def _lines(source: Glyph, output: Glyph) -> list[tuple[int, float, list[tuple[int, int]]]]:
    smap = map_surgery(source, output)
    assert smap is not None
    return sorted(
        (line.axis, line.coordinate, sorted(member.slot for member in line.members))
        for line in smap.lines
    )


def test_replay_at_default_returns_surgery_output() -> None:
    vg, upm = _read(UBUNTU, "o")
    flat = flatten_compatible(vg, upm)
    outcome = GlyphTransformer(GlyphAnalyzer()).transform_with_outcome(flat.default, upm=upm)
    assert outcome.bridge_count >= 1
    smap = map_surgery(flat.default, outcome.glyph)
    assert smap is not None
    assert smap.lines
    replayed = replay(smap, flat.default, outcome.glyph, flat.default)
    assert replayed is not None
    shape = [len(contour.points) for contour in outcome.glyph.contours]
    assert [len(contour.points) for contour in replayed.contours] == shape
    for got, expected in zip(points(replayed), points(outcome.glyph), strict=True):
        assert got.point_type == expected.point_type
        assert got.x == pytest.approx(expected.x, abs=1e-9)
        assert got.y == pytest.approx(expected.y, abs=1e-9)


def test_cuts_on_one_vertical_stem_group_by_their_edge_axis() -> None:
    smap = map_surgery(_glyph(STEM), _glyph(BELOW, ABOVE))
    assert smap is not None
    assert isinstance(smap.sources[0][1], EdgePoint)
    assert isinstance(smap.sources[1][0], EdgePoint)
    assert _lines(_glyph(STEM), _glyph(BELOW, ABOVE)) == [
        (1, 100.0, [(0, 1), (0, 2)]),
        (1, 200.0, [(1, 0), (1, 4)]),
    ]


def test_cut_on_steep_edge_joins_the_vertical_line() -> None:
    triangle = [(0.0, 0.0), (50.0, 300.0), (100.0, 0.0)]
    left = [(0.0, 0.0), (40.0, 240.0), (40.0, 0.0)]
    right = [(40.0, 240.0), (50.0, 300.0), (100.0, 0.0), (40.0, 0.0)]
    assert _lines(_glyph(triangle), _glyph(left, right)) == [
        (0, 40.0, [(0, 1), (0, 2), (1, 0), (1, 3)])
    ]


def test_unmappable_point_gives_no_map() -> None:
    assert map_surgery(_glyph(STEM), _glyph([(0.0, 0.0), (50.0, 150.0), (0.0, 300.0)])) is None


def test_projection_snaps_wrong_side_vertex() -> None:
    source, output = _glyph(STEM), _glyph(BELOW, ABOVE)
    smap = map_surgery(source, output)
    assert smap is not None
    master = _glyph([(0.0, 0.0), (0.0, 300.0), (100.0, 300.0), (100.0, 190.0), (100.0, 0.0)])
    replayed = replay(smap, source, output, master)
    assert replayed is not None
    above = replayed.contours[1].points
    line = above[0].y
    assert line == pytest.approx((200.0 + 190.0 * 20.0 / 21.0) / 2.0)
    assert above[4].y == line
    assert above[3].y == line
    assert above[2].y == 300.0


def test_tied_vertex_that_splits_in_a_master_gives_no_replay() -> None:
    # Points 2 and 3 coincide in the default, so both output corners map to a tied vertex.
    source = _glyph([(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 0.0)])
    smap = map_surgery(source, source)
    assert smap is not None
    assert smap.sources[0][2] == Vertex(2, (3,))
    assert replay(smap, source, source, _scaled(source, 1.1)) is not None
    split = _glyph([(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (110.0, 100.0), (100.0, 0.0)])
    assert replay(smap, source, source, split) is None


def test_bridge_line_missing_a_narrowed_counter_gives_no_replay() -> None:
    hole = [(400.0, 200.0), (600.0, 200.0), (600.0, 500.0), (400.0, 500.0)]
    source = _glyph(OUTER, hole)
    bridge = BridgeConfig(direction=BridgeDirection.VERTICAL)
    outcome = GlyphTransformer(GlyphAnalyzer(), bridge_config=bridge).transform_with_outcome(
        source, upm=UPM
    )
    assert outcome.bridge_count == 1
    smap = map_surgery(source, outcome.glyph)
    assert smap is not None
    assert len(smap.lines) == 2
    assert replay(smap, source, outcome.glyph, _scaled(source, 1.1)) is not None
    # The counter narrows to x 490..510; its cut points keep their default edge parameter,
    # so each line's master coordinate falls outside 490..510 and crosses no counter edge.
    narrow = _glyph(OUTER, [(490.0, 200.0), (510.0, 200.0), (510.0, 500.0), (490.0, 500.0)])
    assert replay(smap, source, outcome.glyph, narrow) is None


def test_validation_locations_stay_inside_the_supported_range() -> None:
    vg, _ = _read(UBUNTU, "o")
    locations = validation_locations(vg)
    assert len(locations) == 6
    assert all(location.get("wdth", 0.0) <= 0.0 for location in locations)
    assert {"wdth": -1.0, "wght": 1.0} in locations


def test_validation_locations_reduce_a_large_grid() -> None:
    tags = tuple(f"ax{i}" for i in range(8))
    supports = [Support(((tag, -1.0, -1.0, 0.0),)) for tag in tags]
    supports += [Support(((tag, 0.0, 1.0, 1.0),)) for tag in tags]
    square = _glyph(OUTER)
    vg = VariableGlyph(square, tuple(supports), tuple(square for _ in supports), tags)
    locations = validation_locations(vg)
    assert len(locations) <= 1 + 16 + 16 + 2
    assert {} in locations
    assert dict.fromkeys(tags, -1.0) in locations
    assert dict.fromkeys(tags, 1.0) in locations
    for tag in tags:
        assert {tag: -1.0} in locations
        assert {tag: 1.0} in locations


def test_flattening_error_returns_input_with_islands_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vg, upm = _read(UBUNTU, "o")

    def fail(*_args: object) -> VariableGlyph:
        raise VariationDataError(vg.name, "curve needs more than 64 subdivisions")

    monkeypatch.setattr(transform, "flatten_compatible", fail)
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.glyph is vg
    assert (outcome.bridge_count, outcome.unbridged_count) == (0, 1)


def test_failed_overlap_removal_returns_input_with_islands_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vg, upm = _read(UBUNTU, "o")
    monkeypatch.setattr(transform, "remove_overlaps_compatible", lambda _vg: None)
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.glyph is vg
    assert (outcome.bridge_count, outcome.unbridged_count) == (0, 1)


def test_partial_surgery_replays_what_was_bridged(monkeypatch: pytest.MonkeyPatch) -> None:
    real = GlyphTransformer.transform_with_outcome

    def leave_right_hole(self: GlyphTransformer, glyph: Glyph, upm: int = 1000) -> TransformOutcome:
        holes = [c for c in glyph.contours if signed_area(c.points) > 0]
        right = max(holes, key=lambda c: min(point.x for point in c.points))
        kept = Glyph(glyph.metadata, [c for c in glyph.contours if c is not right])
        outcome = real(self, kept, upm)
        bridged = Glyph(glyph.metadata, [*outcome.glyph.contours, right])
        return TransformOutcome(bridged, outcome.bridge_count, 1)

    monkeypatch.setattr(GlyphTransformer, "transform_with_outcome", leave_right_hole)
    default = _glyph(OUTER, LEFT_HOLE, RIGHT_HOLE)
    vg = VariableGlyph(default, (WGHT,), (_scaled(default, 1.1),), ("wght",))
    bridge = BridgeConfig(direction=BridgeDirection.VERTICAL)
    outcome = transform.transform_variable_glyph(vg, bridge, GeometryConfig(), UPM)
    assert outcome.bridge_count >= 1
    assert outcome.unbridged_count == 1
    for location in validation_locations(outcome.glyph):
        assert _islands(outcome.glyph.instance(location), UPM) == 1, location
    assert not validate(outcome.glyph, UPM, allowed_islands=0)


def test_validate_holds_islands_to_the_allowance() -> None:
    default = _glyph(OUTER, LEFT_HOLE)
    vg = VariableGlyph(default, (WGHT,), (_scaled(default, 1.1),), ("wght",))
    assert validate(vg, UPM, allowed_islands=1)
    assert not validate(vg, UPM, allowed_islands=0)


@pytest.mark.parametrize(("path", "char"), [(UBUNTU, "0"), (CANTARELL, "b")])
def test_cut_kept_as_input_vertex_stays_on_its_line(path: Path, char: str) -> None:
    vg, upm = _read(path, char)
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count >= 1
    for location in validation_locations(outcome.glyph):
        assert _islands(outcome.glyph.instance(location), upm) == 0, location


def test_worker_reports_errors_with_the_glyph_name() -> None:
    broken = {"default": {"metadata": {"name": "broken"}}}
    result = transform.process_variable_glyph(broken, BridgeConfig().model_dump(), UPM, None)
    assert result["glyph_name"] == "broken"
    assert "error" in result
    assert "traceback" in result
