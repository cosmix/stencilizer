"""Contracts for the variable replay engine (stage variable-engine)."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.core import GlyphAnalyzer
from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.exceptions import GlyphError
from tests.font_helpers import CANTARELL, INTER, UBUNTU, glyph_at, vsindex_cantarell
from tests.font_helpers import island_count as _islands
from tests.font_helpers import points as _points
from tests.font_helpers import units_per_em as _upm


def _glyph_name(font: TTFont, char: str) -> str:
    return str(font.getBestCmap()[ord(char)])


def _read(path: Path, char: str) -> tuple[TTFont, Any]:
    from stencilizer.variable.reader import read_variable_glyph

    font = TTFont(path)
    vg = read_variable_glyph(font, _glyph_name(font, char))
    assert vg is not None
    return font, vg


def _transform(path: Path, char: str, bridge: BridgeConfig | None = None) -> tuple[int, Any]:
    from stencilizer.variable.transform import transform_variable_glyph

    font, vg = _read(path, char)
    upm = _upm(font)
    return upm, transform_variable_glyph(vg, bridge or BridgeConfig(), GeometryConfig(), upm)


def _assert_close(left: Glyph, right: Glyph, tolerance: float) -> None:
    assert [len(c.points) for c in left.contours] == [len(c.points) for c in right.contours]
    for a, b in zip(_points(left), _points(right), strict=True):
        assert a.point_type == b.point_type
        assert abs(a.x - b.x) <= tolerance
        assert abs(a.y - b.y) <= tolerance


def _nonzero(location: dict[str, float]) -> dict[str, float]:
    return {axis: value for axis, value in location.items() if value != 0.0}


def _glyf_index_map(font: TTFont, name: str, default: Glyph) -> list[int | None]:
    """For each domain point of ``default``, its glyf point index (None for a closing duplicate)."""
    _, (_, end_pts, flags, _) = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)
    mapping: list[int | None] = []
    start = 0
    for k, contour in enumerate(default.contours):
        length = end_pts[k] - start + 1
        on_curve = [i for i in range(length) if flags[start + i] & 1]
        if not on_curve:
            mapping.append(None)
        first_on = on_curve[0] if on_curve else 0
        for p in range(len(contour.points) - (0 if on_curve else 1)):
            mapping.append(None if p == length else start + (first_on + p) % length)
        start = end_pts[k] + 1
    return mapping


def test_solver_reproduces_original_deltas() -> None:
    font, vg = _read(UBUNTU, "o")
    name = _glyph_name(font, "o")
    coords, (_, end_pts, _, _) = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)
    mapping = _glyf_index_map(font, name, vg.default)
    tuples = font["gvar"].variations[name]
    deltas = vg.deltas()
    assert len(vg.supports) == len(tuples) == 5
    for index, support in enumerate(vg.supports):
        matches = [
            tv for tv in tuples if {(tag, *r) for tag, r in tv.axes.items()} == set(support.axes)
        ]
        assert len(matches) == 1
        matches[0].calcInferredDeltas(coords, end_pts)
        expected = matches[0].coordinates
        assert len(deltas[index]) == len(mapping)
        for point, glyf_index in enumerate(mapping):
            if glyf_index is None:
                continue
            ex, ey = expected[glyf_index]
            dx, dy = deltas[index][point]
            assert dx == pytest.approx(ex, abs=1e-6)
            assert dy == pytest.approx(ey, abs=1e-6)


def test_singular_supports_raise() -> None:
    from stencilizer.variable.model import Support
    from stencilizer.variable.solver import solve_deltas

    support = Support(axes=(("wght", 0.0, 1.0, 1.0),))
    with pytest.raises(GlyphError) as raised:
        solve_deltas([support, support], [(0.0, 0.0)], [[(10.0, 0.0)], [(20.0, 5.0)]])
    assert raised.type.__name__ == "VariationDataError"


def test_cff2_variable_reader_uses_varstore_regions() -> None:
    font, vg = _read(CANTARELL, "o")
    name = _glyph_name(font, "o")
    assert len(vg.supports) == 2
    for support in vg.supports:
        peak = support.peak()
        expected = glyph_at(font, name, peak)
        _assert_close(vg.instance(peak), expected, 0.5)
    assert len(GlyphAnalyzer().analyze(vg.default, _upm(font)).get_islands()) == 1


def test_masters_share_structure() -> None:
    _, vg = _read(UBUNTU, "o")
    _, outcome = _transform(UBUNTU, "o")
    assert outcome.bridge_count >= 1
    result = outcome.glyph
    assert result.default != vg.default
    shape = [[p.point_type for p in c.points] for c in result.default.contours]
    assert len(result.masters) == len(result.supports) >= 1
    for master in result.masters:
        assert len(master.contours) == len(result.default.contours)
        assert [[p.point_type for p in c.points] for c in master.contours] == shape


def test_realigned_replay_valid_everywhere() -> None:
    from stencilizer.variable.validate import validation_locations

    for char in "o8":
        upm, outcome = _transform(UBUNTU, char)
        assert outcome.bridge_count >= 1, char
        locations = [*validation_locations(outcome.glyph), {"wght": -0.5, "wdth": -0.5}]
        for location in locations:
            assert _islands(outcome.glyph.instance(location), upm) == 0, (char, location)


def test_overlap_built_counter_bridged() -> None:
    from stencilizer.variable.validate import validation_locations

    upm, outcome = _transform(INTER, "P")
    assert outcome.bridge_count >= 1
    for location in validation_locations(outcome.glyph):
        assert _islands(outcome.glyph.instance(location), upm) == 0, location


def test_outcome_valid_or_untouched() -> None:
    from stencilizer.variable.reader import read_variable_glyph
    from stencilizer.variable.transform import transform_variable_glyph
    from stencilizer.variable.validate import validation_locations

    checked = 0
    for path in (UBUNTU, INTER, CANTARELL):
        font = TTFont(path)
        upm = _upm(font)
        for name in font.getGlyphOrder():
            vg = read_variable_glyph(font, name)
            if vg is None:
                continue
            outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
            checked += 1
            if outcome.bridge_count == 0:
                assert outcome.glyph == vg, (path.name, name)
                continue
            for location in validation_locations(outcome.glyph):
                count = _islands(outcome.glyph.instance(location), upm)
                assert count <= outcome.unbridged_count, (path.name, name, location)
    assert checked >= 60


def _square(name: str, dx: float) -> Glyph:
    corners = [(0.0, 0.0), (0.0, 500.0), (500.0, 500.0), (500.0, 0.0)]
    points = [Point(x + dx, y, PointType.ON_CURVE) for x, y in corners]
    return Glyph(metadata=GlyphMetadata(name, None, 600, 0), contours=[Contour(points=points)])


def _square_glyph(supports: list[Any], axis_tags: tuple[str, ...]) -> Any:
    from stencilizer.variable.model import VariableGlyph

    masters = tuple(_square("square", 10.0 * (i + 1)) for i in range(len(supports)))
    return VariableGlyph(_square("square", 0.0), tuple(supports), masters, axis_tags)


def test_validation_locations_cover_corners() -> None:
    from stencilizer.variable.model import Support
    from stencilizer.variable.validate import validation_locations

    supports = [
        Support(axes=(("wdth", -1.0, -1.0, 0.0),)),
        Support(axes=(("wght", -1.0, -1.0, 0.0),)),
        Support(axes=(("wght", 0.0, 1.0, 1.0),)),
    ]
    locations = [
        _nonzero(loc) for loc in validation_locations(_square_glyph(supports, ("wdth", "wght")))
    ]
    assert {"wdth": -1.0, "wght": 1.0} in locations
    assert {"wdth": -1.0, "wght": -1.0} in locations
    assert {} in locations
    assert all(loc.get("wdth", 0.0) <= 0.0 for loc in locations)

    tags = tuple(f"ax{i:02d}" for i in range(13))
    many = [Support(axes=((tag, -1.0, -1.0, 0.0),)) for tag in tags]
    many += [Support(axes=((tag, 0.0, 1.0, 1.0),)) for tag in tags]
    reduced = [_nonzero(loc) for loc in validation_locations(_square_glyph(many, tags))]
    assert len(reduced) <= 64
    assert {} in reduced
    assert dict.fromkeys(tags, -1.0) in reduced
    assert dict.fromkeys(tags, 1.0) in reduced
    for tag in tags:
        assert {tag: -1.0} in reduced
        assert {tag: 1.0} in reduced


def test_bridge_width_config_reaches_engine() -> None:
    _, narrow = _transform(UBUNTU, "o", BridgeConfig(width_percent=30))
    _, wide = _transform(UBUNTU, "o", BridgeConfig(width_percent=110))
    assert narrow.bridge_count >= 1
    assert wide.bridge_count >= 1
    assert _points(narrow.glyph.default) != _points(wide.glyph.default)


def test_variable_glyph_dict_roundtrip() -> None:
    from stencilizer.variable.model import VariableGlyph
    from stencilizer.variable.transform import process_variable_glyph, transform_variable_glyph

    font, vg = _read(UBUNTU, "o")
    upm = _upm(font)
    assert VariableGlyph.from_dict(vg.to_dict()) == vg
    args = (vg.to_dict(), BridgeConfig().model_dump(), upm, GeometryConfig().model_dump())
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        result = pool.submit(process_variable_glyph, *args).result()
    assert "error" not in result, result.get("error")
    assert result["bridges_added"] >= 1
    rebuilt = VariableGlyph.from_dict(result["glyph"])
    assert len(rebuilt.supports) == len(rebuilt.masters) == len(vg.supports) == 5
    expected = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm).glyph
    assert rebuilt == expected
    assert rebuilt.axis_tags == vg.axis_tags


def test_cff2_vsindex_selection() -> None:
    from stencilizer.variable.reader import cff2_vsindex, read_variable_glyph

    _, original = _read(CANTARELL, "o")
    font = vsindex_cantarell()
    name = _glyph_name(font, "o")
    assert cff2_vsindex(font, name) == 0
    top_dict = font["CFF2"].cff.topDictIndex[0]
    charstring = top_dict.CharStrings[name]
    charstring.decompile()
    charstring.program[:0] = [1, "vsindex"]
    assert cff2_vsindex(font, name) == 1
    selected = read_variable_glyph(font, name)
    assert selected is not None
    peaks = [s.peak() for s in selected.supports]
    assert peaks == [s.peak() for s in reversed(original.supports)]
    assert peaks != [s.peak() for s in original.supports]

    private_font = vsindex_cantarell()
    private_font["CFF2"].cff.topDictIndex[0].FDArray[0].Private.vsindex = 1
    with pytest.raises(GlyphError) as raised:
        read_variable_glyph(private_font, name)
    assert raised.type.__name__ == "VariationDataError"

    plain = TTFont(CANTARELL)
    static = [n for n in plain.getGlyphOrder() if cff2_vsindex(plain, n) is None]
    assert static
    vg = read_variable_glyph(plain, static[0])
    assert vg is not None
    assert vg.supports == ()


def test_rounded_glyph_is_validated() -> None:
    from stencilizer.variable.rounding import round_variable_glyph

    for path in (UBUNTU, CANTARELL):
        _, outcome = _transform(path, "o")
        assert outcome.bridge_count >= 1, path.name
        result = outcome.glyph
        rounded = round_variable_glyph(result)
        for location in [{}, *(s.peak() for s in result.supports)]:
            _assert_close(rounded.instance(location), result.instance(location), 1e-9)
        for point in _points(result.default):
            assert float(point.x).is_integer(), (path.name, point)
            assert float(point.y).is_integer(), (path.name, point)
        if path == UBUNTU:
            for per_support in result.deltas():
                for dx, dy in per_support:
                    assert abs(dx - round(dx)) <= 1e-9
                    assert abs(dy - round(dy)) <= 1e-9
