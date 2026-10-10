"""Tests for the variable glyph model, solver, reader, rounding and flattening."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.exceptions import VariationDataError
from stencilizer.variable import flatten_compatible
from stencilizer.variable.flatten import MAX_SUBDIVISIONS, _segments, _subdivisions
from stencilizer.variable.model import Support, VariableGlyph, glyph_coordinates
from stencilizer.variable.reader import cff2_vsindex, read_variable_glyph
from stencilizer.variable.rounding import round_variable_glyph
from stencilizer.variable.solver import solve_deltas
from tests.font_helpers import CANTARELL, INTER, UBUNTU, fvar_only_roboto, units_per_em


def _name(font: TTFont, char: str) -> str:
    return str(font.getBestCmap()[ord(char)])


def _read(path: Path, char: str) -> VariableGlyph:
    font = TTFont(path)
    vg = read_variable_glyph(font, _name(font, char))
    assert vg is not None
    return vg


def _square(dx: float, size: float = 500.0) -> Glyph:
    corners = [(0.0, 0.0), (0.0, size), (size, size), (size, 0.0)]
    points = [Point(x + dx, y) for x, y in corners]
    return Glyph(metadata=GlyphMetadata("sq", None, 600, 0), contours=[Contour(points)])


def test_support_scalar_cases() -> None:
    support = Support((("wght", 0.0, 1.0, 1.0),))
    assert support.scalar({"wght": 0.5}) == pytest.approx(0.5)
    assert support.scalar({"wght": -0.5}) == 0.0
    assert support.scalar({"wght": 1.0}) == pytest.approx(1.0)
    assert support.scalar({}) == 0.0
    assert support.peak() == {"wght": 1.0}
    assert Support((("wght", -1.0, 0.0, 1.0),)).scalar({"wght": 0.7}) == 1.0
    assert Support((("wght", -1.0, 0.5, 1.0),)).scalar({}) == 1.0


def test_dict_roundtrip_excludes_cached_deltas() -> None:
    vg = _read(UBUNTU, "o")
    vg.deltas()
    data = vg.to_dict()
    assert not any("delta" in key for key in data)
    assert VariableGlyph.from_dict(data) == vg


def test_instance_at_peak_equals_master() -> None:
    vg = _read(UBUNTU, "o")
    for support, master in zip(vg.supports, vg.masters, strict=True):
        instance = vg.instance(support.peak())
        for (ax, ay), (bx, by) in zip(
            glyph_coordinates(instance), glyph_coordinates(master), strict=True
        ):
            assert ax == pytest.approx(bx, abs=1e-6)
            assert ay == pytest.approx(by, abs=1e-6)


def test_mismatched_master_structure_raises() -> None:
    bad = _square(0.0)
    bad.contours[0].points.pop()
    with pytest.raises(VariationDataError):
        VariableGlyph(_square(0.0), (Support((("wght", 0.0, 1.0, 1.0),)),), (bad,), ("wght",))


def test_solve_deltas_two_regions() -> None:
    supports = [Support((("wght", 0.0, 1.0, 1.0),)), Support((("wdth", 0.0, 1.0, 1.0),))]
    default = [(0.0, 0.0), (100.0, 0.0)]
    masters = [[(10.0, 0.0), (120.0, 5.0)], [(0.0, 7.0), (100.0, 9.0)]]
    deltas = solve_deltas(supports, default, masters)
    assert deltas[0] == [(10.0, 0.0), (20.0, 5.0)]
    assert deltas[1] == [(0.0, 7.0), (0.0, 9.0)]
    assert solve_deltas([], default, []) == []


def test_solve_deltas_overlapping_supports() -> None:
    supports = [Support((("wght", 0.0, 1.0, 1.0),)), Support((("wght", 0.0, 0.5, 1.0),))]
    deltas = solve_deltas(supports, [(0.0, 0.0)], [[(10.0, 0.0)], [(3.0, 0.0)]])
    # master at 1.0 = d0 + d1 * 0 (scalar of the 0.5 region at 1.0 is 0); at 0.5 = 0.5 * d0 + d1
    assert deltas[0][0][0] == pytest.approx(10.0)
    assert deltas[1][0][0] == pytest.approx(-2.0)


def test_solve_deltas_shared_peak_raises() -> None:
    first = Support((("wght", 0.0, 0.5, 1.0),))
    second = Support((("wght", 0.5, 0.5, 1.0),))
    with pytest.raises(VariationDataError, match="singular"):
        solve_deltas([first, second], [(0.0, 0.0)], [[(1.0, 0.0)], [(2.0, 0.0)]], glyph_name="x")


def test_fvar_without_gvar_reads_no_supports() -> None:
    font = fvar_only_roboto()
    vg = read_variable_glyph(font, _name(font, "o"))
    assert vg is not None
    assert vg.supports == ()
    assert vg.masters == ()
    assert vg.axis_tags == ("wght",)
    assert vg.instance({"wght": 1.0}) == vg.default


def test_cff2_vsindex_and_supports() -> None:
    font = TTFont(CANTARELL)
    name = _name(font, "o")
    assert cff2_vsindex(font, name) == 0
    vg = read_variable_glyph(font, name)
    assert vg is not None
    assert vg.cff2
    assert len(vg.supports) == 2


def test_cff2_vsindex_two_indices_raises() -> None:
    class Charstring:
        def draw(self, _pen: Any, blender: Any) -> None:
            blender(0, [1.0])
            blender(1, [1.0])

    top_dict = SimpleNamespace(CharStrings={"g": Charstring()})
    cff2 = SimpleNamespace(cff=SimpleNamespace(topDictIndex=[top_dict]))
    with pytest.raises(VariationDataError, match="more than one vsindex"):
        cff2_vsindex({"CFF2": cff2}, "g")


@pytest.mark.parametrize("path", [UBUNTU, CANTARELL])
def test_round_variable_glyph(path: Path) -> None:
    vg = _read(path, "o")
    rounded = round_variable_glyph(vg)
    again = round_variable_glyph(rounded)
    for a, b in zip(
        glyph_coordinates(rounded.default), glyph_coordinates(again.default), strict=True
    ):
        assert a == pytest.approx(b, abs=1e-9)
    for m1, m2 in zip(rounded.masters, again.masters, strict=True):
        for a, b in zip(glyph_coordinates(m1), glyph_coordinates(m2), strict=True):
            assert a == pytest.approx(b, abs=1e-9)
    assert all(
        float(x).is_integer() and float(y).is_integer()
        for x, y in glyph_coordinates(rounded.default)
    )
    if not vg.cff2:
        for per_support in rounded.deltas():
            assert all(abs(v - round(v)) <= 1e-9 for d in per_support for v in d)
    for support in vg.supports:
        peak = support.peak()
        for (ax, ay), (bx, by) in zip(
            glyph_coordinates(rounded.instance(peak)),
            glyph_coordinates(vg.instance(peak)),
            strict=True,
        ):
            assert abs(ax - bx) <= 0.5 + 1e-9
            assert abs(ay - by) <= 0.5 + 1e-9


def test_round_keeps_equal_points_equal() -> None:
    support = Support((("wght", 0.0, 1.0, 1.0),))
    default = _square(0.4)
    master = _square(10.7)
    # corners 1 and 2 share x in default and master: they must round equally
    rounded = round_variable_glyph(VariableGlyph(default, (support,), (master,), ("wght",)))
    points = rounded.default.contours[0].points
    assert points[0].x == points[1].x
    peak_points = rounded.masters[0].contours[0].points
    assert peak_points[0].x == peak_points[1].x


def test_round_without_supports() -> None:
    vg = VariableGlyph(_square(0.4), (), (), ())
    rounded = round_variable_glyph(vg)
    assert rounded.masters == ()
    assert rounded.default.contours[0].points[0].x == 0.0


def _curve_counts(vg: VariableGlyph, upm: int) -> list[int]:
    counts = []
    glyphs = [vg.default, *vg.masters]
    for index, contour in enumerate(vg.default.contours):
        if all(p.point_type == PointType.ON_CURVE for p in contour.points):
            continue
        per_glyph = [_segments(g.contours[index].points, vg.name) for g in glyphs]
        for k in range(len(per_glyph[0])):
            counts.append(_subdivisions([s[k] for s in per_glyph], curve_tolerance(upm), vg.name))
    return counts


def test_flatten_compatible_structure_and_tolerance() -> None:
    font = TTFont(INTER)
    upm = units_per_em(font)
    counts: list[int] = []
    for char in "oPe8":
        vg = read_variable_glyph(font, _name(font, char))
        assert vg is not None
        flat = flatten_compatible(vg, upm)
        assert all(
            p.point_type == PointType.ON_CURVE for c in flat.default.contours for p in c.points
        )
        shape = [len(c.points) for c in flat.default.contours]
        assert all([len(c.points) for c in m.contours] == shape for m in flat.masters)
        for original, flattened in zip(
            [vg.default, *vg.masters], [flat.default, *flat.masters], strict=True
        ):
            for contour, poly in zip(original.contours, flattened.contours, strict=True):
                assert flatten_contour(poly, curve_tolerance(upm)) is poly
                reference = flatten_contour(contour, curve_tolerance(upm))
                perimeter = sum(
                    abs(a.x - b.x) + abs(a.y - b.y)
                    for a, b in zip(poly.points, poly.points[1:] + poly.points[:1], strict=True)
                )
                assert (
                    abs(poly.signed_area() - reference.signed_area())
                    <= curve_tolerance(upm) * perimeter
                )
        counts += _curve_counts(vg, upm)
    assert any(n not in {1, 2, 4, 8, 16, 32, 64} for n in counts)
    assert max(counts) <= MAX_SUBDIVISIONS


def test_flatten_polygon_passthrough_and_cap() -> None:
    support = Support((("wght", 0.0, 1.0, 1.0),))
    vg = VariableGlyph(_square(0.0), (support,), (_square(5.0),), ("wght",))
    assert flatten_compatible(vg, 1000) == vg
    huge = Contour([Point(0, 0), Point(0, 50000, PointType.OFF_CURVE_QUAD), Point(50000, 0)])
    glyph = Glyph(metadata=GlyphMetadata("big", None, 600, 0), contours=[huge])
    with pytest.raises(VariationDataError, match="more than 64"):
        flatten_compatible(VariableGlyph(glyph, (), (), ()), 1000)
