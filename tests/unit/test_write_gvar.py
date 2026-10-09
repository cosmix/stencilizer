"""Tests for the TrueType gvar writer."""

import copy
from pathlib import Path

import pytest
from fontTools.misc.textTools import Tag  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.tables._f_v_a_r import Axis, table__f_v_a_r  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.variable.model import VariableGlyph, with_coordinates
from stencilizer.variable.reader import read_variable_glyph
from stencilizer.variable.write_gvar import write_truetype_variable_glyph

FIXTURES = Path(__file__).parent.parent / "fixtures"
UBUNTU = FIXTURES / "variable" / "Ubuntu-VF-subset.ttf"
ROBOTO = FIXTURES / "Roboto-Regular.ttf"


def _read(font: TTFont, name: str) -> VariableGlyph:
    vg = read_variable_glyph(font, name)
    assert vg is not None
    return vg


def _domain(font: TTFont, name: str, location: dict[str, float]) -> Glyph:
    glyph_set = font.getGlyphSet(location=location, normalized=True)
    return fonttools_glyph_to_domain(name, glyph_set[name], font)


def _points(glyph: Glyph) -> list[Point]:
    return [p for contour in glyph.contours for p in contour.points]


def _save(font: TTFont, path: Path) -> TTFont:
    font.save(path)
    return TTFont(path)


def _phantoms(
    font: TTFont, name: str
) -> dict[frozenset[tuple[str, tuple[float, ...]]], list[tuple[float, float]]]:
    coords, controls = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)
    result = {}
    for variation in font["gvar"].variations[name]:
        tv = copy.deepcopy(variation)
        tv.calcInferredDeltas(coords, controls[1])
        key = frozenset((tag, tuple(v)) for tag, v in tv.axes.items())
        result[key] = [
            (0.0, 0.0) if d is None else (float(d[0]), float(d[1])) for d in tv.coordinates[-4:]
        ]
    return result


def _shifted(vg: VariableGlyph, dx: float) -> VariableGlyph:
    """The glyph with every master moved right by ``dx``."""

    def move(glyph: Glyph) -> Glyph:
        return with_coordinates(glyph, [(p.x + dx, p.y) for p in _points(glyph)])

    return VariableGlyph(
        move(vg.default), vg.supports, tuple(move(m) for m in vg.masters), vg.axis_tags, vg.cff2
    )


def test_roundtrip_reproduces_peaks(tmp_path: Path) -> None:
    font = TTFont(UBUNTU)
    name = font.getBestCmap()[ord("o")]
    vg = _read(font, name)
    write_truetype_variable_glyph(font, vg)
    saved = _save(font, tmp_path / "out.ttf")
    for support in vg.supports:
        peak = support.peak()
        expected = vg.instance(peak)
        actual = _domain(saved, name, peak)
        for a, b in zip(_points(actual), _points(expected), strict=True):
            assert abs(a.x - b.x) <= 0.5
            assert abs(a.y - b.y) <= 0.5


def test_phantom_deltas_survive(tmp_path: Path) -> None:
    font = TTFont(UBUNTU)
    name = font.getBestCmap()[ord("o")]
    expected = _phantoms(font, name)
    assert any(d != (0.0, 0.0) for tail in expected.values() for d in tail)
    write_truetype_variable_glyph(font, _read(font, name))
    saved = _save(font, tmp_path / "out.ttf")
    assert _phantoms(saved, name) == expected


def test_lsb_follows_xmin(tmp_path: Path) -> None:
    font = TTFont(UBUNTU)
    name = font.getBestCmap()[ord("o")]
    old_x_min = font["glyf"][name].xMin
    advance, old_lsb = font["hmtx"].metrics[name]
    write_truetype_variable_glyph(font, _shifted(_read(font, name), 3))
    saved = _save(font, tmp_path / "out.ttf")
    assert saved["glyf"][name].xMin == old_x_min + 3
    assert saved["hmtx"].metrics[name] == (advance, old_lsb + 3)


def test_instructions_dropped_for_rewritten_glyph_only() -> None:
    font = TTFont(UBUNTU)
    names = font.getBestCmap()
    target, other = names[ord("o")], names[ord("O")]
    other_bytes = font["glyf"][other].compile(font["glyf"])
    write_truetype_variable_glyph(font, _read(font, target))
    assert font["glyf"][target].program.getBytecode() == b""
    assert font["glyf"][other].compile(font["glyf"]) == other_bytes


def _fvar_only_roboto(path: Path) -> None:
    font = TTFont(ROBOTO)
    axis = Axis()
    axis.axisTag = Tag("wght")
    axis.minValue = 100.0
    axis.defaultValue = 400.0
    axis.maxValue = 900.0
    axis.axisNameID = 256
    fvar = table__f_v_a_r()
    fvar.axes = [axis]
    fvar.instances = []
    font["fvar"] = fvar
    font.save(path)


def test_fvar_only_font_constant_glyph(tmp_path: Path) -> None:
    source = tmp_path / "fvar_only.ttf"
    _fvar_only_roboto(source)
    font = TTFont(source)
    vg = _read(font, "O")
    assert vg.supports == ()
    write_truetype_variable_glyph(font, vg)
    saved = _save(font, tmp_path / "out.ttf")
    assert "gvar" not in saved
    assert "fvar" in saved


def test_fvar_only_font_rejects_varying_glyph(tmp_path: Path) -> None:
    source = tmp_path / "fvar_only.ttf"
    _fvar_only_roboto(source)
    font = TTFont(source)
    varying = _read(TTFont(UBUNTU), "o")
    contours = [Contour(list(c.points), direction=c.direction) for c in varying.default.contours]
    assert varying.supports
    assert contours
    with pytest.raises(ValueError, match="gvar"):
        write_truetype_variable_glyph(
            font,
            VariableGlyph(
                Glyph(metadata=_renamed(varying.default, "O"), contours=contours),
                varying.supports,
                tuple(varying.masters),
                varying.axis_tags,
            ),
        )


def _renamed(glyph: Glyph, name: str) -> GlyphMetadata:
    meta = copy.copy(glyph.metadata)
    meta.name = name
    return meta


def test_cubic_points_rejected() -> None:
    font = TTFont(UBUNTU)
    name = font.getBestCmap()[ord("o")]
    vg = _read(font, name)
    points = [Point(p.x, p.y, PointType.OFF_CURVE_CUBIC) for p in _points(vg.default)]
    coords = [(p.x, p.y) for p in points]
    contour = Contour(points, direction=vg.default.contours[0].direction)
    default = Glyph(metadata=vg.default.metadata, contours=[contour])
    assert len(coords) == len(contour.points)
    with pytest.raises(ValueError, match="cubic"):
        write_truetype_variable_glyph(font, VariableGlyph(default, (), (), vg.axis_tags))
