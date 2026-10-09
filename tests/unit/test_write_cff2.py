"""Tests for the CFF2 blend writer."""

import copy
from pathlib import Path

import pytest
from fontTools.cffLib.specializer import (  # type: ignore[import-untyped]
    commandsToProgram,
    specializeCommands,
)
from fontTools.misc.psCharStrings import T2CharString  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.domain.contour import Point, PointType
from stencilizer.domain.glyph import Glyph
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.reader import read_variable_glyph
from stencilizer.variable.transform import transform_variable_glyph
from stencilizer.variable.write_cff2 import write_cff2_variable_glyph

CANTARELL = Path(__file__).parent.parent / "fixtures" / "variable" / "Cantarell-VF-subset.otf"
LOCATIONS: list[dict[str, float]] = [{"wght": -1.0}, {}, {"wght": 1.0}]


def _name(font: TTFont, char: str) -> str:
    return str(font.getBestCmap()[ord(char)])


def _read(font: TTFont, char: str) -> VariableGlyph:
    vg = read_variable_glyph(font, _name(font, char))
    assert vg is not None
    return vg


def _save(font: TTFont, path: Path) -> TTFont:
    font.save(path)
    return TTFont(path)


def _program(font: TTFont, name: str) -> list[object]:
    charstring = font["CFF2"].cff.topDictIndex[0].CharStrings[name]
    charstring.decompile()
    return list(charstring.program)


def _reread(font: TTFont, name: str, location: dict[str, float]) -> Glyph:
    glyph_set = font.getGlyphSet(location=location, normalized=True)
    return fonttools_glyph_to_domain(name, glyph_set[name], font)


def _points(glyph: Glyph) -> list[Point]:
    return [point for contour in glyph.contours for point in contour.points]


def _assert_close(saved: Glyph, expected: Glyph, tolerance: float) -> None:
    assert [len(c.points) for c in saved.contours] == [len(c.points) for c in expected.contours]
    for a, b in zip(_points(saved), _points(expected), strict=True):
        assert abs(a.x - b.x) <= tolerance
        assert abs(a.y - b.y) <= tolerance


def _vsindex_font() -> TTFont:
    """Cantarell with a second VarData listing VarData 0's regions in reverse order."""
    font = TTFont(CANTARELL)
    store = font["CFF2"].cff.topDictIndex[0].VarStore.otVarStore
    extra = copy.deepcopy(store.VarData[0])
    extra.VarRegionIndex = list(reversed(extra.VarRegionIndex))
    store.VarData.append(extra)
    store.VarDataCount = len(store.VarData)
    return font


def test_untransformed_glyph_round_trips(tmp_path: Path) -> None:
    font = TTFont(CANTARELL)
    vg = _read(font, "o")
    assert vg.supports
    write_cff2_variable_glyph(font, vg)
    saved = _save(font, tmp_path / "out.otf")
    for location in LOCATIONS:
        _assert_close(_reread(saved, vg.name, location), vg.instance(location), 0.5)


def test_flattened_glyph_round_trips(tmp_path: Path) -> None:
    font = TTFont(CANTARELL)
    flat = flatten_compatible(_read(font, "o"), int(font["head"].unitsPerEm))
    points = _points(flat.default)
    assert len(points) > 100
    assert all(p.point_type == PointType.ON_CURVE for p in points)
    write_cff2_variable_glyph(font, flat)
    saved = _save(font, tmp_path / "out.otf")
    for location in LOCATIONS:
        _assert_close(_reread(saved, flat.name, location), flat.instance(location), 1.0)


def test_blend_operand_encoding() -> None:
    commands = [("rlineto", [[100, 10, 20, 1], 0])]
    program = commandsToProgram(
        specializeCommands(commands, generalizeFirst=False, preserveTopology=True)
    )
    assert program[:5] == [100, 10, 20, 1, "blend"]
    assert program[-1] in {"rlineto", "hlineto"}
    font = TTFont(CANTARELL)
    top_dict = font["CFF2"].cff.topDictIndex[0]
    charstring = T2CharString(program=program, private=top_dict.FDArray[0].Private, globalSubrs=[])
    pen = RecordingPen()
    charstring.draw(pen, lambda _vs_index, deltas: 0.5 * deltas[0])
    assert ("lineTo", ((105, 0),)) in pen.value


def test_nonzero_vsindex_is_written(tmp_path: Path) -> None:
    font = _vsindex_font()
    name = _name(font, "o")
    charstring = font["CFF2"].cff.topDictIndex[0].CharStrings[name]
    charstring.decompile()
    charstring.program[:0] = [1, "vsindex"]
    vg = _read(font, "o")
    write_cff2_variable_glyph(font, vg)
    assert _program(font, name)[:2] == [1, "vsindex"]
    saved = _save(font, tmp_path / "out.otf")
    for location in [{}, *(s.peak() for s in vg.supports)]:
        _assert_close(_reread(saved, name, location), vg.instance(location), 0.5)


def test_constant_glyph_has_no_blend(tmp_path: Path) -> None:
    font = TTFont(CANTARELL)
    vg = _read(font, "o")
    constant = VariableGlyph(vg.default, (), (), vg.axis_tags, cff2=True)
    write_cff2_variable_glyph(font, constant)
    program = _program(font, vg.name)
    assert "blend" not in program
    assert "vsindex" not in program
    saved = _save(font, tmp_path / "out.otf")
    for location in LOCATIONS:
        _assert_close(_reread(saved, vg.name, location), vg.default, 0.5)


def test_support_count_mismatch_raises() -> None:
    font = TTFont(CANTARELL)
    vg = _read(font, "o")
    partial = VariableGlyph(vg.default, vg.supports[:1], vg.masters[:1], vg.axis_tags, True)
    with pytest.raises(ValueError, match="regions"):
        write_cff2_variable_glyph(font, partial)


def test_transformed_glyph_saves_and_every_glyph_draws(tmp_path: Path) -> None:
    font = TTFont(CANTARELL)
    upm = int(font["head"].unitsPerEm)
    outcome = transform_variable_glyph(_read(font, "o"), BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count >= 1
    write_cff2_variable_glyph(font, outcome.glyph)
    saved = _save(font, tmp_path / "out.otf")
    for location in LOCATIONS:
        expected = outcome.glyph.instance(location)
        _assert_close(_reread(saved, outcome.glyph.name, location), expected, 1.0)
        glyph_set = saved.getGlyphSet(location=location, normalized=True)
        for name in saved.getGlyphOrder():
            glyph_set[name].draw(RecordingPen())
