"""Contracts for the gvar and CFF2 blend writers (stage variable-writers)."""

import copy
from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import FontFormatError
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.io.writer import FontWriter
from tests.font_helpers import CANTARELL, INTER, UBUNTU, fvar_only_roboto
from tests.font_helpers import glyph_at as _reread
from tests.font_helpers import island_count as _islands
from tests.font_helpers import points as _points
from tests.font_helpers import units_per_em as _upm


def _transform(font: TTFont, name: str) -> Any:
    from stencilizer.variable.reader import read_variable_glyph
    from stencilizer.variable.transform import transform_variable_glyph

    vg = read_variable_glyph(font, name)
    assert vg is not None
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), _upm(font))
    assert outcome.bridge_count >= 1
    return outcome


def _write(font: TTFont, vg: Any, output: Path) -> TTFont:
    writer = FontWriter(font, output)
    writer.update_variable_glyph(vg)
    writer.save()
    return TTFont(output)


def _assert_close(saved: Glyph, expected: Glyph, tolerance: float) -> None:
    assert [len(c.points) for c in saved.contours] == [len(c.points) for c in expected.contours]
    for a, b in zip(_points(saved), _points(expected), strict=True):
        assert abs(a.x - b.x) <= tolerance
        assert abs(a.y - b.y) <= tolerance


def _phantoms(font: TTFont, name: str) -> dict[frozenset[Any], list[tuple[float, float]]]:
    """Phantom-point deltas of every gvar tuple of ``name``, keyed by its axes."""
    coords, (_, end_pts, _, _) = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)
    phantoms: dict[frozenset[Any], list[tuple[float, float]]] = {}
    for variation in font["gvar"].variations[name]:
        tv = copy.deepcopy(variation)
        tv.calcInferredDeltas(coords, end_pts)
        tail = [
            (0.0, 0.0) if d is None else (float(d[0]), float(d[1])) for d in tv.coordinates[-4:]
        ]
        phantoms[frozenset(tv.axes.items())] = tail
    return phantoms


def test_gvar_output_reproduces_instances(tmp_path: Path) -> None:
    from stencilizer.variable.validate import validation_locations

    font = TTFont(UBUNTU)
    upm = _upm(font)
    result = _transform(font, "o")
    saved = _write(font, result.glyph, tmp_path / "out.ttf")
    locations = validation_locations(result.glyph)
    assert locations
    for location in locations:
        reread = _reread(saved, "o", location)
        _assert_close(reread, result.glyph.instance(location), 0.01)
        assert _islands(reread, upm) == 0


def test_gvar_keeps_phantom_deltas(tmp_path: Path) -> None:
    source = TTFont(UBUNTU)
    expected = _phantoms(source, "o")
    advance = source["hmtx"].metrics["o"][0]
    assert any(delta != (0.0, 0.0) for tail in expected.values() for delta in tail)
    result = _transform(source, "o")
    saved = _write(source, result.glyph, tmp_path / "out.ttf")
    assert _phantoms(saved, "o") == expected
    assert saved["hmtx"].metrics["o"][0] == advance


def test_gvar_writer_handles_missing_gvar(tmp_path: Path) -> None:
    from stencilizer.variable.reader import read_variable_glyph

    source_path = tmp_path / "modified.ttf"
    fvar_only_roboto().save(source_path)
    font = TTFont(source_path)
    vg = read_variable_glyph(font, "O")
    assert vg is not None
    assert vg.supports == ()
    result = _transform(font, "O")
    saved = _write(font, result.glyph, tmp_path / "out.ttf")
    assert "gvar" not in saved
    assert "fvar" in saved
    glyph = fonttools_glyph_to_domain("O", saved.getGlyphSet()["O"], saved)
    assert _islands(glyph, _upm(saved)) == 0


def test_cff2_blend_output_reproduces_instances(tmp_path: Path) -> None:
    font = TTFont(CANTARELL)
    result = _transform(font, "o")
    saved = _write(font, result.glyph, tmp_path / "out.otf")
    locations: list[dict[str, float]] = [{"wght": -1.0}, {}, {"wght": 1.0}]
    for location in locations:
        _assert_close(_reread(saved, "o", location), result.glyph.instance(location), 1.0)


def test_variable_names_suffixed(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"
    FontWriter(TTFont(INTER), output).save()
    names = TTFont(output)["name"]
    assert names.getDebugName(25) == "InterVariableStenciled"
    assert names.getDebugName(280) == "InterVariableStenciled-Thin"
    assert names.getDebugName(4) == "Inter Variable Stenciled"


def test_static_writer_refuses_variable(tmp_path: Path) -> None:
    font = TTFont(UBUNTU)
    glyph = fonttools_glyph_to_domain("o", font.getGlyphSet()["o"], font)
    output = tmp_path / "out.ttf"
    with pytest.raises(FontFormatError):
        FontWriter(font, output).update_glyph(glyph)
    assert not output.exists()
    # The rejection moved from save to update_glyph: an untouched variable font saves.
    saved = tmp_path / "saved.ttf"
    FontWriter(TTFont(UBUNTU), saved).save()
    assert saved.exists()
