"""Contracts for variable-font processing and the CLI (stage variable-surfaces)."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.removeOverlaps import removeOverlaps  # type: ignore[import-untyped]
from fontTools.varLib.instancer import instantiateVariableFont  # type: ignore[import-untyped]
from typer.testing import CliRunner, Result

from stencilizer.cli.app import app
from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.exceptions import VariationDataError
from stencilizer.io import FontReader
from stencilizer.io.converter import fonttools_glyph_to_domain
from tests.font_helpers import CANTARELL, INTER, UBUNTU
from tests.font_helpers import glyph_at as _glyph_at
from tests.font_helpers import island_count as _islands
from tests.font_helpers import units_per_em as _upm

KEPT_TABLES = ("fvar", "avar", "STAT", "HVAR", "MVAR")


def _cli(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def _processor(tmp_path: Path) -> FontProcessor:
    return FontProcessor(StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "log.txt")))


def _glyph_bytes(font: TTFont, name: str) -> bytes:
    """Compiled glyf data, or CFF2 charstring bytecode, of one glyph."""
    if "glyf" in font:
        return bytes(font["glyf"][name].compile(font["glyf"]))
    charstring = font["CFF2"].cff.topDictIndex[0].CharStrings[name]
    charstring.compile(isCFF2=True)
    return bytes(charstring.bytecode)


def _gvar_tuples(font: TTFont, name: str) -> list[tuple[Any, Any]]:
    return [(v.axes, list(v.coordinates)) for v in font["gvar"].variations.get(name, [])]


def _assert_tables_kept(source: TTFont, output: TTFont, tags: tuple[str, ...]) -> None:
    for tag in tags:
        if tag in source:
            assert tag in output, tag
            assert output.getTableData(tag) == source.getTableData(tag), tag


def test_cli_writes_variable_stencil(tmp_path: Path) -> None:
    out = tmp_path / "out.ttf"
    result = _cli(str(UBUNTU), "-o", str(out), "--log-file", str(tmp_path / "run.log"), "-q")
    assert result.exit_code == 0, result.output
    source, output = TTFont(UBUNTU), TTFont(out)
    assert "fvar" in output
    assert "gvar" in output
    _assert_tables_kept(source, output, ("fvar", "avar", "STAT", "HVAR"))
    assert _glyph_at(output, "o", {}).to_dict() != _glyph_at(source, "o", {}).to_dict()
    for wght in (-1.0, 1.0):
        assert _islands(_glyph_at(output, "o", {"wght": wght}), _upm(output)) == 0, wght


def test_untouched_glyph_keeps_variations(tmp_path: Path) -> None:
    out = tmp_path / "out.ttf"
    result = _cli(str(UBUNTU), "-o", str(out), "--log-file", str(tmp_path / "run.log"), "-q")
    assert result.exit_code == 0, result.output
    source, output = TTFont(UBUNTU), TTFont(out)
    source_coords = list(source["glyf"]["l"].getCoordinates(source["glyf"])[0])
    output_coords = list(output["glyf"]["l"].getCoordinates(output["glyf"])[0])
    assert output_coords == source_coords
    assert _gvar_tuples(source, "l")
    assert _gvar_tuples(output, "l") == _gvar_tuples(source, "l")


def test_cli_instance_pins_static(tmp_path: Path) -> None:
    out = tmp_path / "out.ttf"
    result = _cli(
        str(INTER),
        "--instance",
        "wght=700",
        "-o",
        str(out),
        "--log-file",
        str(tmp_path / "run.log"),
        "-q",
    )
    assert result.exit_code == 0, result.output
    output = TTFont(out)
    assert "fvar" not in output
    expected = instantiateVariableFont(TTFont(INTER), {"wght": 700, "opsz": 14}, static=True)
    assert output["hmtx"].metrics["o"][0] == expected["hmtx"].metrics["o"][0]
    merged = TTFont(out)
    removeOverlaps(merged, ["P"])
    p_glyph = fonttools_glyph_to_domain("P", merged.getGlyphSet()["P"], merged)
    assert _islands(p_glyph, _upm(merged)) == 0

    out2 = tmp_path / "out650.ttf"
    result = _cli(
        str(INTER),
        "--instance",
        "wght=650",
        "-o",
        str(out2),
        "--log-file",
        str(tmp_path / "run.log"),
        "-q",
    )
    assert result.exit_code == 0, result.output
    unnamed = TTFont(out2)
    assert "fvar" not in unnamed
    family = unnamed["name"].getDebugName(1)
    assert family is not None
    assert family.endswith("Stenciled")


def test_cli_shows_variable_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    result = _cli(str(INTER), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Variable axes" in result.output
    assert "wght 100" in result.output
    assert "900" in result.output

    pinned = _cli(str(INTER), "--dry-run", "--instance", "wght=700")
    assert pinned.exit_code == 0, pinned.output
    assert "Variable axes" not in pinned.output


def _modified_glyphs(source: TTFont, output: TTFont) -> set[str]:
    return {
        name
        for name in source.getGlyphOrder()
        if _glyph_bytes(source, name) != _glyph_bytes(output, name)
    }


def _check_valid_everywhere(font_path: Path, tmp_path: Path) -> None:
    from stencilizer.variable.reader import read_variable_glyph
    from stencilizer.variable.validate import validation_locations

    with FontReader(font_path) as reader:
        classification = _processor(tmp_path).classify_glyphs(reader)
    out = tmp_path / font_path.name
    stats = _processor(tmp_path).process(
        font_path=font_path, output_path=out, classification=classification
    )
    assert stats.error_count == 0, (font_path.name, stats.errors)
    assert stats.bridges_added >= 1, font_path.name
    source, output = TTFont(font_path), TTFont(out)
    modified = _modified_glyphs(source, output)
    assert modified, font_path.name
    assert modified <= {glyph.name for glyph in classification.glyphs_to_process}
    upm = _upm(output)
    for name in sorted(modified):
        vg = read_variable_glyph(output, name)
        assert vg is not None, name
        allowed = _islands(vg.instance({}), upm)
        for location in validation_locations(vg):
            assert _islands(vg.instance(location), upm) <= allowed, (name, location)
    _assert_tables_kept(source, output, KEPT_TABLES)


def test_variable_output_valid_everywhere(tmp_path: Path) -> None:
    for font_path in (UBUNTU, INTER, CANTARELL):
        workdir = tmp_path / font_path.stem
        workdir.mkdir()
        _check_valid_everywhere(font_path, workdir)


def test_unsupported_glyph_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stencilizer.variable.processing as processing

    original = processing.read_variable_glyph

    def failing_read(font: TTFont, name: str, *args: Any) -> Any:
        if name == "o":
            raise VariationDataError(name, "contract: unreadable variation data")
        return original(font, name, *args)

    monkeypatch.setattr(processing, "read_variable_glyph", failing_read)
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        classification: Any = processor.classify_glyphs(reader)
    assert classification.skipped_reasons["o"] == "unsupported variation data"
    assert classification.unsupported_islands["o"] == 1
    out = tmp_path / "out.ttf"
    stats = processor.process(font_path=UBUNTU, output_path=out, classification=classification)
    assert stats.bridges_added >= 1
    assert stats.unbridged_count >= 1
    source, output = TTFont(UBUNTU), TTFont(out)
    assert _glyph_bytes(output, "o") == _glyph_bytes(source, "o")
    assert _gvar_tuples(output, "o") == _gvar_tuples(source, "o")


def test_cli_instance_rejects_bad_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    cases = (("wdth=100", "wdth"), ("wght=5000", "wght"), ("wght", "wght"))
    for index, (spec, axis) in enumerate(cases):
        out = tmp_path / f"bad{index}.ttf"
        result = _cli(str(INTER), "--instance", spec, "-o", str(out), "-q")
        assert result.exit_code == 1, (spec, result.output)
        assert axis in result.output, (spec, result.output)
        assert not out.exists(), spec
