"""Tests for variable-font static instances."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.cli.output import print_font_info
from stencilizer.core import GlyphAnalyzer
from stencilizer.exceptions import FontFormatError
from stencilizer.io import FontReader
from stencilizer.io.instance import instantiate_static, parse_instance_spec

FIXTURES = Path(__file__).parent.parent / "fixtures" / "variable"
INTER = FIXTURES / "Inter-VF-subset.ttf"
CANTARELL = FIXTURES / "Cantarell-VF-subset.otf"


def test_parse_instance_spec_fills_unnamed_axis_defaults() -> None:
    limits = parse_instance_spec("wght=700", TTFont(INTER))

    assert limits == {"opsz": 14.0, "wght": 700.0}


def test_parse_instance_spec_requires_variable_font() -> None:
    font = TTFont(INTER)
    del font["fvar"]

    with pytest.raises(FontFormatError, match="--instance requires a variable font"):
        parse_instance_spec("wght=700", font)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("wght", "malformed --instance item 'wght'"),
        ("wght=bold", "malformed --instance item 'wght=bold'"),
        ("slnt=4", "unknown axis 'slnt'"),
        ("wght=901", "axis 'wght' value 901.0 outside 100.0..900.0"),
    ],
)
def test_parse_instance_spec_rejects_invalid_items(spec: str, message: str) -> None:
    with pytest.raises(FontFormatError, match=message):
        parse_instance_spec(spec, TTFont(INTER))


def test_instantiate_static_removes_variations_and_overlaps(tmp_path: Path) -> None:
    output = instantiate_static(INTER, "wght=700", tmp_path)
    font = TTFont(output)

    assert "fvar" not in font
    assert _island_count(output, "P") == 1


def test_instantiate_static_downgrades_cff2(tmp_path: Path) -> None:
    output = instantiate_static(CANTARELL, "wght=700", tmp_path)
    font = TTFont(output)

    assert "CFF " in font
    assert "CFF2" not in font


def test_instantiate_static_retries_without_stat_name(tmp_path: Path) -> None:
    output = instantiate_static(INTER, "wght=650", tmp_path)

    assert "fvar" not in TTFont(output)


def test_print_font_info_includes_optional_axes(capsys: pytest.CaptureFixture[str]) -> None:
    axes = "wght 100\u2013900"
    print_font_info("font.ttf", "TrueType", 10, 1000, axes=axes)

    assert f"Variable axes: {axes}" in capsys.readouterr().out


def test_print_font_info_omits_axes_when_not_given(capsys: pytest.CaptureFixture[str]) -> None:
    print_font_info("font.ttf", "TrueType", 10, 1000)

    assert "Variable axes" not in capsys.readouterr().out


def _island_count(font_path: Path, glyph_name: str) -> int:
    with FontReader(font_path) as reader:
        glyph = reader.get_glyph(glyph_name)
        assert glyph is not None
        return len(GlyphAnalyzer().analyze(glyph, reader.units_per_em).get_islands())
