"""Tests for variable-font static instances."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.cli.output import print_font_info
from stencilizer.exceptions import FontFormatError
from stencilizer.io import FontReader
from stencilizer.io.instance import InstanceSpecError, instantiate_static, parse_instance_spec
from stencilizer.io.writer import update_font_names
from tests.font_helpers import CANTARELL, INTER, island_count


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
        ("wght", "malformed item 'wght'"),
        ("wght=bold", "malformed item 'wght=bold'"),
        ("slnt=4", "unknown axis 'slnt'"),
        ("wght=901", "axis 'wght' value 901.0 outside 100.0..900.0"),
    ],
)
def test_parse_instance_spec_rejects_invalid_items(spec: str, message: str) -> None:
    with pytest.raises(FontFormatError, match=message):
        parse_instance_spec(spec, TTFont(INTER))


def test_parse_instance_spec_rejects_repeated_axis() -> None:
    with pytest.raises(InstanceSpecError, match="axis 'wght' given more than once"):
        parse_instance_spec("wght=700,wght=800", TTFont(INTER))


@pytest.mark.parametrize("spec", ["wght", "wght=bold", "slnt=4", "wght=901", "wght=1,wght=2"])
def test_parse_instance_spec_errors_are_usage_errors(spec: str) -> None:
    with pytest.raises(InstanceSpecError) as caught:
        parse_instance_spec(spec, TTFont(INTER))

    assert str(caught.value).startswith("Invalid --instance: ")
    assert "Invalid font format" not in str(caught.value)


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


def test_instantiate_static_unnamed_location_keeps_style_in_full_name(tmp_path: Path) -> None:
    names = TTFont(instantiate_static(INTER, "wght=650", tmp_path))["name"]

    assert names.getDebugName(1) == "Inter Variable"
    assert names.getDebugName(4) == "Inter Variable Regular"


def test_instantiate_static_named_location_full_name_is_unchanged(tmp_path: Path) -> None:
    names = TTFont(instantiate_static(INTER, "wght=700", tmp_path))["name"]

    assert names.getDebugName(4) == "Inter Variable Text Bold"


@pytest.mark.parametrize(
    ("spec", "family", "full_name"),
    [
        ("wght=650", "Inter Variable Stenciled", "Inter Variable Stenciled Regular"),
        ("wght=400", "Inter Variable Text Stenciled", "Inter Variable Text Stenciled Regular"),
        ("wght=700", "Inter Variable Text Stenciled", "Inter Variable Text Stenciled Bold"),
    ],
)
def test_instance_stencil_suffix_goes_after_the_family_in_both_names(
    tmp_path: Path, spec: str, family: str, full_name: str
) -> None:
    font = TTFont(instantiate_static(INTER, spec, tmp_path))

    update_font_names(font)

    assert font["name"].getDebugName(1) == family
    assert font["name"].getDebugName(4) == full_name


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
        return island_count(glyph, reader.units_per_em)
