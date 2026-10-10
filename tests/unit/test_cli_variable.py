"""CLI output and --instance handling for variable fonts and untrusted text."""

import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from typer.testing import CliRunner, Result

from stencilizer.cli.app import _classify_font, app
from stencilizer.cli.output import print_error, print_glyph_islands, print_islands_found
from stencilizer.exceptions import FontLoadError
from tests.font_helpers import CANTARELL, INTER

MARKUP_TAG = "[/x]"
EN_DASH = "\N{EN DASH}"


def _cli(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def _flat(text: str) -> str:
    """Drop whitespace so assertions survive the console folding long paths."""
    return "".join(text.split())


def _assert_clean_exit_1(result: Result) -> None:
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output


@pytest.fixture
def markup_axis_font(tmp_path: Path) -> Path:
    """Inter-VF-subset with its wght axis tag renamed to a string that looks like Rich markup."""
    font = TTFont(INTER)
    for axis in font["fvar"].axes:
        if axis.axisTag == "wght":
            axis.axisTag = MARKUP_TAG
    for instance in font["fvar"].instances:
        instance.coordinates = {
            MARKUP_TAG if tag == "wght" else tag: value
            for tag, value in instance.coordinates.items()
        }
    for axis in font["STAT"].table.DesignAxisRecord.Axis:
        if axis.AxisTag == "wght":
            axis.AxisTag = MARKUP_TAG
    path = tmp_path / "markup-axis.ttf"
    font.save(path)
    return path


def test_dry_run_prints_markup_like_axis_tag(markup_axis_font: Path) -> None:
    result = _cli(str(markup_axis_font), "--dry-run")

    assert result.exit_code == 0, result.output
    assert f"Variable axes: opsz 14{EN_DASH}32, {MARKUP_TAG} 100{EN_DASH}900" in result.output


def test_list_islands_prints_markup_like_axis_tag(markup_axis_font: Path) -> None:
    result = _cli(str(markup_axis_font), "--list-islands")

    assert result.exit_code == 0, result.output
    assert f"{MARKUP_TAG} 100" in result.output
    assert "P: 1 island" in result.output


def test_error_text_is_not_parsed_as_markup(capsys: pytest.CaptureFixture[str]) -> None:
    print_error(f"Could not analyze font: {MARKUP_TAG}", details=f"see {MARKUP_TAG} and [bold]")

    out = capsys.readouterr().out
    assert f"Error: Could not analyze font: {MARKUP_TAG}" in out
    assert f"see {MARKUP_TAG} and [bold]" in out


def test_glyph_names_print_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    print_islands_found(2, [MARKUP_TAG, "[bold]"], verbose=True)
    print_glyph_islands(MARKUP_TAG, 2)

    out = capsys.readouterr().out
    assert f"{MARKUP_TAG}, [bold]" in out
    assert f"{MARKUP_TAG}: 2 islands" in out


def test_missing_input_with_markup_in_name_exits_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / f"{MARKUP_TAG}.ttf"

    result = _cli(str(missing))

    _assert_clean_exit_1(result)
    assert _flat(f"Input file not found: {missing}") in _flat(result.output)


def test_instance_value_that_looks_like_markup_exits_cleanly() -> None:
    result = _cli(str(INTER), "--instance", f"wght={MARKUP_TAG}")

    _assert_clean_exit_1(result)
    assert f"Invalid --instance: malformed item 'wght={MARKUP_TAG}'" in result.output


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("wght=abc", "malformed item 'wght=abc'"),
        ("wght", "malformed item 'wght'"),
        ("slnt=4", "unknown axis 'slnt'"),
        ("wght=901", "axis 'wght' value 901.0 outside 100.0..900.0"),
    ],
)
def test_bad_instance_spec_reads_as_instance_usage_error(
    tmp_path: Path, spec: str, message: str
) -> None:
    out = tmp_path / "out.ttf"

    result = _cli(str(INTER), "--instance", spec, "-o", str(out))

    _assert_clean_exit_1(result)
    assert f"Invalid --instance: {message}" in result.output
    assert "Invalid font format" not in result.output
    assert not out.exists()


def test_repeated_instance_axis_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "out.ttf"

    result = _cli(str(INTER), "--instance", "wght=700,wght=800", "-o", str(out))

    _assert_clean_exit_1(result)
    assert "Invalid --instance: axis 'wght' given more than once" in result.output
    assert not out.exists()


def test_static_font_with_instance_is_a_usage_error(tmp_path: Path) -> None:
    static = tmp_path / "static.ttf"
    font = TTFont(INTER)
    del font["fvar"]
    font.save(static)

    result = _cli(str(static), "--instance", "wght=700")

    _assert_clean_exit_1(result)
    assert "--instance requires a variable font" in result.output
    assert "Invalid font format" not in result.output


@pytest.mark.parametrize("mode", ["--dry-run", "--list-islands", None])
def test_instance_run_shows_the_input_path(tmp_path: Path, mode: str | None) -> None:
    args = [
        str(INTER),
        "--instance",
        "wght=700",
        "-o",
        str(tmp_path / "out.ttf"),
        "--log-file",
        str(tmp_path / "run.log"),
    ]
    if mode is not None:
        args.append(mode)

    result = _cli(*args)

    assert result.exit_code == 0, result.output
    assert _flat(str(INTER)) in _flat(result.output)
    assert "stencilizer-instance" not in _flat(result.output)
    assert "Variable axes" not in result.output


def test_load_error_names_the_display_path(tmp_path: Path) -> None:
    temporary = tmp_path / "stencilizer-instance-x" / "font-instance.ttf"
    given = tmp_path / "font.ttf"

    with (
        patch("stencilizer.cli.app.FontReader", side_effect=RuntimeError("unreadable")),
        pytest.raises(FontLoadError) as caught,
    ):
        _classify_font(temporary, Mock(), quiet=True, display_path=given)

    assert caught.value.path == str(given)
    assert caught.value.reason == "unreadable"


def test_cff2_variable_font_instance_is_written_as_a_bridged_cff_font(tmp_path: Path) -> None:
    out = tmp_path / "cantarell-400.otf"

    result = _cli(
        str(CANTARELL),
        "--instance",
        "wght=400",
        "-o",
        str(out),
        "--log-file",
        str(tmp_path / "run.log"),
    )

    assert result.exit_code == 0, result.output
    assert "Could not save font" not in result.output
    bridges = re.search(r"(\d+) bridges", result.output)
    assert bridges is not None, result.output
    assert int(bridges.group(1)) >= 1
    with TTFont(out) as written:
        assert "CFF " in written
        assert "CFF2" not in written
        assert "fvar" not in written
