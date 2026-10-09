"""CLI options for variable-font bridge width scaling and the stencil-first --instance flow."""

import shutil
from pathlib import Path

import pytest
import typer
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.removeOverlaps import removeOverlaps  # type: ignore[import-untyped]
from typer.testing import CliRunner, Result

from stencilizer.cli.app import app
from stencilizer.cli.handlers import CFF2_WARNING, STATIC_WARNING, stencil_pinned
from stencilizer.config import StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.utils import ProcessingStats
from tests.font_helpers import CANTARELL, INTER, ROBOTO, island_count, units_per_em


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")


def _cli(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def _flat(text: str) -> str:
    """Drop whitespace so assertions survive the console folding long lines."""
    return "".join(text.split())


def _merged_island_count(path: Path, name: str) -> int:
    font = TTFont(path)
    removeOverlaps(font, [name])
    glyph = fonttools_glyph_to_domain(name, font.getGlyphSet()[name], font)
    return island_count(glyph, units_per_em(font))


def test_help_lists_width_scaling_options() -> None:
    result = _cli("--help")
    assert result.exit_code == 0
    for option in ("--width-scaling", "--scaling-strength", "--min-bridge-width"):
        assert option in result.output


@pytest.mark.parametrize(
    "args", [("--scaling-strength", "150"), ("--min-bridge-width", "5")], ids=["strength", "min"]
)
def test_out_of_range_values_exit_with_usage_error(args: tuple[str, str]) -> None:
    assert _cli(str(INTER), "--dry-run", *args).exit_code == 2


def test_dry_run_reports_proportional_settings() -> None:
    result = _cli(
        str(INTER),
        "--dry-run",
        "--width-scaling",
        "proportional",
        "--scaling-strength",
        "40",
        "--min-bridge-width",
        "45",
    )
    assert result.exit_code == 0, result.output
    assert _flat("strength 40.0%, minimum 45.0%") in _flat(result.output)


def test_dry_run_default_reports_fixed() -> None:
    result = _cli(str(INTER), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Width scaling         fixed" in result.output


def test_static_font_warns_and_reports_fixed() -> None:
    result = _cli(str(ROBOTO), "--dry-run", "--width-scaling", "proportional")
    assert result.exit_code == 0, result.output
    assert _flat(STATIC_WARNING) in _flat(result.output)
    assert "Width scaling         fixed" in result.output


def test_instance_proportional_stencils_then_pins(tmp_path: Path) -> None:
    out = tmp_path / "pinned.ttf"
    result = _cli(
        str(INTER),
        "-o",
        str(out),
        "--log-file",
        str(tmp_path / "run.log"),
        "--instance",
        "wght=900",
        "--width-scaling",
        "proportional",
    )
    assert result.exit_code == 0, result.output
    assert _flat(str(out)) in _flat(result.output)
    font = TTFont(out)
    assert "fvar" not in font
    assert font["name"].getDebugName(1) == "Inter Variable Text Black Stenciled"
    for name in ("o", "ampersand"):
        assert _merged_island_count(out, name) == 0, name


def test_bad_instance_spec_fails_before_processing(tmp_path: Path) -> None:
    fixed = _cli(str(INTER), "-o", str(tmp_path / "a.ttf"), "--instance", "nope=1")
    proportional = _cli(
        str(INTER),
        "-o",
        str(tmp_path / "b.ttf"),
        "--instance",
        "nope=1",
        "--width-scaling",
        "proportional",
    )
    assert proportional.exit_code == fixed.exit_code == 1
    assert "Processing" not in proportional.output
    assert not (tmp_path / "b.ttf").exists()


def test_cff2_instance_warns_and_pins_first(tmp_path: Path) -> None:
    out = tmp_path / "cff2.otf"
    result = _cli(
        str(CANTARELL),
        "-o",
        str(out),
        "--log-file",
        str(tmp_path / "run.log"),
        "--instance",
        "wght=250",
        "--width-scaling",
        "proportional",
    )
    assert result.exit_code == 0, result.output
    assert _flat(CFF2_WARNING) in _flat(result.output)
    assert "fvar" not in TTFont(out)


def test_stencil_pinned_publishes_when_nothing_to_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        FontProcessor, "classify_glyphs", lambda _self, _reader: GlyphClassification()
    )
    out = tmp_path / "published.ttf"
    stats = stencil_pinned(
        INTER, INTER, "wght=900", tmp_path, out, StencilizerSettings(), workers=None
    )
    assert stats == ProcessingStats()
    assert stats.bridges_added == 0
    font = TTFont(out)
    assert "fvar" not in font
    assert " Stenciled" in font["name"].getDebugName(1)


def test_stencil_pinned_maps_interrupt_to_exit_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*_args: object) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr("stencilizer.cli.handlers.pin_stenciled", interrupt)
    with pytest.raises(typer.Exit) as raised:
        stencil_pinned(
            INTER,
            INTER,
            "wght=900",
            tmp_path,
            tmp_path / "out.ttf",
            StencilizerSettings(),
            workers=None,
        )
    assert raised.value.exit_code == 130


def test_stencil_first_report_sums_bridges_and_takes_second_unbridged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "pinned.ttf"

    def first_pass(*_args: object) -> ProcessingStats:
        return ProcessingStats(processed_count=4, bridges_added=5, unbridged_count=9)

    def second_pass(*_args: object) -> ProcessingStats:
        shutil.copy(INTER, out)
        return ProcessingStats(bridges_added=7, unbridged_count=3)

    monkeypatch.setattr("stencilizer.cli.app._stencil", first_pass)
    monkeypatch.setattr("stencilizer.cli.app.stencil_pinned", second_pass)
    result = _cli(
        str(INTER),
        "-o",
        str(out),
        "--log-file",
        str(tmp_path / "run.log"),
        "--instance",
        "wght=900",
        "--width-scaling",
        "proportional",
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "12bridges" in flat
    assert "3islandsremainedunbridged" in flat
    assert "9islands" not in flat
