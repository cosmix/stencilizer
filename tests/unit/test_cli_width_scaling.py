"""CLI options for variable-font bridge width scaling and the stencil-first --instance flow."""

import re
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import typer
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.removeOverlaps import removeOverlaps  # type: ignore[import-untyped]
from typer.testing import CliRunner, Result

from stencilizer.cli.app import app
from stencilizer.cli.handlers import CFF2_WARNING, STATIC_WARNING, stencil_pinned
from stencilizer.cli.pinning import pin_stenciled
from stencilizer.config import StencilizerSettings
from stencilizer.core import FontProcessor, processor
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

    def second_pass(*_args: object, **_kwargs: object) -> ProcessingStats:
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


PROPORTIONAL_PINNED = ("--instance", "wght=900", "--width-scaling", "proportional")


def _spy_workdirs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the working directory of every pin so a test can check it was removed."""
    workdirs: list[Path] = []
    real_pin = pin_stenciled

    def spy(stenciled: Path, source: Path, instance: str, workdir: Path) -> Path:
        workdirs.append(workdir)
        return real_pin(stenciled, source, instance, workdir)

    monkeypatch.setattr("stencilizer.cli.handlers.pin_stenciled", spy)
    return workdirs


def test_stencil_first_run_writes_a_single_log_file(
    tmp_path: Path, default_log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _cli(str(INTER), *PROPORTIONAL_PINNED, "-o", str(tmp_path / "out.ttf"))
    assert result.exit_code == 0, result.output
    (log_file,) = default_log_dir.glob("stencilizer_*.log")
    # Both passes configure logging; the second appends to the file the first opened.
    assert log_file.read_text(encoding="utf-8").count("Logging initialized") == 2


def test_default_log_path_names_a_timestamped_file_in_the_working_directory(
    real_default_log_path: Callable[[], Path],
) -> None:
    path = real_default_log_path()
    assert not path.is_absolute()
    assert re.fullmatch(r"stencilizer_\d{8}_\d{6}\.log", path.name)


def test_failing_glyph_in_the_second_pass_exits_1_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_transform = processor._transform_glyph

    def fail_on_o(glyph_dict: dict[str, Any], *rest: Any) -> Any:
        if glyph_dict["metadata"]["name"] == "o":
            raise ValueError("broken glyph")
        return real_transform(glyph_dict, *rest)

    def copy_unstenciled(_font: Path, _shown: Path, stenciled: Path, *_rest: object) -> object:
        # Pass 1 would bridge every island; the unbridged copy leaves the second pass real work.
        shutil.copy(INTER, stenciled)
        return ProcessingStats()

    monkeypatch.setattr("stencilizer.cli.app._stencil", copy_unstenciled)
    # Threads keep the patched transform in this process; spawned workers would not see it.
    monkeypatch.setattr(
        "stencilizer.core.processor.ProcessPoolExecutor",
        lambda max_workers=None, **_options: ThreadPoolExecutor(max_workers=max_workers),
    )
    monkeypatch.setattr(processor, "_transform_glyph", fail_on_o)
    workdirs = _spy_workdirs(monkeypatch)
    out = tmp_path / "out.ttf"
    result = _cli(str(INTER), *PROPORTIONAL_PINNED, "-o", str(out))
    assert result.exit_code == 1
    assert _flat("o: broken glyph") in _flat(result.output)
    assert not out.exists()
    assert len(workdirs) == 1
    assert not workdirs[0].exists()


@pytest.mark.parametrize("quiet", [False, True])
def test_stencil_pinned_interrupt_prints_notice_unless_quiet(
    quiet: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "stencilizer.cli.handlers.pin_stenciled", Mock(side_effect=KeyboardInterrupt)
    )
    with pytest.raises(typer.Exit) as raised:
        stencil_pinned(
            INTER,
            INTER,
            "wght=900",
            tmp_path,
            tmp_path / "out.ttf",
            StencilizerSettings(),
            workers=None,
            quiet=quiet,
        )
    assert raised.value.exit_code == 130
    assert ("Cancelling" in capsys.readouterr().out) is not quiet


@pytest.mark.parametrize("option", ["--bridge-width", "--scaling-strength", "--min-bridge-width"])
def test_nan_option_values_exit_with_usage_error(option: str) -> None:
    result = _cli(str(INTER), "--dry-run", option, "nan")
    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output


def test_stencil_first_report_covers_both_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([100.0, 107.5])
    reported: list[ProcessingStats] = []
    monkeypatch.setattr("stencilizer.cli.app.time", SimpleNamespace(time=lambda: next(ticks)))
    monkeypatch.setattr(
        "stencilizer.cli.app._stencil", lambda *_args: ProcessingStats(start_time=1.0, end_time=2.0)
    )
    monkeypatch.setattr("stencilizer.cli.app.stencil_pinned", lambda *_a, **_k: ProcessingStats())
    monkeypatch.setattr(
        "stencilizer.cli.app.finish_run", lambda _out, stats, _quiet: reported.append(stats)
    )
    result = _cli(str(INTER), *PROPORTIONAL_PINNED, "-o", str(tmp_path / "out.ttf"))
    assert result.exit_code == 0, result.output
    assert [stats.duration_seconds for stats in reported] == [7.5]


def test_interrupt_in_the_first_pass_exits_130_and_removes_the_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdirs: list[Path] = []

    def interrupt(_self: FontProcessor, **kwargs: Any) -> ProcessingStats:
        workdirs.append(kwargs["output_path"].parent)
        raise KeyboardInterrupt

    monkeypatch.setattr(FontProcessor, "process", interrupt)
    out = tmp_path / "out.ttf"
    result = _cli(str(INTER), *PROPORTIONAL_PINNED, "-o", str(out))
    assert result.exit_code == 130
    assert "Cancelling" in result.output
    assert len(workdirs) == 1
    assert not workdirs[0].exists()
    assert not out.exists()


def test_stencil_first_output_directory_fails_and_stays_untouched(tmp_path: Path) -> None:
    taken = tmp_path / "taken"
    taken.mkdir()
    result = _cli(str(INTER), *PROPORTIONAL_PINNED, "-o", str(taken))
    assert result.exit_code == 1
    assert _flat("Could not save font") in _flat(result.output)
    assert taken.is_dir()
    assert not list(taken.iterdir())


@pytest.mark.parametrize(
    ("font", "args"),
    [
        (ROBOTO, ("--width-scaling", "proportional")),
        (CANTARELL, ("--instance", "wght=250", "--width-scaling", "proportional")),
    ],
    ids=["static", "cff2"],
)
def test_quiet_suppresses_width_scaling_warnings(font: Path, args: tuple[str, ...]) -> None:
    result = _cli(str(font), "--dry-run", "--quiet", *args)
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert _flat(STATIC_WARNING) not in flat
    assert _flat(CFF2_WARNING) not in flat


@pytest.mark.parametrize("mode", ["--list-islands", "--dry-run"])
def test_analysis_modes_pin_first_and_skip_the_stencil_first_flow(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stencil_first = Mock()
    monkeypatch.setattr("stencilizer.cli.app._run_stencil_first", stencil_first)
    result = _cli(str(INTER), mode, *PROPORTIONAL_PINNED)
    assert result.exit_code == 0, result.output
    stencil_first.assert_not_called()
    # The pinned instance is static, so the font info shows no variable axes.
    assert "Variable axes" not in result.output
