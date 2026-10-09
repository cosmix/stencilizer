"""Contracts for the FontReader and CLI refactor."""

from collections import Counter
from pathlib import Path
from typing import NoReturn, Protocol, cast

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from typer.testing import CliRunner

from stencilizer.cli.app import app
from stencilizer.core.analyzer import ContourHierarchy, GlyphAnalyzer
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import GlyphProcessingError
from stencilizer.io.reader import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
ROBOTO_PATH = FIXTURES_DIR / "Roboto-Regular.ttf"
COMMIT_MONO_PATH = FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf"


class _ReaderWithFont(Protocol):
    @property
    def font(self) -> TTFont: ...


def test_font_reader_font_returns_loaded_ttfont() -> None:
    with FontReader(ROBOTO_PATH) as reader:
        font = cast("_ReaderWithFont", reader).font

        assert isinstance(font, TTFont)
        assert font["head"].unitsPerEm == reader.units_per_em


def test_font_reader_font_before_load_raises_runtime_error() -> None:
    reader = FontReader(ROBOTO_PATH)

    with pytest.raises(RuntimeError, match=r".*"):
        _ = cast("_ReaderWithFont", reader).font


def test_font_reader_font_after_close_raises_runtime_error() -> None:
    reader = FontReader(ROBOTO_PATH)
    reader.load()
    reader.close()

    with pytest.raises(RuntimeError, match=r".*"):
        _ = cast("_ReaderWithFont", reader).font


def test_font_reader_font_is_read_only() -> None:
    property_name = "font"

    with FontReader(ROBOTO_PATH) as reader, pytest.raises(AttributeError, match=r".*"):
        setattr(reader, property_name, object())


def test_font_reader_missing_glyph_returns_none() -> None:
    with FontReader(ROBOTO_PATH) as reader:
        assert reader.get_glyph("__missing_glyph__") is None


def test_font_reader_conversion_failure_preserves_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    original_error = ValueError("boom")

    def fail_conversion(**_: object) -> NoReturn:
        raise original_error

    monkeypatch.setattr("stencilizer.io.reader.fonttools_glyph_to_domain", fail_conversion)

    with FontReader(ROBOTO_PATH) as reader, pytest.raises(GlyphProcessingError, match="A") as error:
        reader.get_glyph("A")

    assert error.value.glyph_name == "A"
    assert error.value.__cause__ is original_error


def test_cli_analyzes_each_glyph_once_in_parent_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    analyze = GlyphAnalyzer.analyze
    calls: Counter[str] = Counter()

    def count_analyze(self: GlyphAnalyzer, glyph: Glyph) -> ContourHierarchy:
        calls[glyph.name] += 1
        return analyze(self, glyph)

    monkeypatch.setattr(GlyphAnalyzer, "analyze", count_analyze)
    output_path = tmp_path / "CommitMono-Stenciled.otf"

    result = CliRunner().invoke(
        app,
        [
            str(COMMIT_MONO_PATH),
            "--output",
            str(output_path),
            "--workers",
            "1",
            "--quiet",
            "--log-file",
            str(tmp_path / "run.log"),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    assert output_path.is_file()
    assert calls, "The CLI did not analyze any glyphs in the parent process"
    duplicates = {name: count for name, count in calls.items() if count > 1}
    assert max(calls.values()) <= 1, f"Glyphs analyzed more than once: {duplicates}"
