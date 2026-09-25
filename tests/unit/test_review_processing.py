"""Failure and outcome contracts for font processing."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from stencilizer.cli.app import _classify_font, app
from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core.processor import FontProcessor, GlyphClassification, process_glyph
from stencilizer.domain import Glyph
from stencilizer.exceptions import (
    FontFormatError,
    FontProcessingError,
    FontSaveError,
    GlyphProcessingError,
)
from stencilizer.utils import ProcessingStats, configure_logging


def test_update_failure_preserves_existing_output(
    tmp_path: Path, settings: StencilizerSettings, sample_glyph_with_island: Glyph
) -> None:
    output = tmp_path / "result.ttf"
    output.write_bytes(b"existing")
    reader = Mock()
    writer = Mock()
    writer.update_glyph.side_effect = ValueError("cannot encode contour")
    with (
        patch("stencilizer.core.processor.configure_logging", return_value=Mock()),
        patch("stencilizer.core.processor.FontWriter", return_value=writer),
    ):
        processor = FontProcessor(settings)
        with pytest.raises(FontSaveError, match="cannot encode contour"):
            processor._save_font(reader, output, {"O": sample_glyph_with_island})

    assert output.read_bytes() == b"existing"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.ttf"]
    writer.save.assert_not_called()


def test_save_failure_does_not_publish_output(
    tmp_path: Path, settings: StencilizerSettings
) -> None:
    output = tmp_path / "result.ttf"
    writer = Mock()
    writer.save.side_effect = FontSaveError(str(output), "serialization failed")
    with (
        patch("stencilizer.core.processor.configure_logging", return_value=Mock()),
        patch("stencilizer.core.processor.FontWriter", return_value=writer),
    ):
        processor = FontProcessor(settings)
        with pytest.raises(FontSaveError, match="serialization failed"):
            processor._save_font(Mock(), output, {})

    assert list(tmp_path.iterdir()) == []


def test_reader_failure_keeps_glyph_context(tmp_path: Path) -> None:
    error = GlyphProcessingError("A", "malformed outline")
    processor = Mock()
    processor.classify_glyphs.side_effect = error
    reader = MagicMock()
    reader.__enter__.return_value = reader
    with (
        patch("stencilizer.cli.app.FontReader", return_value=reader),
        pytest.raises(GlyphProcessingError) as caught,
    ):
        _classify_font(tmp_path / "font.ttf", processor, quiet=True)
    assert caught.value is error


def test_all_workers_failing_gives_nonzero_exit_without_output(
    tmp_path: Path, sample_glyph_with_island: Glyph
) -> None:
    font_path = tmp_path / "input.ttf"
    output = tmp_path / "output.ttf"
    font_path.write_bytes(b"font")
    classification = GlyphClassification(glyphs_to_process=[sample_glyph_with_island])
    with (
        patch("stencilizer.cli.app._classify_font", return_value=classification),
        patch("stencilizer.cli.app.FontProcessor") as processor_class,
    ):
        processor_class.return_value.process.side_effect = FontProcessingError(
            [(sample_glyph_with_island.name, "worker crashed")]
        )
        result = CliRunner().invoke(app, [str(font_path), "--output", str(output), "--quiet"])
    assert result.exit_code == 1
    assert "worker crashed" in result.output
    assert not output.exists()


def test_quiet_mode_reports_unbridged_islands(
    tmp_path: Path, sample_glyph_with_island: Glyph
) -> None:
    font_path = tmp_path / "input.ttf"
    font_path.write_bytes(b"font")
    classification = GlyphClassification(glyphs_to_process=[sample_glyph_with_island])
    with (
        patch("stencilizer.cli.app._classify_font", return_value=classification),
        patch("stencilizer.cli.app.FontProcessor") as processor_class,
    ):
        processor_class.return_value.process.return_value = ProcessingStats(unbridged_count=1)
        result = CliRunner().invoke(app, [str(font_path), "--quiet"])
    assert result.exit_code == 0
    assert "1 island remained unbridged" in result.output


def test_unsupported_format_keeps_its_error_type(tmp_path: Path) -> None:
    error = FontFormatError(str(tmp_path / "font.ttf"), "CFF2 is unsupported")
    reader = MagicMock()
    reader.__enter__.side_effect = error
    with (
        patch("stencilizer.cli.app.FontReader", return_value=reader),
        pytest.raises(FontFormatError) as caught,
    ):
        _classify_font(tmp_path / "font.ttf", Mock(), quiet=True)
    assert caught.value is error


def test_worker_reports_actual_zero_bridges(
    sample_glyph_with_island: Glyph, bridge_config: BridgeConfig
) -> None:
    outcome = SimpleNamespace(glyph=sample_glyph_with_island, bridge_count=0, unbridged_count=1)
    with patch("stencilizer.core.processor.GlyphTransformer") as transformer:
        transformer.return_value.transform_with_outcome.return_value = outcome
        result = process_glyph(sample_glyph_with_island.to_dict(), bridge_config.model_dump(), 1000)
    assert result["bridges_added"] == 0
    assert result["unbridged_count"] == 1


def test_unbridged_islands_are_not_counted_as_failures(settings: StencilizerSettings) -> None:
    glyph = Mock()
    glyph.name = "O"
    future = Mock()
    future.result.return_value = {
        "glyph": {"metadata": {"name": "O"}},
        "bridges_added": 0,
        "unbridged_count": 1,
    }
    stats = ProcessingStats()
    processed: dict[str, Glyph] = {}
    with (
        patch("stencilizer.core.processor.configure_logging", return_value=Mock()),
        patch("stencilizer.core.processor.Glyph.from_dict", return_value=glyph),
    ):
        processor = FontProcessor(settings)
        assert processor._collect_result(future, "O", stats, processed)
    assert stats.bridges_added == 0
    assert stats.unbridged_count == 1
    assert stats.error_count == 0
    assert processed == {}


def test_connected_glyph_is_queued_for_writing(
    settings: StencilizerSettings, sample_glyph_with_island: Glyph
) -> None:
    future = Mock()
    future.result.return_value = {
        "glyph": sample_glyph_with_island.to_dict(),
        "bridges_added": 1,
        "unbridged_count": 0,
    }
    stats = ProcessingStats()
    processed: dict[str, Glyph] = {}
    with patch("stencilizer.core.processor.configure_logging", return_value=Mock()):
        processor = FontProcessor(settings)
        assert processor._collect_result(future, sample_glyph_with_island.name, stats, processed)
    assert processed[sample_glyph_with_island.name].name == sample_glyph_with_island.name
    assert stats.bridges_added == 1


def test_reconfigure_logging_replaces_owned_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    root_level = root.level
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    logger = configure_logging(first, quiet=True)
    logger.info("first message")
    owned = logging.getLogger("stencilizer")
    first_handlers = owned.handlers[:]
    logger = configure_logging(second, quiet=True)
    logger.info("second message")

    assert root.handlers == root_handlers
    assert root.level == root_level
    assert len(owned.handlers) == 2
    assert all(
        handler.stream is None
        for handler in first_handlers
        if isinstance(handler, logging.FileHandler)
    )
    assert "second message" not in first.read_text()
    assert second.read_text().count("second message") == 1
    console = next(
        handler
        for handler in owned.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    )
    assert console.level == logging.ERROR
