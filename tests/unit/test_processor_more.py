"""Additional processor and timing tests."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.domain import Glyph
from stencilizer.exceptions import FontProcessingError


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class TestFontProcessor:
    @patch("stencilizer.core.processor.FontReader")
    @patch("stencilizer.core.processor.FontWriter")
    @patch("stencilizer.core.processor.configure_logging")
    @patch("stencilizer.core.processor.ProcessPoolExecutor")
    def test_process_handles_errors(
        self,
        mock_executor_class,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_with_island: Glyph,
    ):
        """Worker failures abort without publishing a font."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_with_island]
        mock_reader._font = Mock()
        mock_reader.font = MagicMock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        # Mock executor to return error
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = {
            "error": "Test error",
            "glyph_name": "O",
            "traceback": "Traceback...",
        }
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        with patch("stencilizer.core.processor.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            processor = FontProcessor(settings)
            with pytest.raises(FontProcessingError, match="O: Test error") as error:
                processor.process(Path("input.ttf"), max_workers=1)

            assert error.value.errors == [("O", "Test error")]
            mock_writer_class.assert_not_called()

    @patch("stencilizer.core.processor.FontReader")
    @patch("stencilizer.core.processor.FontWriter")
    @patch("stencilizer.core.processor.configure_logging")
    def test_process_custom_output_path(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        tmp_path: Path,
    ):
        """Test processing with custom output path."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 0
        mock_reader.iter_glyphs.return_value = []
        mock_reader._font = Mock()
        mock_reader.font = MagicMock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        custom_output = tmp_path / "custom-output.ttf"
        mock_writer.save.side_effect = lambda: mock_writer_class.call_args[0][1].write_bytes(
            b"saved font"
        )
        processor = FontProcessor(settings)
        processor.process(Path("input.ttf"), output_path=custom_output)

        # Save to a sibling temporary file, then publish at the requested path.
        mock_writer_class.assert_called_once()
        temporary_output = mock_writer_class.call_args[0][1]
        assert temporary_output.parent == tmp_path
        assert temporary_output.suffix == ".ttf"
        assert custom_output.read_bytes() == b"saved font"
        assert sorted(tmp_path.iterdir()) == [custom_output]

    @patch("stencilizer.core.processor.FontReader")
    @patch("stencilizer.core.processor.FontWriter")
    @patch("stencilizer.core.processor.configure_logging")
    def test_process_auto_output_path(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
    ):
        """Test processing with auto-generated output path."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 0
        mock_reader.iter_glyphs.return_value = []
        mock_reader._font = Mock()
        mock_reader.font = MagicMock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("input-Stenciled.ttf")

        processor = FontProcessor(settings)
        processor.process(Path("input.ttf"))

        # Verify get_stenciled_path was called
        mock_writer_class.get_stenciled_path.assert_called_once_with(Path("input.ttf"))

    @patch("stencilizer.core.processor.FontReader")
    @patch("stencilizer.core.processor.configure_logging")
    def test_process_font_not_found(
        self,
        mock_logging,
        mock_reader_class,
        settings: StencilizerSettings,
    ):
        """Test processing with non-existent font file."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.load.side_effect = FileNotFoundError("Font not found")
        mock_reader_class.return_value = mock_reader

        processor = FontProcessor(settings)

        with pytest.raises(FileNotFoundError):
            processor.process(Path("nonexistent.ttf"))


class TestProcessGlyphTiming:
    """Tests for per-glyph timing functionality."""

    def test_process_glyph_returns_duration_ms(
        self, sample_glyph_with_island: Glyph, bridge_config: BridgeConfig
    ):
        """Test that process_glyph returns duration_ms in result."""
        glyph_dict = sample_glyph_with_island.to_dict()
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(glyph_dict, config_dict, upm)

        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0

    def test_process_glyph_error_includes_duration_ms(self, bridge_config: BridgeConfig):
        """Test that error results also include duration_ms."""
        invalid_dict = {"metadata": {"name": "test"}}
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(invalid_dict, config_dict, upm)

        assert "error" in result
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0


class TestProcessingStatsTiming:
    """Tests for ProcessingStats timing aggregation."""

    def test_timing_aggregation_empty(self):
        """Test timing aggregation with no timings."""
        from stencilizer.utils import ProcessingStats

        stats = ProcessingStats()

        assert stats.min_glyph_time_ms is None
        assert stats.max_glyph_time_ms is None
        assert stats.avg_glyph_time_ms is None

    def test_timing_aggregation_single_value(self):
        """Test timing aggregation with single timing."""
        from stencilizer.utils import ProcessingStats

        stats = ProcessingStats()
        stats.glyph_timings_ms.append(10.5)

        assert stats.min_glyph_time_ms == 10.5
        assert stats.max_glyph_time_ms == 10.5
        assert stats.avg_glyph_time_ms == 10.5

    def test_timing_aggregation_multiple_values(self):
        """Test timing aggregation with multiple timings."""
        from stencilizer.utils import ProcessingStats

        stats = ProcessingStats()
        stats.glyph_timings_ms = [10.0, 20.0, 30.0]

        assert stats.min_glyph_time_ms == 10.0
        assert stats.max_glyph_time_ms == 30.0
        assert stats.avg_glyph_time_ms == 20.0

    def test_cancellation_fields_default(self):
        """Test that cancellation fields have correct defaults."""
        from stencilizer.utils import ProcessingStats

        stats = ProcessingStats()

        assert stats.cancelled_count == 0
        assert stats.was_cancelled is False

    def test_cancellation_fields_set(self):
        """Test setting cancellation fields."""
        from stencilizer.utils import ProcessingStats

        stats = ProcessingStats()
        stats.was_cancelled = True
        stats.cancelled_count = 5

        assert stats.was_cancelled is True
        assert stats.cancelled_count == 5


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @patch("stencilizer.core.processor.FontReader")
    @patch("stencilizer.core.processor.FontWriter")
    @patch("stencilizer.core.processor.configure_logging")
    @patch("stencilizer.core.processor.ProcessPoolExecutor")
    def test_progress_callback_invoked(
        self,
        mock_executor_class,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_with_island: Glyph,
    ):
        """Test that progress callback is invoked for processed glyphs."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_with_island]
        mock_reader._font = Mock()
        mock_reader.font = MagicMock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        mock_future = MagicMock()
        mock_future.result.return_value = {
            "glyph": sample_glyph_with_island.to_dict(),
            "bridges_added": 1,
            "duration_ms": 10.5,
        }

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor_class.return_value = mock_executor

        callback_calls = []

        def progress_callback(completed, total, glyph_name, success):
            callback_calls.append((completed, total, glyph_name, success))

        with patch("stencilizer.core.processor.as_completed") as mock_as_completed:

            def as_completed_impl(futures_dict):
                return iter(list(futures_dict.keys()))

            mock_as_completed.side_effect = as_completed_impl

            processor = FontProcessor(settings)
            processor.process(
                Path("input.ttf"),
                progress_callback=progress_callback,
            )

        # Should have been called once (for the one glyph with island)
        assert len(callback_calls) == 1
        # Check callback parameters
        completed, total, glyph_name, success = callback_calls[0]
        assert completed == 1
        assert total == 1
        assert glyph_name == "O"
        assert success is True
