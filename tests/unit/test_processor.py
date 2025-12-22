"""Tests for parallel processing orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point, WindingDirection


@pytest.fixture
def sample_glyph_with_island() -> Glyph:
    """Create a sample glyph with one island."""
    # Outer contour (CW - TrueType convention)
    outer = Contour(
        points=[
            Point(0, 0),
            Point(0, 100),
            Point(100, 100),
            Point(100, 0),
        ],
        direction=WindingDirection.CLOCKWISE,
    )

    # Inner contour/island (CCW - TrueType convention)
    inner = Contour(
        points=[
            Point(25, 25),
            Point(75, 25),
            Point(75, 75),
            Point(25, 75),
        ],
        direction=WindingDirection.COUNTER_CLOCKWISE,
    )

    metadata = GlyphMetadata(
        name="O",
        unicode=ord("O"),
        advance_width=100,
        left_side_bearing=0,
    )

    return Glyph(metadata=metadata, contours=[outer, inner])


@pytest.fixture
def sample_glyph_no_island() -> Glyph:
    """Create a sample glyph without islands."""
    # Outer contour (CW - TrueType convention)
    outer = Contour(
        points=[
            Point(0, 0),
            Point(0, 100),
            Point(50, 100),
            Point(50, 0),
        ],
        direction=WindingDirection.CLOCKWISE,
    )

    metadata = GlyphMetadata(
        name="I",
        unicode=ord("I"),
        advance_width=50,
        left_side_bearing=0,
    )

    return Glyph(metadata=metadata, contours=[outer])


@pytest.fixture
def bridge_config() -> BridgeConfig:
    """Create test bridge configuration."""
    return BridgeConfig(
        width_percent=60.0,
        min_bridges=1,
        sample_count=36,
    )


@pytest.fixture
def settings(bridge_config: BridgeConfig) -> StencilizerSettings:
    """Create test stencilizer settings."""
    settings = StencilizerSettings()
    settings.bridge = bridge_config
    return settings


class TestProcessGlyph:
    """Tests for process_glyph function."""

    def test_process_glyph_with_island(
        self, sample_glyph_with_island: Glyph, bridge_config: BridgeConfig
    ):
        """Test processing a glyph with an island."""
        glyph_dict = sample_glyph_with_island.to_dict()
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(glyph_dict, config_dict, upm)

        assert "error" not in result
        assert "glyph" in result
        assert "bridges_added" in result

        # Verify the glyph can be deserialized
        transformed = Glyph.from_dict(result["glyph"])
        assert transformed.name == "O"
        assert isinstance(transformed, Glyph)

    def test_process_glyph_no_island(
        self, sample_glyph_no_island: Glyph, bridge_config: BridgeConfig
    ):
        """Test processing a glyph without islands."""
        glyph_dict = sample_glyph_no_island.to_dict()
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(glyph_dict, config_dict, upm)

        assert "error" not in result
        assert result["bridges_added"] == 0

        # Glyph should be unchanged
        transformed = Glyph.from_dict(result["glyph"])
        assert len(transformed.contours) == 1

    def test_process_glyph_handles_error(self, bridge_config: BridgeConfig):
        """Test that process_glyph handles errors gracefully."""
        # Invalid glyph dict
        invalid_dict = {"metadata": {"name": "test"}}
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(invalid_dict, config_dict, upm)

        assert "error" in result
        assert "glyph_name" in result
        assert "traceback" in result
        assert result["glyph_name"] == "test"

    def test_process_glyph_serialization_roundtrip(
        self, sample_glyph_with_island: Glyph, bridge_config: BridgeConfig
    ):
        """Test that glyph serialization works correctly."""
        glyph_dict = sample_glyph_with_island.to_dict()
        config_dict = bridge_config.model_dump()
        upm = 1000

        result = process_glyph(glyph_dict, config_dict, upm)

        assert "glyph" in result
        transformed = Glyph.from_dict(result["glyph"])

        # Verify metadata preserved
        assert transformed.metadata.name == sample_glyph_with_island.metadata.name
        assert transformed.metadata.unicode == sample_glyph_with_island.metadata.unicode
        assert transformed.metadata.advance_width == sample_glyph_with_island.metadata.advance_width


class TestFontProcessor:
    """Tests for FontProcessor class."""

    def test_init(self, settings: StencilizerSettings):
        """Test FontProcessor initialization."""
        with patch('stencilizer.core.processor.configure_logging') as mock_logging:
            mock_logging.return_value = Mock()
            processor = FontProcessor(settings)

            assert processor.config == settings
            mock_logging.assert_called_once()

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    def test_process_no_glyphs_to_process(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_no_island: Glyph,
    ):
        """Test processing a font with no glyphs needing processing."""
        # Setup mocks
        mock_logging.return_value = Mock()
        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_no_island]
        mock_reader._font = Mock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        processor = FontProcessor(settings)
        stats = processor.process(Path("input.ttf"))

        assert stats.processed_count == 0
        assert stats.skipped_count == 1
        assert stats.error_count == 0
        assert stats.bridges_added == 0
        assert stats.duration_seconds >= 0

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    @patch('stencilizer.core.processor.ProcessPoolExecutor')
    def test_process_with_glyphs(
        self,
        mock_executor_class,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_with_island: Glyph,
    ):
        """Test processing a font with glyphs requiring processing."""
        # Setup mocks
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_with_island]
        mock_reader._font = Mock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        # Mock executor
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = {
            "glyph": sample_glyph_with_island.to_dict(),
            "bridges_added": 1,
        }
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Mock as_completed to return futures immediately
        with patch('stencilizer.core.processor.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            processor = FontProcessor(settings)
            stats = processor.process(Path("input.ttf"), max_workers=1)

            assert stats.processed_count == 1
            assert stats.bridges_added == 1
            assert stats.error_count == 0

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    def test_process_skip_empty_glyphs(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
    ):
        """Test that empty glyphs are skipped."""
        mock_logging.return_value = Mock()

        # Create empty glyph
        empty_glyph = Glyph(
            metadata=GlyphMetadata(
                name="space",
                unicode=ord(" "),
                advance_width=250,
                left_side_bearing=0,
            ),
            contours=[],
        )

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [empty_glyph]
        mock_reader._font = Mock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        processor = FontProcessor(settings)
        stats = processor.process(Path("input.ttf"))

        assert stats.processed_count == 0
        assert stats.skipped_count == 1

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    def test_process_skip_composite_glyphs(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_with_island: Glyph,
    ):
        """Test that composite glyphs are skipped when configured."""
        mock_logging.return_value = Mock()

        # Mark glyph as composite
        sample_glyph_with_island._is_composite = True

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_with_island]
        mock_reader._font = Mock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("output.ttf")

        # Ensure skip_composite is True
        settings.processing.skip_composite = True

        processor = FontProcessor(settings)
        stats = processor.process(Path("input.ttf"))

        assert stats.processed_count == 0
        assert stats.skipped_count == 1

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    @patch('stencilizer.core.processor.ProcessPoolExecutor')
    def test_process_handles_errors(
        self,
        mock_executor_class,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
        sample_glyph_with_island: Glyph,
    ):
        """Test that processing errors are handled gracefully."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 1
        mock_reader.iter_glyphs.return_value = [sample_glyph_with_island]
        mock_reader._font = Mock()
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

        with patch('stencilizer.core.processor.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            processor = FontProcessor(settings)
            stats = processor.process(Path("input.ttf"), max_workers=1)

            assert stats.processed_count == 0
            assert stats.error_count == 1

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
    def test_process_custom_output_path(
        self,
        mock_logging,
        mock_writer_class,
        mock_reader_class,
        settings: StencilizerSettings,
    ):
        """Test processing with custom output path."""
        mock_logging.return_value = Mock()

        mock_reader = Mock()
        mock_reader.units_per_em = 1000
        mock_reader.format = "TrueType"
        mock_reader.glyph_count = 0
        mock_reader.iter_glyphs.return_value = []
        mock_reader._font = Mock()
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        custom_output = Path("custom-output.ttf")
        processor = FontProcessor(settings)
        processor.process(Path("input.ttf"), output_path=custom_output)

        # Verify FontWriter was created with custom path
        mock_writer_class.assert_called_once()
        call_args = mock_writer_class.call_args
        assert call_args[0][1] == custom_output

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.FontWriter')
    @patch('stencilizer.core.processor.configure_logging')
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
        mock_reader_class.return_value = mock_reader

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer_class.get_stenciled_path.return_value = Path("input-Stenciled.ttf")

        processor = FontProcessor(settings)
        processor.process(Path("input.ttf"))

        # Verify get_stenciled_path was called
        mock_writer_class.get_stenciled_path.assert_called_once_with(Path("input.ttf"))

    @patch('stencilizer.core.processor.FontReader')
    @patch('stencilizer.core.processor.configure_logging')
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
