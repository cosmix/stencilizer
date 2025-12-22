"""Unit tests for the Font I/O layer.

Tests for FontReader, FontWriter, and converter functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from stencilizer.domain.contour import PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.io.reader import FontReader
from stencilizer.io.writer import FontWriter


class TestFontReader:
    """Tests for FontReader class."""

    def test_init(self):
        """Test FontReader initialization."""
        path = Path("test.ttf")
        reader = FontReader(path)
        assert reader._font_path == path
        assert reader._font is None

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises FileNotFoundError."""
        reader = FontReader(Path("nonexistent.ttf"))
        with pytest.raises(FileNotFoundError):
            reader.load()

    def test_format_before_load(self):
        """Test accessing format before loading raises RuntimeError."""
        reader = FontReader(Path("test.ttf"))
        with pytest.raises(RuntimeError, match="Font not loaded"):
            _ = reader.format

    def test_units_per_em_before_load(self):
        """Test accessing units_per_em before loading raises RuntimeError."""
        reader = FontReader(Path("test.ttf"))
        with pytest.raises(RuntimeError, match="Font not loaded"):
            _ = reader.units_per_em

    def test_glyph_count_before_load(self):
        """Test accessing glyph_count before loading raises RuntimeError."""
        reader = FontReader(Path("test.ttf"))
        with pytest.raises(RuntimeError, match="Font not loaded"):
            _ = reader.glyph_count

    def test_iter_glyphs_before_load(self):
        """Test iterating glyphs before loading raises RuntimeError."""
        reader = FontReader(Path("test.ttf"))
        with pytest.raises(RuntimeError, match="Font not loaded"):
            list(reader.iter_glyphs())

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_format_truetype(self, __mock_exists, mock_ttfont):
        """Test format property for TrueType fonts."""
        mock_font = MagicMock()
        mock_font.__contains__ = Mock(side_effect=lambda x: x == "glyf")
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.format == "TrueType"

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_format_opentype(self, __mock_exists, mock_ttfont):
        """Test format property for OpenType fonts."""
        mock_font = MagicMock()
        mock_font.__contains__ = Mock(side_effect=lambda x: x == "CFF ")
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.otf"))
        reader.load()

        assert reader.format == "OpenType"

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_units_per_em(self, __mock_exists, mock_ttfont):
        """Test units_per_em property."""
        mock_font = MagicMock()
        mock_font.__getitem__ = Mock(return_value=MagicMock(unitsPerEm=1000))
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.units_per_em == 1000

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_glyph_count(self, __mock_exists, mock_ttfont):
        """Test glyph_count property."""
        mock_font = MagicMock()
        mock_font.__getitem__ = Mock(return_value=MagicMock(numGlyphs=256))
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.glyph_count == 256

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_context_manager(self, __mock_exists, mock_ttfont):
        """Test FontReader as context manager."""
        mock_font = MagicMock()
        mock_ttfont.return_value = mock_font

        with FontReader(Path("test.ttf")) as reader:
            assert reader._font is not None

        mock_font.close.assert_called_once()


class TestFontWriter:
    """Tests for FontWriter class."""

    def test_init(self):
        """Test FontWriter initialization."""
        mock_font = MagicMock()
        path = Path("output.ttf")
        writer = FontWriter(mock_font, path)

        assert writer._font == mock_font
        assert writer._output_path == path

    def test_get_stenciled_path(self):
        """Test get_stenciled_path static method."""
        test_cases = [
            (Path("font.ttf"), Path("font-stenciled.ttf")),
            (Path("Roboto-Regular.otf"), Path("Roboto-Regular-stenciled.otf")),
            (Path("/path/to/MyFont.ttf"), Path("/path/to/MyFont-stenciled.ttf")),
        ]

        for input_path, expected_output in test_cases:
            result = FontWriter.get_stenciled_path(input_path)
            assert result == expected_output

    def test_save(self):
        """Test save method calls font.save."""
        mock_font = MagicMock()
        output_path = Path("output.ttf")
        writer = FontWriter(mock_font, output_path)

        writer.save()

        mock_font.save.assert_called_once_with(str(output_path))

    def test_update_glyph_not_found(self):
        """Test update_glyph raises ValueError for unknown glyph."""
        mock_font = MagicMock()
        mock_font.getGlyphOrder.return_value = ["A", "B", "C"]

        writer = FontWriter(mock_font, Path("output.ttf"))

        glyph = Glyph(
            metadata=GlyphMetadata("Z", None, 500, 0),
            contours=[]
        )

        with pytest.raises(ValueError, match="not found in font"):
            writer.update_glyph(glyph)


class TestConverter:
    """Tests for converter functions."""

    def test_recording_to_contours_simple(self):
        """Test conversion of simple recording to contours."""
        from stencilizer.io.converter import _recording_to_contours

        recording = [
            ("moveTo", ((0.0, 0.0),)),
            ("lineTo", ((100.0, 0.0),)),
            ("lineTo", ((100.0, 100.0),)),
            ("lineTo", ((0.0, 100.0),)),
            ("closePath", ()),
        ]

        contours = _recording_to_contours(recording)

        assert len(contours) == 1
        assert len(contours[0].points) == 4
        assert all(p.point_type == PointType.ON_CURVE for p in contours[0].points)

    def test_recording_to_contours_quadratic(self):
        """Test conversion with quadratic curves."""
        from stencilizer.io.converter import _recording_to_contours

        recording = [
            ("moveTo", ((0.0, 0.0),)),
            ("qCurveTo", ((50.0, 50.0), (100.0, 0.0))),
            ("closePath", ()),
        ]

        contours = _recording_to_contours(recording)

        assert len(contours) == 1
        assert len(contours[0].points) == 3
        assert contours[0].points[0].point_type == PointType.ON_CURVE
        assert contours[0].points[1].point_type == PointType.OFF_CURVE_QUAD
        assert contours[0].points[2].point_type == PointType.ON_CURVE

    def test_recording_to_contours_cubic(self):
        """Test conversion with cubic curves."""
        from stencilizer.io.converter import _recording_to_contours

        recording = [
            ("moveTo", ((0.0, 0.0),)),
            ("curveTo", ((33.0, 33.0), (66.0, 66.0), (100.0, 0.0))),
            ("closePath", ()),
        ]

        contours = _recording_to_contours(recording)

        assert len(contours) == 1
        assert len(contours[0].points) == 4
        assert contours[0].points[0].point_type == PointType.ON_CURVE
        assert contours[0].points[1].point_type == PointType.OFF_CURVE_CUBIC
        assert contours[0].points[2].point_type == PointType.OFF_CURVE_CUBIC
        assert contours[0].points[3].point_type == PointType.ON_CURVE

    def test_recording_to_contours_multiple(self):
        """Test conversion with multiple contours."""
        from stencilizer.io.converter import _recording_to_contours

        recording = [
            ("moveTo", ((0.0, 0.0),)),
            ("lineTo", ((10.0, 0.0),)),
            ("closePath", ()),
            ("moveTo", ((20.0, 20.0),)),
            ("lineTo", ((30.0, 20.0),)),
            ("closePath", ()),
        ]

        contours = _recording_to_contours(recording)

        assert len(contours) == 2
        assert len(contours[0].points) == 2
        assert len(contours[1].points) == 2
