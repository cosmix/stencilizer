"""Unit tests for the Font I/O layer.

Tests for FontReader, FontWriter, and converter functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from stencilizer.domain.contour import PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.io.reader import FontReader
from stencilizer.io.writer import FontWriter, update_font_names


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
    def test_format_truetype(self, _mock_exists, mock_ttfont):  # noqa: ARG002
        """Test format property for TrueType fonts."""
        mock_font = MagicMock()
        mock_font.__contains__ = Mock(side_effect=lambda x: x == "glyf")
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.format == "TrueType"

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_format_opentype(self, _mock_exists, mock_ttfont):  # noqa: ARG002
        """Test format property for OpenType fonts."""
        mock_font = MagicMock()
        mock_font.__contains__ = Mock(side_effect=lambda x: x == "CFF ")
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.otf"))
        reader.load()

        assert reader.format == "OpenType"

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_units_per_em(self, _mock_exists, mock_ttfont):  # noqa: ARG002
        """Test units_per_em property."""
        mock_font = MagicMock()
        mock_font.__getitem__ = Mock(return_value=MagicMock(unitsPerEm=1000))
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.units_per_em == 1000

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_glyph_count(self, _mock_exists, mock_ttfont):  # noqa: ARG002
        """Test glyph_count property."""
        mock_font = MagicMock()
        mock_font.__getitem__ = Mock(return_value=MagicMock(numGlyphs=256))
        mock_ttfont.return_value = mock_font

        reader = FontReader(Path("test.ttf"))
        reader.load()

        assert reader.glyph_count == 256

    @patch("stencilizer.io.reader.TTFont")
    @patch.object(Path, "exists", return_value=True)
    def test_context_manager(self, _mock_exists, mock_ttfont):  # noqa: ARG002
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
            (Path("font.ttf"), Path("font-Stenciled.ttf")),
            (Path("Roboto-Regular.otf"), Path("Roboto-Regular-Stenciled.otf")),
            (Path("/path/to/MyFont.ttf"), Path("/path/to/MyFont-Stenciled.ttf")),
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


class TestUpdateFontNames:
    """Tests for update_font_names function."""

    def _make_name_record(self, name_id: int, value: str):
        """Create a mock name record."""
        record = MagicMock()
        record.nameID = name_id
        record.platformID = 3
        record.platEncID = 1
        record.langID = 0x409
        record.toUnicode.return_value = value
        return record

    def test_family_name_gets_suffix(self):
        """Test that family name (nameID 1) gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(1, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "Roboto Stenciled", 1, 3, 1, 0x409
        )

    def test_full_name_inserts_suffix_before_style(self):
        """Test that full name (nameID 4) inserts suffix before style."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(4, "Roboto Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "Roboto Stenciled Regular", 4, 3, 1, 0x409
        )

    def test_full_name_single_word(self):
        """Test that single-word full name gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(4, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "Roboto Stenciled", 4, 3, 1, 0x409
        )

    def test_postscript_name_inserts_before_hyphen(self):
        """Test that PostScript name (nameID 6) inserts suffix before hyphen."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(6, "Roboto-Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "RobotoStenciled-Regular", 6, 3, 1, 0x409
        )

    def test_postscript_name_no_hyphen(self):
        """Test that PostScript name without hyphen gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(6, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "RobotoStenciled", 6, 3, 1, 0x409
        )

    def test_typographic_family_gets_suffix(self):
        """Test that typographic family (nameID 16) gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(16, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with(
            "Roboto Stenciled", 16, 3, 1, 0x409
        )

    def test_multiple_name_records(self):
        """Test that all relevant name records are updated."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [
            self._make_name_record(1, "Roboto"),
            self._make_name_record(4, "Roboto Regular"),
            self._make_name_record(6, "Roboto-Regular"),
        ]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        assert mock_name_table.setName.call_count == 3

    def test_custom_suffix(self):
        """Test that custom suffix can be used."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(1, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font, suffix=" Custom")

        mock_name_table.setName.assert_called_once_with(
            "Roboto Custom", 1, 3, 1, 0x409
        )

    def test_ignores_other_name_ids(self):
        """Test that other nameIDs are not modified."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        # nameID 2 is Subfamily (Regular, Bold, etc.) - should not be modified
        mock_name_table.names = [self._make_name_record(2, "Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_not_called()
