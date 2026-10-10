"""Unit tests for update_font_names."""

from unittest.mock import MagicMock, Mock

from stencilizer.io.writer import update_font_names


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

        mock_name_table.setName.assert_called_once_with("Roboto Stenciled", 1, 3, 1, 0x409)

    def test_full_name_inserts_suffix_before_style(self):
        """Test that full name (nameID 4) inserts suffix before style."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(4, "Roboto Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with("Roboto Stenciled Regular", 4, 3, 1, 0x409)

    def test_full_name_single_word(self):
        """Test that single-word full name gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(4, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with("Roboto Stenciled", 4, 3, 1, 0x409)

    def test_postscript_name_inserts_before_hyphen(self):
        """Test that PostScript name (nameID 6) inserts suffix before hyphen."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(6, "Roboto-Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with("RobotoStenciled-Regular", 6, 3, 1, 0x409)

    def test_postscript_name_no_hyphen(self):
        """Test that PostScript name without hyphen gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(6, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with("RobotoStenciled", 6, 3, 1, 0x409)

    def test_typographic_family_gets_suffix(self):
        """Test that typographic family (nameID 16) gets suffix appended."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        mock_name_table.names = [self._make_name_record(16, "Roboto")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_called_once_with("Roboto Stenciled", 16, 3, 1, 0x409)

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

        mock_name_table.setName.assert_called_once_with("Roboto Custom", 1, 3, 1, 0x409)

    def test_ignores_other_name_ids(self):
        """Test that other nameIDs are not modified."""
        mock_font = MagicMock()
        mock_name_table = MagicMock()
        # nameID 2 is Subfamily (Regular, Bold, etc.) - should not be modified
        mock_name_table.names = [self._make_name_record(2, "Regular")]
        mock_font.__getitem__ = Mock(return_value=mock_name_table)

        update_font_names(mock_font)

        mock_name_table.setName.assert_not_called()
