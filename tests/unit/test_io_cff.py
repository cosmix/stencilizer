"""Unit tests for CFF and static CFF2 glyph writes in the converter."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.io.instance import instantiate_static
from stencilizer.io.reader import FontReader
from stencilizer.io.writer import FontWriter
from tests.font_helpers import CANTARELL, island_count


class TestCffGlyphUpdate:
    """Tests for CFF glyph update functionality."""

    def test_update_cff_glyph_passes_private_and_global_subrs(self):
        """Test that _update_cff_glyph passes Private dict and GlobalSubrs to charstring."""
        from stencilizer.io.converter import _update_cff_glyph

        # Create mock font structure
        mock_font = MagicMock()
        mock_cff_table = MagicMock()
        mock_top_dict = MagicMock()
        mock_charstrings: dict[str, MagicMock] = {}
        mock_private = MagicMock()
        mock_global_subrs = MagicMock()

        # Set up the CFF structure
        mock_cff_table.cff.topDictIndex = [mock_top_dict]
        mock_top_dict.CharStrings = mock_charstrings
        mock_top_dict.Private = mock_private
        mock_cff_table.cff.GlobalSubrs = mock_global_subrs

        mock_font.__getitem__ = Mock(return_value=mock_cff_table)
        mock_font.getGlyphSet.return_value = {}

        # Create a simple glyph with one contour
        glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[])

        # Mock the T2CharStringPen to track getCharString calls
        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = MagicMock()
            mock_charstring = MagicMock()
            mock_pen.getCharString.return_value = mock_charstring
            mock_pen_class.return_value = mock_pen

            _update_cff_glyph(glyph, None, mock_font)

            # Verify getCharString was called with private and globalSubrs
            mock_pen.getCharString.assert_called_once_with(
                private=mock_private, globalSubrs=mock_global_subrs, optimize=False
            )

    def test_update_cff_glyph_stores_charstring_in_font(self):
        """Test that the updated charstring is stored in the CharStrings dict."""
        from stencilizer.io.converter import _update_cff_glyph

        # Create mock font structure
        mock_font = MagicMock()
        mock_cff_table = MagicMock()
        mock_top_dict = MagicMock()
        mock_charstrings: dict[str, MagicMock] = {}
        mock_private = MagicMock()
        mock_global_subrs = MagicMock()

        # Set up the CFF structure
        mock_cff_table.cff.topDictIndex = [mock_top_dict]
        mock_top_dict.CharStrings = mock_charstrings
        mock_top_dict.Private = mock_private
        mock_cff_table.cff.GlobalSubrs = mock_global_subrs

        mock_font.__getitem__ = Mock(return_value=mock_cff_table)
        mock_font.getGlyphSet.return_value = {}

        # Create a glyph
        glyph = Glyph(metadata=GlyphMetadata("B", None, 600, 0), contours=[])

        # Mock the T2CharStringPen
        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = MagicMock()
            mock_charstring = MagicMock()
            mock_pen.getCharString.return_value = mock_charstring
            mock_pen_class.return_value = mock_pen

            _update_cff_glyph(glyph, None, mock_font)

            # Verify the charstring was stored in the CharStrings dict
            assert "B" in mock_charstrings
            assert mock_charstrings["B"] == mock_charstring

    def test_update_cff_glyph_empty_contours(self):
        """Test that _update_cff_glyph handles glyphs with no contours."""
        from stencilizer.io.converter import _update_cff_glyph

        # Create mock font structure
        mock_font = MagicMock()
        mock_cff_table = MagicMock()
        mock_top_dict = MagicMock()
        mock_charstrings: dict[str, MagicMock] = {}
        mock_private = MagicMock()
        mock_global_subrs = MagicMock()

        # Set up the CFF structure
        mock_cff_table.cff.topDictIndex = [mock_top_dict]
        mock_top_dict.CharStrings = mock_charstrings
        mock_top_dict.Private = mock_private
        mock_cff_table.cff.GlobalSubrs = mock_global_subrs

        mock_font.__getitem__ = Mock(return_value=mock_cff_table)
        mock_font.getGlyphSet.return_value = {}

        # Create a glyph with empty contours (e.g., space character)
        glyph = Glyph(metadata=GlyphMetadata("space", None, 250, 0), contours=[])

        # Mock the T2CharStringPen
        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = MagicMock()
            mock_charstring = MagicMock()
            mock_pen.getCharString.return_value = mock_charstring
            mock_pen_class.return_value = mock_pen

            _update_cff_glyph(glyph, None, mock_font)

            # Verify no drawing operations were performed (no moveTo calls)
            mock_pen.moveTo.assert_not_called()

            # Verify getCharString was still called (to create empty glyph)
            mock_pen.getCharString.assert_called_once()

            # Verify the charstring was stored
            assert "space" in mock_charstrings


class TestCff2GlyphUpdate:
    """Tests for static CFF2 glyph update functionality."""

    @staticmethod
    def _font() -> tuple[MagicMock, dict[str, MagicMock], MagicMock, MagicMock]:
        mock_font = MagicMock()
        mock_cff_table = MagicMock()
        mock_top_dict = MagicMock()
        charstrings = MagicMock()
        stored: dict[str, MagicMock] = {}
        charstrings.getItemAndSelector.return_value = (MagicMock(), 1)
        charstrings.__setitem__.side_effect = stored.__setitem__
        private = MagicMock()
        fd_other = MagicMock()
        mock_top_dict.FDArray = [fd_other, MagicMock(Private=private)]
        mock_top_dict.CharStrings = charstrings
        mock_cff_table.cff.topDictIndex = [mock_top_dict]
        mock_font.__getitem__ = Mock(return_value=mock_cff_table)
        mock_font.getGlyphSet.return_value = {}
        return mock_font, stored, private, mock_cff_table.cff.GlobalSubrs

    def test_uses_fdarray_private_and_global_subrs(self):
        from stencilizer.io.converter import _update_cff2_glyph

        mock_font, stored, private, global_subrs = self._font()
        glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[])

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = mock_pen_class.return_value
            _update_cff2_glyph(glyph, None, mock_font)

            mock_pen.getCharString.assert_called_once_with(
                private=private, globalSubrs=global_subrs, optimize=False
            )
        assert stored["A"] is mock_pen.getCharString.return_value

    def test_passes_cff2_flag_and_no_width_to_pen(self):
        from stencilizer.io.converter import _update_cff2_glyph

        mock_font, _, _, _ = self._font()
        glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[])

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            _update_cff2_glyph(glyph, None, mock_font)

        mock_pen_class.assert_called_once_with(width=None, glyphSet={}, CFF2=True)

    def test_reverses_points_to_cff_winding(self):
        from stencilizer.io.converter import _update_cff2_glyph

        mock_font, _, _, _ = self._font()
        points = [Point(0, 0), Point(10, 0), Point(10, 10)]
        glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[Contour(points=points)])

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            _update_cff2_glyph(glyph, None, mock_font)
            mock_pen = mock_pen_class.return_value

            mock_pen.moveTo.assert_called_once_with((10, 10))
            assert [call.args[0] for call in mock_pen.lineTo.call_args_list] == [(10, 0), (0, 0)]


class TestCidKeyedCffGlyphUpdate:
    """CID-keyed CFF fonts have no top-level Private; each glyph's FDSelect entry names it."""

    @staticmethod
    def _font(nominal_width: float = 0) -> tuple[MagicMock, dict[str, MagicMock], MagicMock]:
        mock_font = MagicMock()
        mock_cff_table = MagicMock()
        top_dict = MagicMock()
        charstrings = MagicMock()
        stored: dict[str, MagicMock] = {}
        charstrings.getItemAndSelector.return_value = (MagicMock(), 1)
        charstrings.__setitem__.side_effect = stored.__setitem__
        private = MagicMock(nominalWidthX=nominal_width)
        top_dict.Private = None
        top_dict.FDArray = [MagicMock(), MagicMock(Private=private)]
        top_dict.CharStrings = charstrings
        mock_cff_table.cff.topDictIndex = [top_dict]
        mock_font.__getitem__ = Mock(return_value=mock_cff_table)
        mock_font.getGlyphSet.return_value = {}
        return mock_font, stored, private

    def test_uses_the_private_dict_of_the_glyphs_font_dict(self):
        from stencilizer.io.converter import _update_cff_glyph

        mock_font, stored, private = self._font()
        glyph = Glyph(metadata=GlyphMetadata("cid00007", None, 500, 0), contours=[])

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = mock_pen_class.return_value
            _update_cff_glyph(glyph, None, mock_font)

            mock_pen.getCharString.assert_called_once()
            assert mock_pen.getCharString.call_args.kwargs["private"] is private
        assert stored["cid00007"] is mock_pen.getCharString.return_value

    def test_width_is_stored_relative_to_nominal_width(self):
        from stencilizer.io.converter import _update_cff_glyph

        mock_font, _, _ = self._font(nominal_width=100)
        glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[])

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            _update_cff_glyph(glyph, None, mock_font)

        mock_pen_class.assert_called_once_with(width=400, glyphSet={})


def _cff_glyph_state(path: Path, name: str) -> tuple[list[Any], float, bool]:
    """The glyph's charstring program, decoded width, and whether it uses its FD's Private."""
    with TTFont(path) as font:
        top_dict = font["CFF "].cff.topDictIndex[0]
        charstring, fd_index = top_dict.CharStrings.getItemAndSelector(name)
        charstring.draw(RecordingPen())
        expected = top_dict.FDArray[fd_index or 0].Private
        return list(charstring.program), charstring.width, charstring.private is expected


def _moved(glyph: Glyph, dx: float, dy: float) -> Glyph:
    contours = [
        Contour(points=[replace(p, x=p.x + dx, y=p.y + dy) for p in contour.points])
        for contour in glyph.contours
    ]
    return Glyph(metadata=glyph.metadata, contours=contours)


def _xy(glyph: Glyph) -> list[list[tuple[float, float]]]:
    return [[(p.x, p.y) for p in contour.points] for contour in glyph.contours]


@pytest.fixture(scope="module")
def cid_keyed_font(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A static CID-keyed CFF font: the Cantarell variable font pinned at wght=400."""
    return instantiate_static(CANTARELL, "wght=400", tmp_path_factory.mktemp("cid-keyed"))


class TestCidKeyedCffRoundTrip:
    """Writing and re-reading a glyph of a real CID-keyed CFF font."""

    def test_fixture_is_cid_keyed_with_a_nonzero_nominal_width(self, cid_keyed_font: Path):
        with TTFont(cid_keyed_font) as font:
            top_dict = font["CFF "].cff.topDictIndex[0]
            assert "CFF2" not in font
            assert hasattr(top_dict, "ROS")
            assert getattr(top_dict, "Private", None) is None
            assert top_dict.FDArray[0].Private.nominalWidthX != 0

    def test_written_glyph_reads_back_with_the_same_advance(
        self, cid_keyed_font: Path, tmp_path: Path
    ):
        output = tmp_path / "out.otf"
        with FontReader(cid_keyed_font) as reader:
            upm = reader.units_per_em
            glyph = next(g for g in reader.iter_glyphs() if island_count(g, upm) > 0)
            moved = _moved(glyph, 7, 3)
            before_program, _, _ = _cff_glyph_state(cid_keyed_font, glyph.name)

            writer = FontWriter(reader.font, output)
            writer.update_glyph(moved)
            writer.save()

        after_program, width, uses_fd_private = _cff_glyph_state(output, glyph.name)
        assert after_program != before_program
        assert uses_fd_private
        assert width == glyph.metadata.advance_width
        with FontReader(output) as reread:
            reloaded = reread.get_glyph(glyph.name)
        assert reloaded is not None
        assert _xy(reloaded) == _xy(moved)
        assert reloaded.metadata.advance_width == glyph.metadata.advance_width
