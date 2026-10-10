"""Tests for the static CFF2 glyph write path on a real converted font."""

from dataclasses import replace
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour
from stencilizer.domain.glyph import Glyph
from stencilizer.io import FontReader, FontWriter
from stencilizer.io.converter import _update_cff2_glyph
from tests.font_helpers import cff2_program, write_commit_mono_cff2

SHIFT = 10


@pytest.fixture
def cff2_path(tmp_path: Path) -> Path:
    """The CommitMono fixture converted to a static CFF2 font."""
    return write_commit_mono_cff2(tmp_path / "commitmono-cff2.otf")


def _shifted(glyph: Glyph) -> Glyph:
    contours = [
        Contour([replace(point, x=point.x + SHIFT) for point in contour.points])
        for contour in glyph.contours
    ]
    return Glyph(metadata=glyph.metadata, contours=contours)


def test_no_fdselect_uses_first_font_dict_private(cff2_path: Path) -> None:
    """A CFF2 font without FDSelect reports no selector and takes FDArray[0]."""
    with FontReader(cff2_path) as reader:
        glyph = reader.get_glyph("o")
        assert glyph is not None
        top_dict = reader.font["CFF2"].cff.topDictIndex[0]
        _, selector = top_dict.CharStrings.getItemAndSelector("o")
        assert selector is None

        _update_cff2_glyph(glyph, None, reader.font)

        stored = top_dict.CharStrings["o"]
        assert stored.private is top_dict.FDArray[0].Private


def test_update_glyph_rewrites_cff2_charstring(cff2_path: Path, tmp_path: Path) -> None:
    """FontWriter.update_glyph stores new CFF2 outlines that survive save and reopen."""
    output = tmp_path / "out.otf"
    with TTFont(cff2_path) as original:
        original_program = cff2_program(original, "o")

    with FontReader(cff2_path) as reader:
        glyph = reader.get_glyph("o")
        assert glyph is not None
        moved = _shifted(glyph)
        writer = FontWriter(reader.font, output)
        writer.update_glyph(moved)
        writer.save()

    saved = TTFont(output)
    assert "CFF2" in saved
    assert "CFF " not in saved
    assert cff2_program(saved, "o") != original_program
    assert saved["hmtx"]["o"][0] == glyph.metadata.advance_width
    saved.close()

    with FontReader(output) as reread:
        result = reread.get_glyph("o")
    assert result is not None
    assert [contour.points for contour in result.contours] == [
        contour.points for contour in moved.contours
    ]
