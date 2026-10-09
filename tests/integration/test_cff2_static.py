"""Integration tests for processing static CFF2 fonts."""

from pathlib import Path

from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import StencilizerSettings
from stencilizer.core import GlyphAnalyzer
from stencilizer.core.processor import FontProcessor
from stencilizer.io import FontReader


def _cff2_font(tmp_path: Path) -> Path:
    """Convert the CommitMono fixture to a static CFF2 font."""
    source = Path(__file__).parent.parent / "fixtures" / "CommitMono-Cosmix-700-Regular.otf"
    font = TTFont(source)
    convertCFFToCFF2(font)
    path = tmp_path / "commitmono-cff2.otf"
    font.save(path)
    font.close()
    return path


def test_cff2_process_keeps_cff2_table(tmp_path: Path) -> None:
    """Processing a static CFF2 font preserves its CFF2 outlines."""
    source = _cff2_font(tmp_path)
    output = tmp_path / "out.otf"

    stats = FontProcessor(StencilizerSettings()).process(
        font_path=source,
        output_path=output,
    )

    output_font = TTFont(output)
    assert "CFF2" in output_font
    assert "CFF " not in output_font
    assert stats.bridges_added > 0
    output_font.close()


def test_cff2_output_glyphs_have_no_islands(tmp_path: Path) -> None:
    """Processing bridges the islands in representative CFF2 glyphs."""
    source = _cff2_font(tmp_path)
    output = tmp_path / "out.otf"
    FontProcessor(StencilizerSettings()).process(font_path=source, output_path=output)

    with FontReader(output) as reader:
        analyzer = GlyphAnalyzer()
        upm = reader.units_per_em

        for glyph_name in ("O", "zero", "B"):
            glyph = reader.get_glyph(glyph_name)
            assert glyph is not None
            assert analyzer.analyze(glyph, upm).get_islands() == []


def test_cff2_untouched_glyph_charstring_unchanged(tmp_path: Path) -> None:
    """Glyphs without islands retain their original CFF2 charstrings."""
    source = _cff2_font(tmp_path)
    output = tmp_path / "out.otf"
    FontProcessor(StencilizerSettings()).process(font_path=source, output_path=output)

    input_font = TTFont(source)
    output_font = TTFont(output)
    input_charstring = input_font["CFF2"].cff.topDictIndex[0].CharStrings["l"]
    output_charstring = output_font["CFF2"].cff.topDictIndex[0].CharStrings["l"]
    input_charstring.decompile()
    output_charstring.decompile()

    assert output_charstring.program == input_charstring.program

    input_font.close()
    output_font.close()
