"""Contracts for static CFF2 read and write (stage cff2-static)."""

from pathlib import Path

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import GlyphAnalyzer
from stencilizer.core.processor import FontProcessor
from stencilizer.io import FontReader
from tests.font_helpers import COMMIT_MONO, write_commit_mono_cff2


def test_cff2_read_normalizes_winding(tmp_path: Path) -> None:
    """CFF2 'O' reads with TrueType winding, so the analyzer finds the same island as CFF."""
    cff2_path = write_commit_mono_cff2(tmp_path / "converted.otf")
    analyzer = GlyphAnalyzer()

    with FontReader(COMMIT_MONO) as cff_reader:
        cff_glyph = cff_reader.get_glyph("O")
        assert cff_glyph is not None
        cff_islands = analyzer.analyze(cff_glyph, cff_reader.units_per_em).islands

    reader = FontReader(cff2_path)
    reader.load()
    assert "CFF2" in reader.font
    glyph = reader.get_glyph("O")
    assert glyph is not None
    hierarchy = analyzer.analyze(glyph, reader.units_per_em)

    assert len(hierarchy.islands) == 1
    assert hierarchy.islands == cff_islands


def test_cff2_static_roundtrip_writes_bridges(tmp_path: Path) -> None:
    """Processing a CFF2 font writes a CFF2 font whose 'O' is bridged."""
    input_path = write_commit_mono_cff2(tmp_path / "converted.otf")
    output_path = tmp_path / "out.otf"
    settings = StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "run.log"))

    stats = FontProcessor(settings).process(font_path=input_path, output_path=output_path)

    assert stats.error_count == 0
    assert stats.bridges_added >= 1
    output = TTFont(output_path)
    assert "CFF2" in output
    assert "CFF " not in output
    source = TTFont(input_path)
    output_bytes = output["CFF2"].cff.topDictIndex[0].CharStrings["O"]
    source_bytes = source["CFF2"].cff.topDictIndex[0].CharStrings["O"]
    output_bytes.compile(isCFF2=True)
    source_bytes.compile(isCFF2=True)
    assert output_bytes.bytecode != source_bytes.bytecode

    analyzer = GlyphAnalyzer()
    with FontReader(output_path) as reader:
        glyph = reader.get_glyph("O")
        assert glyph is not None
        assert analyzer.analyze(glyph, reader.units_per_em).islands == []
        assert len(glyph.contours) >= 2
