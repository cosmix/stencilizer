"""Contract: the GUI session opens static CFF2 fonts (stage cff2-static)."""

from pathlib import Path

from stencilizer.core import FontProcessor
from stencilizer.gui.session import FontSession


def test_gui_session_opens_cff2(cff2_font_path: Path, processor: FontProcessor) -> None:
    """Opening a static CFF2 font succeeds and lists 'O' as an island glyph."""
    session = FontSession.open(cff2_font_path, processor)

    assert "O" in {glyph.name for glyph in session.island_glyphs}
    assert session.glyph("O") is not None
