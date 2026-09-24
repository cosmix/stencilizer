"""Tests for glyph direction and unbridged markers in the GUI."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from pytestqt.qtbot import QtBot

from stencilizer.config.settings import BridgeDirection
from stencilizer.domain import Glyph
from stencilizer.gui.glyph_grid import (
    BASE_TOOLTIP_ROLE,
    UNBRIDGED_ROLE,
    GlyphGrid,
)
from stencilizer.gui.glyph_view import ComparisonView
from stencilizer.gui.session import PreviewResult
from stencilizer.io import FontReader


def _load_glyphs(font_path: Path, names: list[str]) -> list[Glyph]:
    """Load named glyphs from a font in the requested order."""
    with FontReader(font_path) as reader:
        glyphs = [reader.get_glyph(name) for name in names]
    assert all(glyph is not None for glyph in glyphs)
    return [glyph for glyph in glyphs if glyph is not None]


def test_grid_direction_marker(qtbot: QtBot, roboto_path: Path) -> None:
    """Direction markers change only the displayed text of the named glyph."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    item = grid.item(0)
    assert item is not None

    grid.set_direction_marker("O", BridgeDirection.HORIZONTAL)
    assert item.text() == "O ↔"
    assert item.data(Qt.ItemDataRole.UserRole) == "O"

    grid.set_direction_marker("O", BridgeDirection.VERTICAL)
    assert item.text() == "O ↕"
    grid.set_direction_marker("O", BridgeDirection.AUTO)
    assert item.text() == "O"
    grid.set_direction_marker("missing", BridgeDirection.HORIZONTAL)
    assert item.text() == "O"


def test_grid_marks_unbridged_glyphs(qtbot: QtBot, roboto_path: Path) -> None:
    """Unbridged styling is reversible and preserves a glyph's direction marker."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    items = [grid.item(index) for index in range(grid.count())]
    assert all(item is not None for item in items)
    o_item, b_item, eight_item = items
    assert o_item is not None
    assert b_item is not None
    assert eight_item is not None
    base_tooltip = b_item.data(BASE_TOOLTIP_ROLE)

    grid.set_direction_marker("B", BridgeDirection.VERTICAL)
    grid.set_unbridged({"B"})

    assert o_item.data(UNBRIDGED_ROLE) is False
    assert b_item.data(UNBRIDGED_ROLE) is True
    assert eight_item.data(UNBRIDGED_ROLE) is False
    assert b_item.toolTip() == f"{base_tooltip}\nNo bridge could be placed"
    assert b_item.data(Qt.ItemDataRole.ForegroundRole) == QColor("#c62828")

    grid.set_unbridged(set())

    assert b_item.data(UNBRIDGED_ROLE) is False
    assert b_item.toolTip() == base_tooltip
    assert b_item.data(Qt.ItemDataRole.ForegroundRole) is None
    assert b_item.text() == "B ↕"


def test_preview_text_reports_no_bridge(qtbot: QtBot, roboto_path: Path) -> None:
    """A completed preview distinguishes a failed bridge from a bridged island."""
    glyph = _load_glyphs(roboto_path, ["O"])[0]
    view = ComparisonView()
    qtbot.addWidget(view)
    no_bridge = PreviewResult(
        glyph_name="O",
        original=glyph,
        stenciled=glyph,
        bridges_added=0,
        error=None,
        duration_ms=1.2,
    )

    view.show_preview(no_bridge, 2146, -555)

    assert "no bridge could be placed" in view.info_label.text()
    bridged = PreviewResult(
        glyph_name="O",
        original=glyph,
        stenciled=glyph,
        bridges_added=1,
        error=None,
        duration_ms=1.2,
    )
    view.show_preview(bridged, 2146, -555)
    assert "1 island(s) bridged" in view.info_label.text()
