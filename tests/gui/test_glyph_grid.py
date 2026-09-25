"""Tests for the glyph thumbnail grid."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QListWidgetItem
from pytestqt.qtbot import QtBot

from stencilizer.domain import Glyph
from stencilizer.gui.glyph_grid import THUMBNAIL_SIZE, GlyphGrid
from stencilizer.io import FontReader


def _load_glyphs(font_path: Path, names: list[str]) -> list[Glyph]:
    """Load named glyphs from a font in the requested order."""
    with FontReader(font_path) as reader:
        glyphs = [reader.get_glyph(name) for name in names]
    assert all(glyph is not None for glyph in glyphs)
    return [glyph for glyph in glyphs if glyph is not None]


def _grid_items(grid: GlyphGrid) -> list[QListWidgetItem]:
    """Return every grid item after checking the indexed lookups."""
    items: list[QListWidgetItem] = []
    for index in range(grid.count()):
        item = grid.item(index)
        assert item is not None
        items.append(item)
    return items


def test_set_glyphs_renders_glyphs_in_input_order(qtbot: QtBot, roboto_path: Path) -> None:
    """The grid displays ordered glyph names, icons, and Unicode tooltips."""
    glyphs = _load_glyphs(roboto_path, ["O", "B", "eight"])
    grid = GlyphGrid()
    qtbot.addWidget(grid)

    grid.set_glyphs(glyphs, 2146, -555)

    assert grid.count() == 3
    items = _grid_items(grid)
    assert [item.text() for item in items] == ["O", "B", "eight"]
    assert all(not item.icon().isNull() for item in items)
    assert "U+004F" in items[0].toolTip()


def test_select_glyph_emits_selected_name(qtbot: QtBot, roboto_path: Path) -> None:
    """Programmatic selection emits the selected glyph name."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    selected: list[str] = []
    grid.glyph_selected.connect(selected.append)

    assert grid.select_glyph("B")
    assert selected == ["B"]


def test_select_missing_glyph_returns_false_without_emission(
    qtbot: QtBot, roboto_path: Path
) -> None:
    """A name absent from the grid does not emit a selection."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    selected: list[str] = []
    grid.glyph_selected.connect(selected.append)

    assert not grid.select_glyph("missing")
    assert selected == []


def test_set_glyphs_replaces_contents(qtbot: QtBot, roboto_path: Path) -> None:
    """A later font selection replaces all earlier grid items."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)

    grid.set_glyphs(_load_glyphs(roboto_path, ["B"]), 2146, -555)

    assert grid.count() == 1
    assert _grid_items(grid)[0].text() == "B"


def test_clear_after_selection_emits_no_signal(qtbot: QtBot, roboto_path: Path) -> None:
    """Clearing a populated, selected grid emits no selection and does not raise.

    The middle glyph is selected (rather than the first) so removing it does not shift an
    adjacent item into the current position first: clearing goes straight to no current item.
    """
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    assert grid.select_glyph("B")
    selected: list[str] = []
    grid.glyph_selected.connect(selected.append)

    grid.clear()

    assert grid.count() == 0
    assert selected == []


def test_keyboard_navigation_emits_selection(qtbot: QtBot, roboto_path: Path) -> None:
    """The right arrow moves from O to B and emits the new glyph name."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.resize(480, 160)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B", "eight"]), 2146, -555)
    assert grid.select_glyph("O")
    selected: list[str] = []
    grid.glyph_selected.connect(selected.append)
    grid.show()
    grid.setFocus()

    qtbot.keyClick(grid, Qt.Key.Key_Right)  # type: ignore[no-untyped-call]

    assert selected == ["B"]


def test_thumbnails_rerender_when_palette_changes(qtbot: QtBot, roboto_path: Path) -> None:
    """Palette changes redraw thumbnails with the new background colour."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B"]), 1900, -500)
    palette = grid.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#121418"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))

    grid.setPalette(palette)

    item = grid.item(0)
    assert item is not None
    image = item.icon().pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE).toImage()
    assert image.pixelColor(0, 0) == QColor("#121418")


def test_unbridged_mark_is_light_red_on_dark_base(qtbot: QtBot, roboto_path: Path) -> None:
    """Unbridged glyphs use a visible red against a dark background."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B"]), 1900, -500)
    palette = grid.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#121418"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))
    grid.setPalette(palette)

    grid.set_unbridged(["O"])

    item = grid.item(0)
    assert item is not None
    assert item.data(Qt.ItemDataRole.ForegroundRole) == QColor("#ff8a80")


def test_unbridged_mark_recolours_on_palette_switch(qtbot: QtBot, roboto_path: Path) -> None:
    """Palette changes update marked glyphs without styling unmarked ones."""
    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.set_glyphs(_load_glyphs(roboto_path, ["O", "B"]), 1900, -500)
    grid.set_unbridged(["O"])
    marked_item = grid.item(0)
    assert marked_item is not None
    assert marked_item.data(Qt.ItemDataRole.ForegroundRole) == QColor("#c62828")
    palette = grid.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#121418"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))

    grid.setPalette(palette)

    assert marked_item.data(Qt.ItemDataRole.ForegroundRole) == QColor("#ff8a80")
    unmarked_item = grid.item(1)
    assert unmarked_item is not None
    assert unmarked_item.data(Qt.ItemDataRole.ForegroundRole) is None
