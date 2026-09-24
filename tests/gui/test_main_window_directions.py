"""Tests for per-glyph bridge direction in the top-level GUI window."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem
from pytestqt.qtbot import QtBot

from stencilizer.config.settings import BridgeDirection
from stencilizer.gui.controller import GuiController
from stencilizer.gui.glyph_grid import UNBRIDGED_ROLE
from stencilizer.gui.main_window import MainWindow
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats

LOAD_TIMEOUT = 30_000
SAVE_TIMEOUT = 120_000


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot) -> Iterator[MainWindow]:
    """Create a window backed by a controller that is shut down after each test."""
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _load_font(window: MainWindow, qtbot: QtBot, path: Path) -> None:
    """Load ``path`` and wait until the automatic initial preview is visible."""
    with qtbot.waitSignal(window.controller.font_loaded, timeout=LOAD_TIMEOUT):
        window.load_font(path)
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)


def _grid_item(window: MainWindow, name: str) -> QListWidgetItem:
    """Return the grid item whose stored glyph name matches ``name``."""
    for index in range(window.grid.count()):
        item = window.grid.item(index)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
            return item
    raise AssertionError(f"no grid item named {name!r}")


def _choose_direction(window: MainWindow, direction: BridgeDirection) -> None:
    """Select a bridge direction in the direction picker's combo box."""
    combo = window.direction_picker.combo
    index = combo.findData(direction.value)
    assert index >= 0
    combo.setCurrentIndex(index)


def _spans(bbox: tuple[float, float, float, float], value: float) -> bool:
    """Return whether an (min_x, min_y, max_x, max_y) bbox spans a coordinate on the y axis."""
    return bbox[1] < value < bbox[3]


def test_grid_lists_bridged_composites(window: MainWindow, qtbot: QtBot, roboto_path: Path) -> None:
    """The grid displays every island and composite glyph, and the info label counts them."""
    _load_font(window, qtbot, roboto_path)

    names = {
        window.grid.item(index).data(Qt.ItemDataRole.UserRole)
        for index in range(window.grid.count())
    }
    assert window.grid.count() == 1027
    assert "Aacute" in names
    assert "465 composites" in window.controls.font_info_label.text()


def test_choosing_direction_updates_preview_and_marker(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """Choosing a direction for the selected glyph updates its preview, marker and state."""
    _load_font(window, qtbot, roboto_path)
    assert window.grid.select_glyph("O")

    _choose_direction(window, BridgeDirection.HORIZONTAL)

    glyph = window.comparison.after_canvas.glyph
    assert glyph is not None
    assert not any(_spans(c.bounding_box(), 728) for c in glyph.contours)
    assert _grid_item(window, "O").text() == "O ↔"
    assert window.controller.direction_for("O") is BridgeDirection.HORIZONTAL


def test_composite_selection_shows_following_picker(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """Selecting a composite whose base has a direction shows a disabled, following picker."""
    _load_font(window, qtbot, roboto_path)
    assert window.grid.select_glyph("A")
    _choose_direction(window, BridgeDirection.VERTICAL)

    assert window.grid.select_glyph("Aacute")

    assert not window.direction_picker.combo.isEnabled()
    assert window.direction_picker.combo.currentData() == BridgeDirection.VERTICAL.value
    assert window.direction_picker.source_label.text() == "Follows A"


def test_unbridged_glyphs_marked_in_grid(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """Glyphs the survey reports as unbridged are marked in the grid; others are not."""
    with qtbot.waitSignal(window.controller.unbridged_changed, timeout=10_000):
        _load_font(window, qtbot, roboto_path)

    assert _grid_item(window, "four").data(UNBRIDGED_ROLE) is True
    assert _grid_item(window, "O").data(UNBRIDGED_ROLE) is False


def test_saved_font_uses_chosen_direction(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """A save writes the direction chosen in the window, and composites keep referencing it."""
    _load_font(window, qtbot, roboto_path)
    window.controls.workers_spin.setValue(1)
    assert window.grid.select_glyph("O")
    _choose_direction(window, BridgeDirection.HORIZONTAL)
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font(output_path)

    stats = blocker.args[0]
    assert isinstance(stats, ProcessingStats)
    assert stats.error_count == 0
    with FontReader(output_path) as reader:
        saved_o = reader.get_glyph("O")
        assert saved_o is not None
        assert not any(_spans(c.bounding_box(), 728) for c in saved_o.contours)

        glyph_set = reader.font.getGlyphSet()
        pen = RecordingPen()
        glyph_set["Oacute"].draw(pen)
    assert any(operation == "addComponent" and args[0] == "O" for operation, args in pen.value)
