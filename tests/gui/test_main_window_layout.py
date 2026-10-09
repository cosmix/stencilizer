"""Tests for the window's header, empty state and status-bar progress."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from stencilizer.gui.controller import GuiController
from stencilizer.gui.main_window import MainWindow
from stencilizer.utils import ProcessingStats

LOAD_TIMEOUT = 30_000


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


def test_progress_starts_hidden(window: MainWindow) -> None:
    """A fresh window shows neither the progress bar nor its percentage label."""
    assert not window.progress_bar.isVisibleTo(window)
    assert not window.progress_label.isVisibleTo(window)


def test_progress_percentage_sits_beside_the_bar(window: MainWindow) -> None:
    """The bar's own text shows in a status label next to a textless bar, hidden on reset."""
    window.set_progress(3, 10)

    assert not window.progress_bar.isTextVisible()
    assert window.progress_label.isVisibleTo(window)
    assert window.progress_label.text() == "30%"
    assert window.statusBar().isAncestorOf(window.progress_label)

    window.reset_progress()

    assert not window.progress_label.isVisibleTo(window)
    assert window.progress_label.text() == ""


def test_progress_can_be_shown_and_reset(window: MainWindow) -> None:
    """Showing progress reveals the bar with its bounds, resetting hides it again."""
    window.set_progress(3, 10)

    assert window.progress_bar.isVisibleTo(window)
    assert window.progress_bar.maximum() == 10
    assert window.progress_bar.value() == 3

    window.reset_progress()

    assert not window.progress_bar.isVisibleTo(window)


def test_empty_state_until_font_loads(window: MainWindow, qtbot: QtBot, roboto_path: Path) -> None:
    """The stack shows the empty state before a font loads, then the grid."""
    assert window.grid_stack.currentWidget() is window.empty_state

    _load_font(window, qtbot, roboto_path)

    assert window.grid_stack.currentWidget() is window.grid


def test_sidebar_describes_loaded_font(window: MainWindow, qtbot: QtBot, roboto_path: Path) -> None:
    """Loading Roboto fills the sidebar info panel, names the window and enables save."""
    _load_font(window, qtbot, roboto_path)

    values = [label.text() for label in window.controls.font_info.value_labels]
    assert "Roboto-Regular.ttf" in values
    assert "465" in values
    assert window.windowTitle() == "Roboto-Regular.ttf — Stencilizer"
    assert window.header.save_button.isEnabled()


def test_progress_lives_in_status_bar(window: MainWindow) -> None:
    """Save progress shows in the status bar's permanent widget and hides on finish."""
    window.controller.save_progress.emit(3, 10)

    assert window.progress_bar.isVisibleTo(window)
    assert window.progress_bar.value() == 3
    assert window.progress_bar.maximum() == 10
    assert window.statusBar().isAncestorOf(window.progress_bar)

    window.controller.save_finished.emit(ProcessingStats())

    assert not window.progress_bar.isVisibleTo(window)


def test_header_open_button_opens_dialog(
    window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking the header's open button raises the file chooser dialog."""
    calls: list[tuple[object, ...]] = []

    def record_dialog(*args: object) -> tuple[str, str]:
        """Record a getOpenFileName call and decline to choose a file."""
        calls.append(args)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", record_dialog)
    window.show()

    qtbot.mouseClick(window.header.open_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]

    assert len(calls) == 1
    assert window.controller.session is None
