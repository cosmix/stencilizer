"""Contracts for the GUI width-scaling controls (stage width-scaling)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeWidthScaling
from stencilizer.gui.controller import GuiController
from stencilizer.gui.controls import ControlPanel
from stencilizer.gui.main_window import MainWindow
from tests.font_helpers import INTER, ROBOTO
from tests.gui.conftest import LOAD_TIMEOUT

pytestmark = pytest.mark.usefixtures("staging_root")


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    """A main window whose controller is shut down after each test.

    Controller errors reach the test through the error signal; the modal warning
    box is replaced so a failed load cannot block the offscreen event loop.
    """
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: None)
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _load_window(window: MainWindow, qtbot: QtBot, path: Path) -> None:
    loaded: list[object] = []
    errors: list[str] = []
    window.controller.font_loaded.connect(loaded.append)
    window.controller.error.connect(errors.append)
    window.load_font(path)
    qtbot.waitUntil(lambda: bool(loaded or errors), timeout=LOAD_TIMEOUT)
    assert not errors, errors
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)


def _select_proportional(panel: ControlPanel) -> None:
    index = panel.scaling_combo.findData(BridgeWidthScaling.PROPORTIONAL)
    assert index >= 0
    panel.scaling_combo.setCurrentIndex(index)


def test_gui_controls_reach_bridge_config(qtbot: QtBot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    _select_proportional(panel)
    panel.strength_spin.setValue(50)
    panel.min_width_spin.setValue(40)
    config = panel.bridge_config()
    assert config.width_scaling is BridgeWidthScaling.PROPORTIONAL
    assert config.scaling_strength == 50.0
    assert config.min_width_percent == 40.0


def test_gui_scaling_shown_for_variable_only(window: MainWindow, qtbot: QtBot) -> None:
    _load_window(window, qtbot, INTER)
    assert window.controls.scaling_box.isVisibleTo(window)
    _select_proportional(window.controls)
    assert window.controls.bridge_config().width_scaling is BridgeWidthScaling.PROPORTIONAL
    _load_window(window, qtbot, ROBOTO)
    assert not window.controls.scaling_box.isVisibleTo(window)
    assert window.controls.bridge_config().width_scaling is BridgeWidthScaling.FIXED
