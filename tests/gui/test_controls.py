"""Tests for the GUI control panel."""

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.gui.controls import ControlPanel


def test_control_panel_defaults(panel: ControlPanel) -> None:
    """The panel starts with default parameters and no loaded font."""
    assert panel.bridge_config() == BridgeConfig()
    assert panel.max_workers() is None
    assert not panel.save_button.isEnabled()
    assert not panel.progress_bar.isVisibleTo(panel)


def test_width_controls_have_spec_range(panel: ControlPanel) -> None:
    """The width slider and spin box both span the 30-110 percent spec range."""
    assert panel.width_slider.minimum() == 30
    assert panel.width_slider.maximum() == 110
    assert panel.width_spin.minimum() == 30
    assert panel.width_spin.maximum() == 110


def test_set_font_info_updates_label(panel: ControlPanel) -> None:
    """Font information is displayed in the wrapped label."""
    panel.set_font_info("Roboto Regular — 3387 glyphs")

    assert panel.font_info_label.text() == "Roboto Regular — 3387 glyphs"


def test_width_controls_sync_and_emit_once(panel: ControlPanel) -> None:
    """Changing either width control updates its pair and emits once."""
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))

    panel.width_spin.setValue(80)

    assert panel.width_slider.value() == 80
    assert len(changes) == 1

    changes.clear()
    panel.width_slider.setValue(40)

    assert panel.width_spin.value() == 40
    assert len(changes) == 1
    assert panel.bridge_config().width_percent == 40.0


def test_spanning_toggle_changes_bridge_config(panel: ControlPanel) -> None:
    """Toggling spanning bridges emits one parameter change."""
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))

    panel.spanning_check.setChecked(False)

    assert len(changes) == 1
    assert panel.bridge_config().use_spanning_bridges is False


def test_worker_limit_does_not_emit_parameter_change(panel: ControlPanel) -> None:
    """The worker limit is independent of bridge parameter updates."""
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))

    panel.workers_spin.setValue(2)

    assert panel.max_workers() == 2
    assert not changes


def test_busy_state_restores_save_availability(panel: ControlPanel) -> None:
    """Busy state disables actions and restores save based on font availability."""
    panel.set_font_loaded(True)
    panel.set_busy(True)

    assert not panel.open_button.isEnabled()
    assert not panel.save_button.isEnabled()

    panel.set_busy(False)

    assert panel.open_button.isEnabled()
    assert panel.save_button.isEnabled()

    panel.set_font_loaded(False)
    panel.set_busy(False)

    assert not panel.save_button.isEnabled()


def test_open_and_save_buttons_emit_requests(qtbot: QtBot, panel: ControlPanel) -> None:
    """Clicking the available action buttons emits their request signals."""
    open_calls: list[None] = []
    save_calls: list[None] = []
    panel.open_requested.connect(lambda: open_calls.append(None))
    panel.save_requested.connect(lambda: save_calls.append(None))
    panel.set_font_loaded(True)
    panel.show()

    qtbot.mouseClick(panel.open_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    qtbot.mouseClick(panel.save_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]

    assert len(open_calls) == 1
    assert len(save_calls) == 1


def test_progress_can_be_shown_and_reset(panel: ControlPanel) -> None:
    """Progress reports its range and value, then reset hides it."""
    panel.set_progress(3, 10)

    assert panel.progress_bar.isVisibleTo(panel)
    assert panel.progress_bar.maximum() == 10
    assert panel.progress_bar.value() == 3

    panel.reset_progress()

    assert not panel.progress_bar.isVisibleTo(panel)


@pytest.fixture
def panel(qtbot: QtBot) -> ControlPanel:
    """Create a control panel owned by pytest-qt."""
    control_panel = ControlPanel()
    qtbot.addWidget(control_panel)
    return control_panel
