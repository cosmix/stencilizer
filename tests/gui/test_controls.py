"""Tests for the GUI control panel."""

import os

import pytest
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.gui.controls import ControlPanel


def test_control_panel_defaults(panel: ControlPanel) -> None:
    """The panel starts with default parameters and automatic workers."""
    assert panel.bridge_config() == BridgeConfig()
    assert panel.workers_slider.value() == 0
    assert panel.workers_value_label.text() == "Auto"
    assert panel.max_workers() is None


def test_width_controls_have_spec_range(panel: ControlPanel) -> None:
    """The width slider and spin box both span the 30-110 percent spec range."""
    assert panel.width_slider.minimum() == 30
    assert panel.width_slider.maximum() == 110
    assert panel.width_spin.minimum() == 30
    assert panel.width_spin.maximum() == 110


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

    panel.workers_slider.setValue(2)

    assert panel.max_workers() == 2
    assert not changes


def test_workers_label_follows_slider(panel: ControlPanel) -> None:
    """The worker label and limit follow the slider's value."""
    top = panel.workers_slider.maximum()

    assert top == (os.cpu_count() or 1)

    panel.workers_slider.setValue(top)

    assert panel.workers_value_label.text() == str(top)
    assert panel.max_workers() == top

    panel.workers_slider.setValue(0)

    assert panel.workers_value_label.text() == "Auto"
    assert panel.max_workers() is None


@pytest.fixture
def panel(qtbot: QtBot) -> ControlPanel:
    """Create a control panel owned by pytest-qt."""
    control_panel = ControlPanel()
    qtbot.addWidget(control_panel)
    return control_panel
