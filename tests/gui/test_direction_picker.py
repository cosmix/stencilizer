"""Tests for per-glyph bridge direction selection."""

import pytest
from pytestqt.qtbot import QtBot

from stencilizer.config.settings import BridgeDirection
from stencilizer.gui.direction_picker import DirectionPicker


def test_picker_emits_only_on_user_change(picker: DirectionPicker) -> None:
    """Programmatic selection is silent while a user change emits its value."""
    chosen: list[str] = []
    picker.direction_chosen.connect(chosen.append)

    picker.show_for("O", ("O",), BridgeDirection.VERTICAL)

    assert picker.combo.currentData() == BridgeDirection.VERTICAL.value
    assert picker.combo.isEnabled()
    assert chosen == []

    picker.combo.setCurrentIndex(picker.combo.findData(BridgeDirection.HORIZONTAL.value))

    assert chosen == [BridgeDirection.HORIZONTAL.value]


def test_picker_disables_for_composites(picker: DirectionPicker) -> None:
    """Composite and empty glyphs show their source status and cannot choose."""
    chosen: list[str] = []
    picker.direction_chosen.connect(chosen.append)

    picker.show_for("Aring", ("A", "ring"), BridgeDirection.AUTO)

    assert not picker.combo.isEnabled()
    assert picker.source_label.text() == "Follows A, ring"

    picker.show_for("space", (), BridgeDirection.AUTO)

    assert not picker.combo.isEnabled()
    assert picker.source_label.text() == "No islands to bridge"

    picker.show_for("O", ("O",), BridgeDirection.VERTICAL)
    picker.clear()

    assert not picker.combo.isEnabled()
    assert picker.combo.currentData() == BridgeDirection.AUTO.value
    assert picker.source_label.text() == ""
    assert chosen == []


@pytest.fixture
def picker(qtbot: QtBot) -> DirectionPicker:
    """Create a picker owned by pytest-qt."""
    direction_picker = DirectionPicker()
    qtbot.addWidget(direction_picker)
    return direction_picker
