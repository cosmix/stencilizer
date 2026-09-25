"""Tests for the window's top bar."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy
from pytestqt.qtbot import QtBot

from stencilizer.gui.header import HeaderBar


@pytest.fixture
def header(qtbot: QtBot) -> HeaderBar:
    """Create a header bar registered for automatic Qt cleanup."""
    bar = HeaderBar()
    qtbot.addWidget(bar)
    return bar


def test_header_defaults(header: HeaderBar) -> None:
    """A new header presents its open action but cannot yet save."""
    assert not header.save_button.isEnabled()
    assert header.open_button.isEnabled()
    assert header.font_name_label.text() == "No font loaded"


def test_set_font_info_updates_labels(header: HeaderBar) -> None:
    """Font metadata is shown in the header's descriptive labels."""
    header.set_font_info("Roboto-Regular.ttf", "TrueType · 2048 UPM")

    assert header.font_name_label.text() == "Roboto-Regular.ttf"
    assert header.font_details_label.text() == "TrueType · 2048 UPM"


def test_busy_state_restores_save_availability(header: HeaderBar) -> None:
    """Busy state disables actions and restores save based on font availability."""
    header.set_font_loaded(True)
    header.set_busy(True)

    assert not header.open_button.isEnabled()
    assert not header.save_button.isEnabled()

    header.set_busy(False)

    assert header.open_button.isEnabled()
    assert header.save_button.isEnabled()

    header.set_font_loaded(False)
    header.set_busy(False)

    assert not header.save_button.isEnabled()


def test_open_and_save_buttons_emit_requests(qtbot: QtBot, header: HeaderBar) -> None:
    """Clicking the available action buttons emits their request signals."""
    open_calls: list[None] = []
    save_calls: list[None] = []
    header.open_requested.connect(lambda: open_calls.append(None))
    header.save_requested.connect(lambda: save_calls.append(None))
    header.set_font_loaded(True)
    header.show()

    qtbot.mouseClick(header.open_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    qtbot.mouseClick(header.save_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]

    assert len(open_calls) == 1
    assert len(save_calls) == 1


def test_action_buttons_do_not_expand(header: HeaderBar) -> None:
    """Action buttons retain their size hint rather than consuming free width."""
    assert header.open_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Maximum
    assert header.save_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Maximum
