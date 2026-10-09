"""Tests for the window's top bar."""

import xml.etree.ElementTree as ET
from importlib.resources import files

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy
from pytestqt.qtbot import QtBot

from stencilizer.gui import assets
from stencilizer.gui.header import WORDMARK_FILE, WORDMARK_HEIGHT, HeaderBar, Wordmark

SVG_NS = "{http://www.w3.org/2000/svg}"


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


def test_wordmark_leads_the_header(header: HeaderBar) -> None:
    """The wordmark is the header's first item, sized to the header height."""
    layout = header.layout()
    assert isinstance(layout, QHBoxLayout)
    first = layout.itemAt(0)
    assert first is not None
    assert first.widget() is header.logo
    assert isinstance(header.logo, Wordmark)
    assert header.logo.height() == WORDMARK_HEIGHT
    assert header.logo.width() > 4 * WORDMARK_HEIGHT
    assert header.logo.accessibleName() == "Stencilizer"


def test_wordmark_resource_is_a_valid_outline_svg() -> None:
    """The packaged SVG is one currentColor path with a viewBox and no text or raster."""
    raw = files(assets).joinpath(WORDMARK_FILE).read_bytes()
    root = ET.fromstring(raw)

    assert root.tag == f"{SVG_NS}svg"
    assert len(root.get("viewBox", "").split()) == 4
    paths = root.findall(f"{SVG_NS}path")
    assert len(paths) == 1
    assert paths[0].get("fill") == "currentColor"
    assert paths[0].get("d", "").startswith("M")
    assert root.findall(f".//{SVG_NS}text") == []
    assert root.findall(f".//{SVG_NS}image") == []
    assert QSvgRenderer(raw).isValid()


def test_wordmark_paints_in_the_palette_text_colour(qtbot: QtBot) -> None:
    """A palette change recolours the mark, so it follows the theme."""
    mark = Wordmark()
    qtbot.addWidget(mark)
    palette = mark.palette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ff0000"))
    mark.setPalette(palette)

    image = mark.grab().toImage()
    red = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).red() > 200 and image.pixelColor(x, y).green() < 60
    ]

    assert red
