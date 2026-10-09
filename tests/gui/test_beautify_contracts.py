"""Frozen contracts for the header bar, sidebar, workers slider and system-following theme."""

import multiprocessing
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QSlider
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.gui import app
from stencilizer.gui.controller import GuiController
from stencilizer.gui.main_window import MainWindow
from stencilizer.io import FontReader


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot) -> Iterator[MainWindow]:
    """Create a window backed by a controller that is shut down after each test."""
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _relative_luminance(color: str) -> float:
    """WCAG 2 relative luminance of a "#rrggbb" colour."""
    qcolor = QColor(color)
    assert qcolor.isValid(), color

    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linear(qcolor.red())
        + 0.7152 * linear(qcolor.green())
        + 0.0722 * linear(qcolor.blue())
    )


def _contrast(first: str, second: str) -> float:
    """WCAG 2 contrast ratio between two "#rrggbb" colours."""
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_theme_colors_are_legible() -> None:
    """Both themes meet WCAG AAA for body text and AA for secondary and accent text."""
    from stencilizer.gui.theme import DARK, LIGHT

    for colors in (LIGHT, DARK):
        for background in (colors.base, colors.surface, colors.window):
            assert _contrast(colors.text, background) >= 7.0, (colors, background)
        for background in (colors.surface, colors.window):
            assert _contrast(colors.muted_text, background) >= 4.5, (colors, background)
        assert _contrast(colors.accent_text, colors.accent) >= 4.5, colors


def test_theme_follows_system_scheme_changes(qapp: QApplication) -> None:
    """A later system light/dark switch restyles the running application."""
    from stencilizer.gui.theme import DARK, LIGHT, apply_theme

    saved_palette = QPalette(qapp.palette())
    saved_stylesheet = qapp.styleSheet()
    try:
        apply_theme(qapp)

        qapp.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        dark_window = qapp.palette().color(QPalette.ColorRole.Window)
        assert dark_window == QColor(DARK.window)
        assert dark_window.lightness() < 128
        assert qapp.styleSheet() != ""

        qapp.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Light)
        light_window = qapp.palette().color(QPalette.ColorRole.Window)
        assert light_window == QColor(LIGHT.window)
        assert light_window.lightness() >= 128
    finally:
        qapp.setPalette(saved_palette)
        qapp.setStyleSheet(saved_stylesheet)


def test_main_applies_theme_before_showing_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main themes the application after creating it and before the window appears."""
    events: list[str] = []
    applications: list[object] = []
    theme_calls: list[tuple[object, object]] = []

    class ApplicationStub:
        """Record QApplication construction without starting Qt."""

        def __init__(self, _arguments: list[str]) -> None:
            """Record construction."""
            events.append("application")
            applications.append(self)

        def exec(self) -> int:
            """Return a successful application exit code."""
            return 0

    class WindowStub:
        """Record that the main window was displayed."""

        def show(self) -> None:
            """Record the show request."""
            events.append("show")

    def record_theme(application: object, scheme: object = None) -> None:
        """Record the themed application and the requested scheme."""
        events.append("theme")
        theme_calls.append((application, scheme))

    def create_window(_font: Path | None, _log_file: Path) -> WindowStub:
        """Return a displayable stub."""
        return WindowStub()

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_kwargs: "spawn")
    monkeypatch.setattr(app, "QApplication", ApplicationStub)
    monkeypatch.setattr(app, "apply_theme", record_theme)
    monkeypatch.setattr(app, "create_window", create_window)

    assert app.main(["x.ttf"]) == 0

    assert events == ["application", "theme", "show"]
    assert len(theme_calls) == 1
    assert theme_calls[0][0] is applications[0]
    assert theme_calls[0][1] is None


def test_font_info_shows_font_strings_as_plain_text(qtbot: QtBot) -> None:
    """Markup in font-derived strings is shown literally, never rendered."""
    from PySide6.QtWidgets import QLabel

    from stencilizer.gui.font_info import FontInfo, InfoSection
    from stencilizer.gui.font_info_panel import FontInfoPanel

    panel = FontInfoPanel()
    qtbot.addWidget(panel)

    panel.set_info(FontInfo("<b>x</b>", (InfoSection("Font", (("Family", "<i>y</i>"),)),)))

    labels = panel.findChildren(QLabel)
    assert labels
    assert all(label.textFormat() == Qt.TextFormat.PlainText for label in labels)
    assert {label.text() for label in labels} >= {"<b>x</b>", "<i>y</i>"}


def test_action_buttons_stay_compact_in_top_bar(window: MainWindow, qtbot: QtBot) -> None:
    """Open and Save keep their natural width in a bar above the sidebar, Save on the right."""
    window.resize(1280, 800)
    with qtbot.waitExposed(window):
        window.show()

    header = window.header
    for button in (header.open_button, header.save_button):
        assert button.width() <= 1.5 * button.sizeHint().width(), button.text()

    header_bottom = header.mapTo(window, QPoint(0, 0)).y() + header.height()
    sidebar_top = window.controls.mapTo(window, QPoint(0, 0)).y()
    assert header_bottom <= sidebar_top

    assert header.save_button.mapTo(window, QPoint(0, 0)).x() > window.width() / 2


def test_workers_slider_reaches_controller(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the workers slider updates the controller's worker count, 0 meaning Auto."""
    recorded: list[int | None] = []

    def record(_bridge: BridgeConfig, max_workers: int | None) -> None:
        """Record the worker count the window passes on."""
        recorded.append(max_workers)

    monkeypatch.setattr(window.controller, "set_parameters", record)
    slider = window.controls.workers_slider
    assert isinstance(slider, QSlider)
    maximum = os.cpu_count() or 1
    assert slider.maximum() == maximum

    slider.setValue(maximum)
    assert recorded
    assert recorded[-1] == maximum

    recorded.clear()
    slider.setValue(0)
    assert recorded
    assert recorded[-1] is None


def test_grid_thumbnails_rerender_on_palette_change(qtbot: QtBot, roboto_path: Path) -> None:
    """Thumbnails drawn under the light palette are redrawn on the dark base after a switch."""
    from stencilizer.gui.glyph_grid import THUMBNAIL_SIZE, GlyphGrid
    from stencilizer.gui.theme import DARK, LIGHT, palette_for

    assert QColor(DARK.base) != QColor(LIGHT.base)
    with FontReader(roboto_path) as reader:
        glyph = reader.get_glyph("O")
    assert glyph is not None

    grid = GlyphGrid()
    qtbot.addWidget(grid)
    grid.setPalette(palette_for(LIGHT))
    grid.set_glyphs([glyph], 1900, -500)
    grid.setPalette(palette_for(DARK))

    item = grid.item(0)
    assert item is not None
    image = item.icon().pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE).toImage()
    assert image.pixelColor(0, 0) == QColor(DARK.base)
