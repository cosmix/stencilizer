"""Tests for the light and dark palettes, the stylesheet and system-scheme following."""

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from stencilizer.gui import theme
from stencilizer.gui.theme import (
    DARK,
    LIGHT,
    ThemeColors,
    apply_theme,
    colors_for,
    palette_for,
    stylesheet_for,
)

ROLE_FIELDS = (
    (QPalette.ColorRole.Window, "window"),
    (QPalette.ColorRole.WindowText, "text"),
    (QPalette.ColorRole.Base, "base"),
    (QPalette.ColorRole.AlternateBase, "surface"),
    (QPalette.ColorRole.Text, "text"),
    (QPalette.ColorRole.Button, "surface"),
    (QPalette.ColorRole.ButtonText, "text"),
    (QPalette.ColorRole.Highlight, "accent"),
    (QPalette.ColorRole.HighlightedText, "accent_text"),
    (QPalette.ColorRole.PlaceholderText, "muted_text"),
    (QPalette.ColorRole.Mid, "border"),
    (QPalette.ColorRole.ToolTipBase, "surface"),
    (QPalette.ColorRole.ToolTipText, "text"),
    (QPalette.ColorRole.Link, "accent"),
)

HOOK_SELECTORS = (
    "#headerBar",
    "#appTitle",
    "#fontName",
    "#fontDetails",
    'QPushButton[role="secondary"]',
    'QPushButton[role="primary"]',
    "#sidebar",
    'QLabel[role="sectionTitle"]',
    'QFrame[role="card"]',
    'QLabel[role="value"]',
    'QLabel[role="hint"]',
    'QLabel[role="status"]',
    "QListWidget#glyphGrid",
    "#emptyState",
    "#previewPane",
    "QProgressBar#saveProgress",
    "QSplitter::handle",
    "QStatusBar",
    "QSlider::groove",
    "QSlider::sub-page",
    "QSlider::handle",
    "QCheckBox::indicator",
    "QSpinBox",
    "QComboBox",
    "QToolTip",
    "QScrollBar:vertical",
    "QListWidget#glyphGrid::item:hover",
    "QListWidget#glyphGrid::item:selected",
)

THEMES = pytest.mark.parametrize("colors", [LIGHT, DARK], ids=["light", "dark"])


@pytest.fixture
def themed_app(qapp: QApplication) -> Iterator[QApplication]:
    """Hand out the application and restore its palette and stylesheet afterwards."""
    palette = QPalette(qapp.palette())
    stylesheet = qapp.styleSheet()
    yield qapp
    qapp.setPalette(palette)
    qapp.setStyleSheet(stylesheet)


def _rule_body(stylesheet: str, selector: str) -> str:
    """Return the declarations of the first rule whose selector list is exactly ``selector``."""
    start = stylesheet.index(f"\n{selector} {{") + len(selector) + 3
    return stylesheet[start : stylesheet.index("}", start)]


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        (Qt.ColorScheme.Unknown, LIGHT),
        (Qt.ColorScheme.Light, LIGHT),
        (Qt.ColorScheme.Dark, DARK),
    ],
)
def test_colors_for_picks_dark_only_for_a_dark_scheme(
    scheme: Qt.ColorScheme, expected: ThemeColors
) -> None:
    """Unknown and Light schemes use the light tokens; only Dark uses the dark ones."""
    assert colors_for(scheme) is expected


@THEMES
def test_palette_for_maps_tokens_onto_active_and_inactive_roles(colors: ThemeColors) -> None:
    """Every mapped role carries its token in both the Active and Inactive groups."""
    palette = palette_for(colors)

    expected = {role: QColor(getattr(colors, field)) for role, field in ROLE_FIELDS}
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        assert {role: palette.color(group, role) for role in expected} == expected, group


@THEMES
def test_palette_for_mutes_disabled_text(colors: ThemeColors) -> None:
    """Disabled text, button text and window text use the muted colour."""
    palette = palette_for(colors)

    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        assert palette.color(QPalette.ColorGroup.Disabled, role) == QColor(colors.muted_text), role


@THEMES
def test_stylesheet_for_names_every_hook(colors: ThemeColors) -> None:
    """The stylesheet has a rule for every styling hook the layout exposes."""
    stylesheet = stylesheet_for(colors)

    missing = [selector for selector in HOOK_SELECTORS if selector not in stylesheet]
    assert missing == []


@THEMES
def test_stylesheet_for_paints_grid_and_empty_state_on_base(colors: ThemeColors) -> None:
    """The grid and the empty state use the base colour, which the grid renders thumbnails on."""
    stylesheet = stylesheet_for(colors)

    assert f"background: {colors.base};" in _rule_body(stylesheet, "QListWidget#glyphGrid")
    assert f"background: {colors.base};" in _rule_body(stylesheet, "#emptyState")


def test_apply_theme_installs_the_requested_scheme(themed_app: QApplication) -> None:
    """An explicit scheme installs that scheme's palette and stylesheet."""
    apply_theme(themed_app, Qt.ColorScheme.Dark)

    assert themed_app.palette().color(QPalette.ColorRole.Window) == QColor(DARK.window)
    assert themed_app.styleSheet() == stylesheet_for(DARK)


def test_apply_theme_follows_scheme_changes_once(
    themed_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated apply_theme calls connect one follower, so a scheme change re-themes once."""
    applied: list[ThemeColors] = []
    build_palette = theme.palette_for

    def record(colors: ThemeColors) -> QPalette:
        """Record the colours being applied and build the real palette."""
        applied.append(colors)
        return build_palette(colors)

    monkeypatch.setattr(theme, "palette_for", record)
    apply_theme(themed_app)
    apply_theme(themed_app)
    applied.clear()

    themed_app.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)

    assert applied == [DARK]
    assert themed_app.palette().color(QPalette.ColorRole.Window) == QColor(DARK.window)
