"""Tests for the light and dark palettes, the stylesheet and system-scheme following."""

import re
from collections.abc import Iterator
from dataclasses import asdict

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from stencilizer.gui import theme
from stencilizer.gui.theme import (
    DARK,
    LIGHT,
    ThemeColors,
    _derived_colors,
    apply_theme,
    colors_for,
    palette_for,
    stylesheet_for,
)
from tests.gui.test_beautify_contracts import _contrast

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

FOCUS_RINGS = (
    ('QPushButton[role="primary"]:focus', ("accent", "surface")),
    ('QPushButton[role="secondary"]:focus', ("surface", "hover")),
    ("QSlider::handle:horizontal:focus", ("accent", "surface", "border")),
    ("QCheckBox::indicator:focus", ("base", "surface")),
    ("QCheckBox::indicator:checked:focus", ("accent", "surface")),
    ("QListWidget#glyphGrid::item:focus", ("base", "selection")),
)
"""Each focus rule and the fills its ring borders: the control's own fill and what is behind it."""

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


def _declared_color(body: str, properties: str) -> str:
    """Return the ``#rrggbb`` colour of the first declaration named by the ``properties`` regex."""
    match = re.search(rf"(?:{properties}): (?:[^;]* )?(#[0-9a-f]{{6}});", body)
    assert match is not None, (properties, body)
    return match.group(1)


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


@THEMES
def test_strong_border_meets_non_text_contrast(colors: ThemeColors) -> None:
    """The border that alone bounds buttons, the check box and inputs reaches 3:1 everywhere."""
    border = _derived_colors(colors)["border_strong"]

    for background in (colors.surface, colors.base, colors.window):
        assert _contrast(border, background) >= 3.0, background


@THEMES
def test_stylesheet_marks_keyboard_focus(colors: ThemeColors) -> None:
    """Buttons, sliders, the check box and the glyph grid each have a focus rule."""
    stylesheet = stylesheet_for(colors)

    missing = [selector for selector, _fills in FOCUS_RINGS if f"\n{selector} {{" not in stylesheet]
    assert missing == []


@THEMES
def test_focus_rings_contrast_with_their_fills(colors: ThemeColors) -> None:
    """Every focus ring reaches 3:1 against its control's fill and every background."""
    stylesheet = stylesheet_for(colors)
    tokens = {**asdict(colors), **_derived_colors(colors)}

    for selector, fills in FOCUS_RINGS:
        ring = _declared_color(_rule_body(stylesheet, selector), "border|border-color")
        for fill in (*fills, "base", "window"):
            assert _contrast(ring, tokens[fill]) >= 3.0, (selector, fill)


@THEMES
def test_selected_glyph_label_stays_legible(colors: ThemeColors) -> None:
    """A selected grid cell draws its label in the text colour, legible on the selection."""
    body = _rule_body(stylesheet_for(colors), "QListWidget#glyphGrid::item:selected")

    assert f"color: {colors.text};" in body
    assert _contrast(colors.text, _derived_colors(colors)["selection"]) >= 4.5


def _vertical_padding(body: str) -> int:
    """Sum the top and bottom pixels of a 2- or 4-value ``padding`` shorthand declaration."""
    match = re.search(r"padding: ([^;]+);", body)
    assert match is not None, body
    values = [int(pixels) for pixels in re.findall(r"(\d+)px", match.group(1))]
    return values[0] * 2 if len(values) == 2 else values[0] + values[2]


@THEMES
def test_focus_on_accent_keeps_the_accent_fill_and_press_feedback(colors: ThemeColors) -> None:
    """A focused primary button and slider handle keep the accent fill, with padding-only press."""
    stylesheet = stylesheet_for(colors)

    for selector in ('QPushButton[role="primary"]:focus', "QSlider::handle:horizontal:focus"):
        body = _rule_body(stylesheet, selector)
        assert _declared_color(body, "background") == colors.accent, selector
        ring = _declared_color(body, "border|border-color")
        assert _contrast(ring, colors.accent) >= 3.0, selector

    focus_padding = _vertical_padding(_rule_body(stylesheet, 'QPushButton[role="primary"]:focus'))
    pressed_padding = _vertical_padding(
        _rule_body(stylesheet, 'QPushButton[role="primary"]:focus:pressed')
    )
    assert pressed_padding == focus_padding


@THEMES
def test_save_progress_chunk_stands_out_from_its_groove(colors: ThemeColors) -> None:
    """The filled part of the save progress bar reaches 3:1 against the empty part."""
    stylesheet = stylesheet_for(colors)
    groove = _declared_color(_rule_body(stylesheet, "QProgressBar#saveProgress"), "background")
    chunk = _declared_color(
        _rule_body(stylesheet, "QProgressBar#saveProgress::chunk"), "background"
    )

    assert _contrast(chunk, groove) >= 3.0
