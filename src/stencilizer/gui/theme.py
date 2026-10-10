"""Light and dark palettes and the stylesheet that give the desktop GUI its look."""

import sys
from dataclasses import asdict, dataclass
from string import Template

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    """Colour tokens of one theme, every field a ``#rrggbb`` string."""

    window: str
    surface: str
    base: str
    border: str
    text: str
    muted_text: str
    accent: str
    accent_text: str


LIGHT = ThemeColors(
    window="#f4f5f7",
    surface="#ffffff",
    base="#ffffff",
    border="#dde1e6",
    text="#1c1f24",
    muted_text="#5f6670",
    accent="#2563eb",
    accent_text="#ffffff",
)

DARK = ThemeColors(
    window="#16181d",
    surface="#1e2127",
    base="#121418",
    border="#2c3038",
    text="#e6e8eb",
    muted_text="#9aa1ab",
    accent="#2563eb",
    accent_text="#ffffff",
)

_FOLLOWER_NAME = "themeSchemeFollower"

_ROLE_FIELDS: tuple[tuple[QPalette.ColorRole, str], ...] = (
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

_DISABLED_ROLES = (
    QPalette.ColorRole.Text,
    QPalette.ColorRole.ButtonText,
    QPalette.ColorRole.WindowText,
)

_CHROME_QSS = """
#headerBar { background: $surface; border-bottom: 1px solid $border; }
#sidebar { background: $window; }
#previewPane { background: $window; }
QFrame[role="card"] { background: $surface; border: 1px solid $border; border-radius: 10px; }
QLabel[role="sectionTitle"] { font-size: 8pt; font-weight: 700; color: $muted_text; }
QLabel[role="value"] { font-weight: 600; color: $text; }
QLabel[role="infoTitle"] { font-size: 10pt; font-weight: 700; color: $text; }
QLabel[role="hint"] { font-size: 8pt; color: $muted_text; }
QLabel[role="status"] { color: $muted_text; }
QSplitter::handle { background: $border; }
QStatusBar { background: $surface; color: $muted_text; border-top: 1px solid $border; }
QStatusBar::item { border: none; }
QProgressBar#saveProgress {
    background: $base;
    border: 1px solid $border;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
}
QProgressBar#saveProgress::chunk { background: $accent; border-radius: 3px; }
QToolTip { background: $surface; color: $text; border: 1px solid $border; padding: 4px 8px; }
"""

_BUTTON_QSS = """
QPushButton[role="primary"], QPushButton[role="secondary"] {
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
}
QPushButton[role="primary"] { background: $accent; color: $accent_text; border: 1px solid $accent; }
QPushButton[role="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[role="primary"]:pressed { background: $accent_pressed; border-color: $accent_pressed; }
QPushButton[role="primary"]:disabled {
    background: $border;
    color: $muted_text;
    border-color: $border;
}
QPushButton[role="secondary"] {
    background: $surface;
    color: $text;
    border: 1px solid $border_strong;
}
QPushButton[role="secondary"]:hover { background: $hover; }
QPushButton[role="secondary"]:pressed { background: $pressed; }
QPushButton[role="secondary"]:disabled { color: $muted_text; border-color: $border; }
QPushButton[role="primary"]:focus {
    background: $accent;
    border: 2px solid $focus_on_accent;
    padding: 5px 13px;
}
QPushButton[role="primary"]:focus:pressed { padding: 6px 13px 4px 13px; }
QPushButton[role="secondary"]:focus { border: 2px solid $focus; padding: 5px 13px; }
"""

_INPUT_QSS = """
QSlider:horizontal { min-height: 22px; }
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 10px;
    margin: -5px 0;
    background: $accent;
    border: 2px solid $accent;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: $accent_hover; border-color: $accent_hover; }
QSlider::handle:horizontal:focus { background: $accent; border-color: $focus_on_accent; }
QSlider::handle:horizontal:disabled, QSlider::sub-page:horizontal:disabled {
    background: $muted_text;
    border-color: $muted_text;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    background: $base;
    border: 1px solid $border_strong;
    border-radius: 5px;
}
QCheckBox::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; }
QCheckBox::indicator:checked:disabled { background: $muted_text; border-color: $muted_text; }
QCheckBox::indicator:focus { width: 16px; height: 16px; border: 2px solid $focus; }
QCheckBox::indicator:checked:focus { border-color: $focus_on_accent; }
QSpinBox, QDoubleSpinBox {
    background: $base;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 3px 6px;
    selection-background-color: $accent;
    selection-color: $accent_text;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: $accent; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: transparent; }
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid $base;
    border-right: 4px solid $base;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { border-bottom: 5px solid $muted_text; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { border-top: 5px solid $muted_text; }
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover { border-bottom-color: $text; }
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover { border-top-color: $text; }
QComboBox {
    background: $base;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 4px 10px;
}
QComboBox:focus { border-color: $accent; }
QComboBox:disabled { background: $surface; color: $muted_text; border-color: $border; }
QComboBox::drop-down {
    width: 24px;
    border: none;
    background: transparent;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid $base;
    border-right: 5px solid $base;
    border-top: 6px solid $muted_text;
}
QComboBox::down-arrow:disabled { border-color: $surface; border-top-color: $border; }
QComboBox QAbstractItemView {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    padding: 4px;
    outline: 0;
    selection-background-color: $accent;
    selection-color: $accent_text;
}
"""

_CONTENT_QSS = """
#emptyState { background: $base; color: $muted_text; font-size: 11pt; padding: 24px; }
QListWidget#glyphGrid { background: $base; border: none; padding: 8px; outline: 0; }
QListWidget#glyphGrid::item { border: 2px solid transparent; border-radius: 8px; padding: 2px; }
QListWidget#glyphGrid::item:hover { border-color: $border; }
QListWidget#glyphGrid::item:selected {
    background: $selection;
    border-color: $selection;
    color: $text;
}
QListWidget#glyphGrid::item:focus { border-color: $focus; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px 2px 2px 0; }
QScrollBar::handle:vertical { background: $border; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: $muted_text; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""

_STYLESHEET = Template("".join((_CHROME_QSS, _BUTTON_QSS, _INPUT_QSS, _CONTENT_QSS)))


def colors_for(scheme: Qt.ColorScheme) -> ThemeColors:
    """Return the dark tokens for a dark scheme and the light tokens otherwise."""
    return DARK if scheme == Qt.ColorScheme.Dark else LIGHT


def palette_for(colors: ThemeColors) -> QPalette:
    """Map the tokens onto palette roles; canvases and thumbnails paint with Base, Text and Mid."""
    tokens = asdict(colors)
    palette = QPalette(QColor(colors.window))
    for role, field in _ROLE_FIELDS:
        palette.setColor(role, QColor(tokens[field]))
    muted = QColor(colors.muted_text)
    for role in _DISABLED_ROLES:
        palette.setColor(QPalette.ColorGroup.Disabled, role, muted)
    return palette


def _blend(first: str, second: str, amount: float) -> str:
    """Mix ``amount`` (0..1) of ``second`` into ``first``, both ``#rrggbb``."""
    start, end = QColor(first), QColor(second)
    return QColor(
        round(start.red() + (end.red() - start.red()) * amount),
        round(start.green() + (end.green() - start.green()) * amount),
        round(start.blue() + (end.blue() - start.blue()) * amount),
    ).name()


def _derived_colors(colors: ThemeColors) -> dict[str, str]:
    """Return the colours the stylesheet derives from the tokens, keyed by placeholder name.

    ``border_strong`` bounds the secondary button, the check box and the input fields at 3:1 or
    more against surface, base and window. ``selection`` is the accent at 30 % over the base, the
    wash Qt lays over a selected thumbnail, so a selected grid cell painted with it swallows the
    thumbnail's edges. ``focus`` rings a focused control on a neutral fill at 3:1 or more in both
    themes, where the plain accent falls short on dark hover and selection fills.
    ``focus_on_accent`` rings a control filled with the accent (the primary button, a checked
    box, a slider handle), which no accent-like ring stands out from.
    """
    return {
        "border_strong": _blend(colors.border, colors.text, 0.45),
        "hover": _blend(colors.surface, colors.text, 0.06),
        "pressed": _blend(colors.surface, colors.text, 0.12),
        "accent_hover": _blend(colors.accent, colors.text, 0.15),
        "accent_pressed": _blend(colors.accent, colors.text, 0.3),
        "selection": _blend(colors.base, colors.accent, 0.3),
        "focus": _blend(colors.accent, colors.text, 0.3),
        "focus_on_accent": colors.text,
    }


def stylesheet_for(colors: ThemeColors) -> str:
    """Build the application stylesheet from the tokens and the colours derived from them.

    Arrows are border triangles whose side edges take the field colour: QSS skips the mitre
    next to a transparent edge. Focus rings are 2 px: a focus rule over a 1 px border takes 1 px
    off the padding or the size so the content stays in place, and the slider handle keeps a
    2 px accent border at rest for the same reason.
    """
    return _STYLESHEET.substitute(asdict(colors), **_derived_colors(colors))


def _apply_colors(app: QApplication, colors: ThemeColors) -> None:
    """Install one theme's palette and stylesheet on the application."""
    app.setPalette(palette_for(colors))
    app.setStyleSheet(stylesheet_for(colors))


class _SchemeFollower(QObject):
    """Re-theme the application whenever the system colour scheme changes."""

    def __init__(self, app: QApplication) -> None:
        """Attach to ``app`` and listen for colour-scheme changes."""
        super().__init__(app)
        self.setObjectName(_FOLLOWER_NAME)
        self._app = app
        self._portal_scheme = Qt.ColorScheme.Unknown
        if sys.platform == "linux":
            from stencilizer.gui.portal_theme import PortalTheme

            portal = PortalTheme(self)
            self._portal_scheme = portal.scheme
            portal.changed.connect(self._on_portal_changed)
        app.styleHints().colorSchemeChanged.connect(self._on_scheme_changed)

    def resolve(self, scheme: Qt.ColorScheme) -> Qt.ColorScheme:
        """Use the desktop portal when the Qt platform plugin has no preference."""
        return self._portal_scheme if scheme == Qt.ColorScheme.Unknown else scheme

    def _on_portal_changed(self, scheme: Qt.ColorScheme) -> None:
        self._portal_scheme = scheme
        self._on_scheme_changed(self._app.styleHints().colorScheme())

    def _on_scheme_changed(self, scheme: Qt.ColorScheme) -> None:
        """Apply the theme matching the new scheme."""
        _apply_colors(self._app, colors_for(self.resolve(scheme)))


def apply_theme(app: QApplication, scheme: Qt.ColorScheme | None = None) -> None:
    """Style ``app`` for ``scheme``, or for the system scheme, following later system changes."""
    app.setStyle("Fusion")
    if scheme is None:
        scheme = app.styleHints().colorScheme()
        follower = app.findChild(_SchemeFollower, _FOLLOWER_NAME)
        if follower is None:
            follower = _SchemeFollower(app)
        scheme = follower.resolve(scheme)
    _apply_colors(app, colors_for(scheme))
