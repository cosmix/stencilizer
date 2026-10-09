"""Sidebar panel listing everything known about the open font."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from stencilizer.gui.font_info import FontInfo, InfoSection

PLACEHOLDER_TITLE = "No font loaded"
PLACEHOLDER_HINT = "Open a TrueType or OpenType font (.ttf, .otf)"
_LABEL_COLUMN_WIDTH = 104


def _plain_label(text: str, parent: QWidget, role: str | None = None) -> QLabel:
    """Create a word-wrapped label that shows ``text`` literally, never as rich text."""
    label = QLabel(text, parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if role is not None:
        label.setProperty("role", role)
    return label


class FontInfoPanel(QWidget):
    """A titled, scrollable card of font facts that wraps inside the sidebar width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the panel showing the no-font placeholder."""
        super().__init__(parent)
        self.setObjectName("fontInfoPanel")
        self.value_labels: list[QLabel] = []
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("fontInfoScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(120)
        self._scroll.viewport().setAutoFillBackground(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)
        self.clear()

    def clear(self) -> None:
        """Show the placeholder that asks for a font."""
        self.value_labels = []
        card, layout = self._new_card()
        layout.addWidget(_plain_label(PLACEHOLDER_TITLE, card, "infoTitle"))
        layout.addWidget(_plain_label(PLACEHOLDER_HINT, card, "hint"))
        layout.addStretch()
        self._scroll.setWidget(card)

    def set_info(self, info: FontInfo) -> None:
        """Replace the shown rows with those of ``info``."""
        self.value_labels = []
        card, layout = self._new_card()
        layout.addWidget(_plain_label(info.title, card, "infoTitle"))
        for section in info.sections:
            layout.addWidget(_plain_label(section.title.upper(), card, "sectionTitle"))
            layout.addLayout(self._rows(section, card))
        layout.addStretch()
        self._scroll.setWidget(card)
        self._scroll.verticalScrollBar().setValue(0)

    def _new_card(self) -> tuple[QFrame, QVBoxLayout]:
        """Create the card that fills the scroll area, with its vertical layout."""
        card = QFrame()
        card.setObjectName("fontInfoCard")
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        return card, layout

    def _rows(self, section: InfoSection, card: QWidget) -> QGridLayout:
        """Lay out one section as a muted label column beside wrapping values."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnMinimumWidth(0, _LABEL_COLUMN_WIDTH)
        grid.setColumnStretch(1, 1)
        for row, (name, value) in enumerate(section.rows):
            label = _plain_label(name, card, "hint")
            label.setFixedWidth(_LABEL_COLUMN_WIDTH)
            text = _plain_label(value, card)
            text.setObjectName("fontInfoValue")
            text.setToolTip(value)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            # Ignored lets an unbreakable value (a URL) clip instead of forcing a horizontal scroll.
            text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(text, row, 1, Qt.AlignmentFlag.AlignTop)
            self.value_labels.append(text)
        return grid
