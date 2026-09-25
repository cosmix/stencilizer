"""Top bar with the loaded font's identity and the open and save actions."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class HeaderBar(QFrame):
    """Present font information and expose the window's primary actions."""

    open_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the header and its action controls."""
        super().__init__(parent)
        self.setObjectName("headerBar")
        self._font_loaded = False
        self._busy = False
        self._build_widgets()
        self._build_layout()
        self.open_button.clicked.connect(self._emit_open_requested)
        self.save_button.clicked.connect(self._emit_save_requested)

    def _build_widgets(self) -> None:
        """Create the labels and action buttons displayed in the header."""
        self.title_label = QLabel("Stencilizer", self)
        self.title_label.setObjectName("appTitle")

        self.font_name_label = QLabel("No font loaded", self)
        self.font_name_label.setObjectName("fontName")
        self.font_details_label = QLabel("Open a TrueType or OpenType font (.ttf, .otf)", self)
        self.font_details_label.setObjectName("fontDetails")
        for label in (self.font_name_label, self.font_details_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.open_button = QPushButton("Open Font…", self)
        self.open_button.setProperty("role", "secondary")
        self.open_button.setToolTip("Open a TTF or OTF font")
        self.save_button = QPushButton("Stencilize && Save…", self)
        self.save_button.setProperty("role", "primary")
        self.save_button.setToolTip("Stencilize every glyph and save a new font file")
        self.save_button.setEnabled(False)
        for button in (self.open_button, self.save_button):
            button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

    def _build_layout(self) -> None:
        """Arrange the title, font details, and actions in one row."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)
        layout.addWidget(self.title_label)
        info = QVBoxLayout()
        info.setSpacing(2)
        info.addWidget(self.font_name_label)
        info.addWidget(self.font_details_label)
        layout.addLayout(info, 1)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)

    def _emit_open_requested(self, _checked: bool = False) -> None:
        """Emit the request to choose a source font."""
        self.open_requested.emit()

    def _emit_save_requested(self, _checked: bool = False) -> None:
        """Emit the request to stencilize and save the loaded font."""
        self.save_requested.emit()

    def set_font_info(self, name: str, details: str) -> None:
        """Display the loaded font's name and descriptive details."""
        self.font_name_label.setText(name)
        self.font_name_label.setToolTip(name)
        self.font_details_label.setText(details)

    def set_font_loaded(self, loaded: bool) -> None:
        """Update save availability for the current font state."""
        self._font_loaded = loaded
        self.save_button.setEnabled(loaded and not self._busy)

    def set_busy(self, busy: bool) -> None:
        """Disable actions while an operation is in progress."""
        self._busy = busy
        self.open_button.setEnabled(not busy)
        self.save_button.setEnabled(self._font_loaded and not busy)
