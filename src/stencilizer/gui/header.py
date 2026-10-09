"""Top bar with the wordmark and the open and save actions."""

from importlib.resources import files

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPaintEvent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from stencilizer.gui import assets

WORDMARK_FILE = "stencilizer.svg"
WORDMARK_HEIGHT = 22


class Wordmark(QWidget):
    """Paint the stencilled wordmark in the palette's text colour, crisp at any pixel ratio.

    The SVG is a single ``currentColor`` path; the colour is substituted before each load so
    the mark follows the theme like a label would.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Load the wordmark and size the widget to its aspect ratio at the header height."""
        super().__init__(parent)
        self.setObjectName("appLogo")
        self.setAccessibleName("Stencilizer")
        self._svg = files(assets).joinpath(WORDMARK_FILE).read_bytes()
        self._renderer = QSvgRenderer(self)
        self._load()
        box = self._renderer.viewBoxF()
        self.setFixedSize(round(WORDMARK_HEIGHT * box.width() / box.height()), WORDMARK_HEIGHT)

    def _load(self) -> None:
        """Re-read the SVG with the current palette's text colour filled in."""
        color = self.palette().windowText().color().name().encode("ascii")
        self._renderer.load(self._svg.replace(b"currentColor", color))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Recolour the wordmark when the theme's palette changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._load()
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        """Render the SVG into the widget rectangle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(painter, QRectF(self.rect()))


class HeaderBar(QFrame):
    """Show the wordmark and expose the window's primary actions."""

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
        """Create the wordmark and action buttons displayed in the header."""
        self.logo = Wordmark(self)

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
        """Arrange the wordmark and actions in one row."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)
        layout.addWidget(self.logo)
        layout.addSpacing(12)
        layout.addStretch(1)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)

    def _emit_open_requested(self, _checked: bool = False) -> None:
        """Emit the request to choose a source font."""
        self.open_requested.emit()

    def _emit_save_requested(self, _checked: bool = False) -> None:
        """Emit the request to stencilize and save the loaded font."""
        self.save_requested.emit()

    def set_font_loaded(self, loaded: bool) -> None:
        """Update save availability for the current font state."""
        self._font_loaded = loaded
        self.save_button.setEnabled(loaded and not self._busy)

    def set_busy(self, busy: bool) -> None:
        """Disable actions while an operation is in progress."""
        self._busy = busy
        self.open_button.setEnabled(not busy)
        self.save_button.setEnabled(self._font_loaded and not busy)
