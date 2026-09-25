"""Widget for choosing a glyph's bridge direction."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from stencilizer.config.settings import BridgeDirection


class DirectionPicker(QWidget):
    """Choose the bridge direction of the selected glyph."""

    direction_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the direction controls in their cleared state."""
        super().__init__(parent)
        self.source_label = QLabel("", self)
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        self.combo = QComboBox(self)
        self.combo.addItem("Auto", BridgeDirection.AUTO.value)
        self.combo.addItem("Vertical (cuts top and bottom)", BridgeDirection.VERTICAL.value)
        self.combo.addItem("Horizontal (cuts left and right)", BridgeDirection.HORIZONTAL.value)

        layout = QVBoxLayout(self)
        layout.addWidget(self.source_label)
        layout.addWidget(self.combo)

        self.combo.currentIndexChanged.connect(self._emit_direction_chosen)
        self.clear()

    def show_for(self, name: str, sources: tuple[str, ...], direction: BridgeDirection) -> None:
        """Show direction and source information for one displayed glyph."""
        blocker = QSignalBlocker(self.combo)
        self.combo.setCurrentIndex(self.combo.findData(direction.value))
        del blocker

        enabled = sources == (name,)
        self.combo.setEnabled(enabled)
        if enabled:
            self.source_label.setText(f"Bridge direction for {name}")
        elif sources:
            self.source_label.setText(f"Follows {', '.join(sources)}")
        else:
            self.source_label.setText("No islands to bridge")

    def clear(self) -> None:
        """Reset to Auto and hide direction controls until a glyph is selected."""
        blocker = QSignalBlocker(self.combo)
        self.combo.setCurrentIndex(self.combo.findData(BridgeDirection.AUTO.value))
        del blocker
        self.combo.setEnabled(False)
        self.source_label.clear()

    def _emit_direction_chosen(self, _index: int) -> None:
        """Forward the selected direction value when the user changes it."""
        self.direction_chosen.emit(str(self.combo.currentData()))
