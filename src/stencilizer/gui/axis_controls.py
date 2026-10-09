"""Variable-font axis sliders shown as a sidebar card."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from stencilizer.gui.variable_session import AxisInfo

COMPOSITE_NOTE = "Composite glyphs preview at the default axis location."
_MAX_ROWS_HEIGHT = 240
_FINE_RANGE = 50.0


class AxisRow:
    """One axis: a label, an integer-stepped slider and a spin box kept in sync."""

    def __init__(self, axis: AxisInfo, parent: QWidget) -> None:
        """Build the row's widgets for ``axis`` with the slider at its default."""
        self.axis = axis
        # Narrow axes (Ubuntu wdth 75..100) get tenth-unit steps.
        self.scale = 10 if axis.maximum - axis.minimum < _FINE_RANGE else 1
        # The name comes from the font's name table: never render it as rich text, and double
        # each "&" so the buddy does not read it as a mnemonic marker.
        self.label = QLabel(axis.name.replace("&", "&&"), parent)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setObjectName(f"axis-slider-{axis.tag}")
        self.slider.setRange(round(axis.minimum * self.scale), round(axis.maximum * self.scale))
        self.slider.setValue(round(axis.default * self.scale))
        self.spin = QDoubleSpinBox(parent)
        self.spin.setObjectName(f"axis-spin-{axis.tag}")
        self.spin.setDecimals(1 if self.scale == 10 else 0)
        self.spin.setRange(axis.minimum, axis.maximum)
        self.spin.setValue(axis.default)
        self.label.setBuddy(self.spin)

    @property
    def value(self) -> float:
        """The slider's position in user-space units."""
        return self.slider.value() / self.scale

    def sync_spin(self) -> None:
        """Show the slider position in the spin box without re-emitting."""
        blocker = QSignalBlocker(self.spin)
        self.spin.setValue(self.value)
        del blocker

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable both inputs."""
        self.slider.setEnabled(enabled)
        self.spin.setEnabled(enabled)


class AxisPanel(QFrame):
    """A card with one slider per variable axis; empty (hidden) for static fonts."""

    location_changed = Signal(dict)
    axes_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the hidden card with its scroll area and composite note."""
        super().__init__(parent)
        self.setProperty("role", "card")
        self._rows: list[AxisRow] = []
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.viewport().setAutoFillBackground(False)
        self._note = QLabel(COMPOSITE_NOTE, self)
        self._note.setObjectName("axis-composite-note")
        self._note.setProperty("role", "hint")
        self._note.setWordWrap(True)
        self._note.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._scroll)
        layout.addWidget(self._note)
        self.hide()

    def set_axes(self, axes: tuple[AxisInfo, ...]) -> None:
        """Rebuild one row per axis at its default; hide the card when there are none."""
        content = QWidget()
        content.setAutoFillBackground(False)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)
        self._rows = [AxisRow(axis, content) for axis in axes]
        for row in self._rows:
            header = QHBoxLayout()
            header.addWidget(row.label)
            header.addStretch()
            header.addWidget(row.spin)
            layout.addLayout(header)
            layout.addWidget(row.slider)
            row.slider.valueChanged.connect(lambda _value, r=row: self._on_slider(r))
            row.spin.valueChanged.connect(lambda value, r=row: self._on_spin(r, value))
        layout.addStretch()
        self._scroll.setWidget(content)
        self._scroll.setFixedHeight(min(content.sizeHint().height() + 2, _MAX_ROWS_HEIGHT))
        self._note.hide()
        self.setVisible(bool(axes))
        self.axes_changed.emit(bool(axes))

    def location(self) -> dict[str, float]:
        """Return the current user-space ``{tag: value}``."""
        return {row.axis.tag: row.value for row in self._rows}

    def set_location_applies(self, applies: bool) -> None:
        """Disable the inputs and show the composite note when the location has no effect."""
        for row in self._rows:
            row.set_enabled(applies)
        self._note.setVisible(bool(self._rows) and not applies)

    def _on_slider(self, row: AxisRow) -> None:
        row.sync_spin()
        self.location_changed.emit(self.location())

    def _on_spin(self, row: AxisRow, value: float) -> None:
        row.slider.setValue(round(value * row.scale))
