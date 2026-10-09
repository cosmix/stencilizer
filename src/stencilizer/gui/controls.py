"""Sidebar with the bridge and processing parameters."""

import os

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stencilizer.config import BridgeConfig
from stencilizer.gui.axis_controls import AxisPanel
from stencilizer.gui.font_info_panel import FontInfoPanel

_BRIDGE_WIDTH_TOOLTIP = "Bridge width as percent of a reference stroke of 10% of font UPM (30-110)"
_WORKERS_TOOLTIP = "Worker processes used when saving; Auto lets the processor decide"


def _section_title(text: str, parent: QWidget) -> QLabel:
    """Create a styled sidebar section title."""
    label = QLabel(text, parent)
    label.setProperty("role", "sectionTitle")
    return label


class ControlPanel(QWidget):
    """Bridge and processing parameters for the preview and the save."""

    parameters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the font controls and connect their signals."""
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(240)
        self.setMaximumWidth(360)
        self._build_widgets()
        self._build_layout()
        self._connect_signals()

    def _build_widgets(self) -> None:
        """Create every control this panel owns, with its initial state."""
        self.width_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.width_slider.setRange(30, 110)
        self.width_slider.setValue(60)
        self.width_slider.setToolTip(_BRIDGE_WIDTH_TOOLTIP)

        self.width_spin = QSpinBox(self)
        self.width_spin.setRange(30, 110)
        self.width_spin.setValue(60)
        self.width_spin.setSuffix(" %")
        self.width_spin.setToolTip(_BRIDGE_WIDTH_TOOLTIP)

        self.spanning_check = QCheckBox("Spanning bridges for stacked islands", self)
        self.spanning_check.setChecked(True)

        self.workers_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.workers_slider.setRange(0, os.cpu_count() or 1)
        self.workers_slider.setValue(0)
        self.workers_slider.setPageStep(1)
        self.workers_slider.setToolTip(_WORKERS_TOOLTIP)

        self.workers_value_label = QLabel("Auto", self)
        self.workers_value_label.setProperty("role", "value")
        self.workers_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.axes_panel = AxisPanel(self)
        self.font_info = FontInfoPanel(self)

    def _build_layout(self) -> None:
        """Arrange the bridge and processing controls into styled sections."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(_section_title("BRIDGES", self))
        layout.addWidget(self._bridges_card())
        layout.addSpacing(6)
        layout.addWidget(_section_title("PROCESSING", self))
        layout.addWidget(self._processing_card())
        layout.addSpacing(6)
        self._axes_title = _section_title("AXES", self)
        self._axes_title.hide()
        layout.addWidget(self._axes_title)
        layout.addWidget(self.axes_panel)
        layout.addSpacing(6)
        layout.addWidget(_section_title("FONT", self))
        layout.addWidget(self.font_info, 1)

    def _bridges_card(self) -> QFrame:
        """Build the card containing bridge parameter controls."""
        card = QFrame(self)
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        width_row = QHBoxLayout()
        width_label = QLabel("Width", card)
        width_label.setBuddy(self.width_spin)
        width_row.addWidget(width_label)
        width_row.addStretch()
        width_row.addWidget(self.width_spin)
        layout.addLayout(width_row)
        layout.addWidget(self.width_slider)
        layout.addWidget(self.spanning_check)
        return card

    def _processing_card(self) -> QFrame:
        """Build the card containing worker-process controls."""
        card = QFrame(self)
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        workers_row = QHBoxLayout()
        workers_label = QLabel("Workers", card)
        workers_label.setBuddy(self.workers_slider)
        workers_row.addWidget(workers_label)
        workers_row.addStretch()
        workers_row.addWidget(self.workers_value_label)
        layout.addLayout(workers_row)
        layout.addWidget(self.workers_slider)

        hint = QLabel("Parallel processes used when saving the font", card)
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return card

    def _connect_signals(self) -> None:
        """Wire widget signals to their syncing and forwarding slots."""
        self.width_spin.valueChanged.connect(self.width_slider.setValue)
        self.width_slider.valueChanged.connect(self._sync_width_spin)
        self.spanning_check.toggled.connect(self._emit_parameters_changed)
        self.workers_slider.valueChanged.connect(self._sync_workers_label)
        self.axes_panel.axes_changed.connect(self._axes_title.setVisible)

    def _emit_parameters_changed(self, _checked: bool) -> None:
        """Forward a spanning-bridge change through the parameter signal."""
        self.parameters_changed.emit()

    def _sync_width_spin(self, value: int) -> None:
        """Keep the spin box in sync and emit one parameter change."""
        blocker = QSignalBlocker(self.width_spin)
        self.width_spin.setValue(value)
        del blocker
        self.parameters_changed.emit()

    def _sync_workers_label(self, value: int) -> None:
        """Show automatic processing or the selected worker count."""
        self.workers_value_label.setText("Auto" if value == 0 else str(value))

    def bridge_config(self) -> BridgeConfig:
        """Return the selected bridge parameters."""
        return BridgeConfig(
            width_percent=float(self.width_slider.value()),
            use_spanning_bridges=self.spanning_check.isChecked(),
        )

    def max_workers(self) -> int | None:
        """Return the worker limit, or ``None`` when automatic is selected."""
        value = self.workers_slider.value()
        return None if value == 0 else value
