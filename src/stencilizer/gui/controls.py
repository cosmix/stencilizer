"""Sidebar with the bridge and processing parameters."""

import os
from functools import partial

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stencilizer.config import BridgeConfig, BridgeWidthScaling
from stencilizer.gui.axis_controls import AxisPanel
from stencilizer.gui.font_info_panel import FontInfoPanel

_BRIDGE_WIDTH_TOOLTIP = "Bridge width as percent of a reference stroke of 10% of font UPM (30-110)"
_SCALING_TOOLTIP = (
    "Fixed: the same gap in every master. Proportional: gaps follow the weight of each master. "
    "The default master is the same in both modes."
)
_STRENGTH_TOOLTIP = (
    "How strongly bridge gaps follow stroke thickness: 0 keeps the width fixed, "
    "100 is fully proportional"
)
_MIN_WIDTH_TOOLTIP = (
    "Smallest bridge gap in light masters, as percent of a reference stroke of 10% of font UPM"
)
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

        self._build_scaling_widgets()

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

    def _build_scaling_widgets(self) -> None:
        """Create the width-scaling mode, strength and minimum controls."""
        self.scaling_box = QWidget(self)
        self.scaling_box.hide()
        self.scaling_combo = QComboBox(self.scaling_box)
        self.scaling_combo.addItem("Fixed", BridgeWidthScaling.FIXED)
        self.scaling_combo.addItem("Proportional", BridgeWidthScaling.PROPORTIONAL)
        self.scaling_combo.setToolTip(_SCALING_TOOLTIP)
        self.strength_slider, self.strength_spin = self._slider_spin(
            (0, 100), 100, _STRENGTH_TOOLTIP
        )
        self.min_width_slider, self.min_width_spin = self._slider_spin(
            (10, 110), 30, _MIN_WIDTH_TOOLTIP
        )

    def _slider_spin(
        self, bounds: tuple[int, int], value: int, tooltip: str
    ) -> tuple[QSlider, QSpinBox]:
        """Create a slider and a percent spin box with the same range and value."""
        slider = QSlider(Qt.Orientation.Horizontal, self.scaling_box)
        spin = QSpinBox(self.scaling_box)
        for widget in (slider, spin):
            widget.setRange(*bounds)
            widget.setValue(value)
            widget.setToolTip(tooltip)
        spin.setSuffix(" %")
        return slider, spin

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
        layout.addWidget(self._scaling_group(card))
        layout.addWidget(self.spanning_check)
        return card

    def _scaling_group(self, card: QFrame) -> QWidget:
        """Lay out the width-scaling controls inside their container."""
        box = self.scaling_box
        box.setParent(card)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        mode_label = QLabel("Width scaling", box)
        mode_label.setBuddy(self.scaling_combo)
        layout.addWidget(mode_label)
        layout.addWidget(self.scaling_combo)
        self._scaling_rows = [
            self._scaling_row("Strength", self.strength_spin, self.strength_slider),
            self._scaling_row("Minimum", self.min_width_spin, self.min_width_slider),
        ]
        for row in self._scaling_rows:
            layout.addWidget(row)
            row.setEnabled(False)
        box.hide()
        return box

    def _scaling_row(self, text: str, spin: QSpinBox, slider: QSlider) -> QWidget:
        """Build a labelled spin box above its slider."""
        row = QWidget(self.scaling_box)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        label = QLabel(text, row)
        label.setBuddy(spin)
        header.addWidget(label)
        header.addStretch()
        header.addWidget(spin)
        layout.addLayout(header)
        layout.addWidget(slider)
        return row

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
        for slider, spin in (
            (self.width_slider, self.width_spin),
            (self.strength_slider, self.strength_spin),
            (self.min_width_slider, self.min_width_spin),
        ):
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(partial(self._sync_spin, spin))
        self.scaling_combo.currentIndexChanged.connect(self._on_scaling_changed)
        self.spanning_check.toggled.connect(self._emit_parameters_changed)
        self.workers_slider.valueChanged.connect(self._sync_workers_label)
        self.axes_panel.axes_changed.connect(self._axes_title.setVisible)

    def _emit_parameters_changed(self, _checked: bool) -> None:
        """Forward a spanning-bridge change through the parameter signal."""
        self.parameters_changed.emit()

    def _sync_spin(self, spin: QSpinBox, value: int) -> None:
        """Keep the spin box in sync and emit one parameter change."""
        blocker = QSignalBlocker(spin)
        spin.setValue(value)
        del blocker
        self.parameters_changed.emit()

    def _on_scaling_changed(self, _index: int) -> None:
        """Enable the strength and minimum rows for proportional scaling."""
        proportional = self.scaling_combo.currentData() == BridgeWidthScaling.PROPORTIONAL
        for row in self._scaling_rows:
            row.setEnabled(proportional)
        self.parameters_changed.emit()

    def set_variable(self, variable: bool) -> None:
        """Show the width-scaling controls for variable fonts only."""
        self.scaling_box.setVisible(variable)
        if not variable and self.scaling_combo.currentIndex() != 0:
            self.scaling_combo.setCurrentIndex(0)

    def _sync_workers_label(self, value: int) -> None:
        """Show automatic processing or the selected worker count."""
        self.workers_value_label.setText("Auto" if value == 0 else str(value))

    def bridge_config(self) -> BridgeConfig:
        """Return the selected bridge parameters."""
        return BridgeConfig(
            width_percent=float(self.width_slider.value()),
            use_spanning_bridges=self.spanning_check.isChecked(),
            width_scaling=BridgeWidthScaling(self.scaling_combo.currentData()),
            scaling_strength=float(self.strength_slider.value()),
            min_width_percent=float(self.min_width_slider.value()),
        )

    def max_workers(self) -> int | None:
        """Return the worker limit, or ``None`` when automatic is selected."""
        value = self.workers_slider.value()
        return None if value == 0 else value
