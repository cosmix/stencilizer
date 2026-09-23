"""Controls for loading, configuring, and saving a stencilized font."""

import os

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stencilizer.config import BridgeConfig

_BRIDGE_WIDTH_TOOLTIP = "Bridge width as percent of a reference stroke of 10% of font UPM (30-110)"


class ControlPanel(QWidget):
    """Font selection, parameters, and the save action."""

    open_requested = Signal()
    save_requested = Signal()
    parameters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the font controls and connect their signals."""
        super().__init__(parent)
        self._font_loaded = False
        self._busy = False
        self._build_widgets()
        self._build_layout()
        self._connect_signals()

    def _build_widgets(self) -> None:
        """Create every control this panel owns, with its initial state."""
        self.open_button = QPushButton("Open Font...", self)
        self.font_info_label = QLabel("No font loaded", self)
        self.font_info_label.setWordWrap(True)

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

        self.workers_spin = QSpinBox(self)
        self.workers_spin.setRange(0, os.cpu_count() or 1)
        self.workers_spin.setValue(0)
        self.workers_spin.setSpecialValueText("Auto")

        self.save_button = QPushButton("Stencilize && Save...", self)
        self.save_button.setEnabled(False)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.hide()

    def _build_layout(self) -> None:
        """Arrange the widgets built by ``_build_widgets`` into the panel layout."""
        width_row = QHBoxLayout()
        width_label = QLabel("Bridge width", self)
        width_label.setBuddy(self.width_spin)
        width_row.addWidget(width_label)
        width_row.addWidget(self.width_slider, stretch=1)
        width_row.addWidget(self.width_spin)

        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Workers", self))
        workers_row.addWidget(self.workers_spin)

        layout = QVBoxLayout(self)
        layout.addWidget(self.open_button)
        layout.addWidget(self.font_info_label)
        layout.addLayout(width_row)
        layout.addWidget(self.spanning_check)
        layout.addLayout(workers_row)
        layout.addWidget(self.save_button)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Wire widget signals to the panel's forwarding and syncing slots."""
        self.open_button.clicked.connect(self._emit_open_requested)
        self.save_button.clicked.connect(self._emit_save_requested)
        self.width_spin.valueChanged.connect(self.width_slider.setValue)
        self.width_slider.valueChanged.connect(self._sync_width_spin)
        self.spanning_check.toggled.connect(self._emit_parameters_changed)

    def _emit_open_requested(self, _checked: bool = False) -> None:
        """Forward the open button click through the request signal."""
        self.open_requested.emit()

    def _emit_save_requested(self, _checked: bool = False) -> None:
        """Forward the save button click through the request signal."""
        self.save_requested.emit()

    def _emit_parameters_changed(self, _checked: bool) -> None:
        """Forward a spanning-bridge change through the parameter signal."""
        self.parameters_changed.emit()

    def _sync_width_spin(self, value: int) -> None:
        """Keep the spin box in sync and emit one parameter change."""
        blocker = QSignalBlocker(self.width_spin)
        self.width_spin.setValue(value)
        del blocker
        self.parameters_changed.emit()

    def bridge_config(self) -> BridgeConfig:
        """Return the selected bridge parameters."""
        return BridgeConfig(
            width_percent=float(self.width_slider.value()),
            use_spanning_bridges=self.spanning_check.isChecked(),
        )

    def max_workers(self) -> int | None:
        """Return the worker limit, or ``None`` when automatic is selected."""
        value = self.workers_spin.value()
        return None if value == 0 else value

    def set_font_info(self, text: str) -> None:
        """Display information about the currently loaded font."""
        self.font_info_label.setText(text)

    def set_font_loaded(self, loaded: bool) -> None:
        """Track whether a font is loaded and update save availability."""
        self._font_loaded = loaded
        self.save_button.setEnabled(loaded and not self._busy)

    def set_busy(self, busy: bool) -> None:
        """Disable font actions during work and restore their prior state."""
        self._busy = busy
        self.open_button.setEnabled(not busy)
        self.save_button.setEnabled(self._font_loaded and not busy)

    def set_progress(self, completed: int, total: int) -> None:
        """Show progress for the current operation."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_bar.show()

    def reset_progress(self) -> None:
        """Hide the progress bar and reset its value."""
        self.progress_bar.reset()
        self.progress_bar.hide()
