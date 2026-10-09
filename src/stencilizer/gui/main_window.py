"""Top-level application window for the stencilizer desktop GUI."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from stencilizer.config.settings import BridgeDirection
from stencilizer.gui.controller import GuiController
from stencilizer.gui.controls import ControlPanel
from stencilizer.gui.direction_picker import DirectionPicker
from stencilizer.gui.glyph_grid import GlyphGrid
from stencilizer.gui.glyph_view import ComparisonView
from stencilizer.gui.header import HeaderBar
from stencilizer.io.writer import FontWriter

if TYPE_CHECKING:
    from stencilizer.gui.session import FontSession, PreviewResult
    from stencilizer.utils import ProcessingStats


class MainWindow(QMainWindow):
    """Top-level window: controls | glyph grid | before/after comparison."""

    def __init__(self, controller: GuiController, parent: QWidget | None = None) -> None:
        """Create the window, its three panes, and controller signal wiring."""
        super().__init__(parent)
        self.controller = controller
        self._output_path: Path | None = None
        self._current_glyph: str | None = None
        self.setWindowTitle("Stencilizer")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)
        self._build_panes()
        self._build_status_bar()
        self._connect_signals()
        self._connect_direction_signals()
        self._update_parameters()

    def _build_panes(self) -> None:
        """Create the header and three-pane main content area."""
        self.header = HeaderBar()
        self.controls = ControlPanel()
        self.grid = GlyphGrid()
        self.empty_state = QLabel("Open a font to see the glyphs that need bridges")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setWordWrap(True)
        self.grid_stack = QStackedWidget()
        self.grid_stack.addWidget(self.empty_state)
        self.grid_stack.addWidget(self.grid)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.grid_stack)
        splitter.addWidget(self._build_preview_pane())
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 520, 460])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _build_preview_pane(self) -> QWidget:
        """Create the comparison and bridge-direction controls pane."""
        right_pane = QWidget()
        right_pane.setObjectName("previewPane")
        right_pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)
        self.comparison = ComparisonView(right_pane)
        picker_card = QFrame(right_pane)
        picker_card.setProperty("role", "card")
        picker_layout = QVBoxLayout(picker_card)
        picker_layout.setContentsMargins(4, 4, 4, 4)
        self.direction_picker = DirectionPicker(picker_card)
        picker_layout.addWidget(self.direction_picker)
        right_layout.addWidget(self.comparison, 1)
        right_layout.addWidget(picker_card)
        return right_pane

    def _build_status_bar(self) -> None:
        """Add the save progress percentage and indicator to the status bar."""
        self.progress_label = QLabel()
        self.progress_label.setProperty("role", "status")
        self.progress_label.hide()
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("saveProgress")
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_label)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _connect_signals(self) -> None:
        """Connect pane actions and controller updates."""
        self.header.open_requested.connect(self.open_font_dialog)
        self.header.save_requested.connect(self.save_font_dialog)
        self.controls.parameters_changed.connect(self._update_parameters)
        self.controls.workers_slider.valueChanged.connect(self._update_parameters)
        self.grid.glyph_selected.connect(self.controller.select_glyph)
        self.controls.axes_panel.location_changed.connect(self.controller.set_location)
        self.controller.font_loaded.connect(self._on_font_loaded)
        self.controller.preview_ready.connect(self._on_preview_ready)
        self.controller.save_progress.connect(self.set_progress)
        self.controller.busy_changed.connect(self.header.set_busy)
        self.controller.save_finished.connect(self._on_save_finished)
        self.controller.error.connect(self._on_error)

    def _connect_direction_signals(self) -> None:
        """Connect glyph-direction controls to the controller and grid."""
        self.grid.glyph_selected.connect(self._on_glyph_selected)
        self.direction_picker.direction_chosen.connect(self._on_direction_chosen)
        self.controller.direction_changed.connect(self._on_direction_changed)
        self.controller.unbridged_changed.connect(self._on_unbridged_changed)

    def load_font(self, path: Path) -> None:
        """Ask the controller to load a font from ``path``."""
        self.controller.open_font(path)

    def save_font(self, path: Path) -> None:
        """Ask the controller to save; remember ``path`` only when not already busy."""
        if not self.controller.is_busy:
            self._output_path = path
        self.controller.save(path)

    def open_font_dialog(self) -> None:
        """Open a font chooser and load its selected font."""
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Open Font", "", "Fonts (*.ttf *.otf)"
        )
        if selected_path:
            self.load_font(Path(selected_path))

    def save_font_dialog(self) -> None:
        """Open a save chooser when a font session is available."""
        session = self.controller.session
        if session is None:
            return
        default_path = FontWriter.get_stenciled_path(session.path)
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Stenciled Font",
            str(default_path),
            "Fonts (*.ttf *.otf)",
        )
        if selected_path:
            self.save_font(Path(selected_path))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Keep the window open while its controller has outstanding work."""
        if self.controller.is_busy:
            event.ignore()
            self.statusBar().showMessage("Wait for the current operation to finish")
            return
        self.controller.shutdown()
        event.accept()

    def _update_parameters(self) -> None:
        """Send the control panel's selected processing parameters to the controller."""
        self.controller.set_parameters(self.controls.bridge_config(), self.controls.max_workers())

    def set_progress(self, completed: int, total: int) -> None:
        """Show save progress for the completed portion of the glyph set."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_label.setText(self.progress_bar.text())
        self.progress_label.show()
        self.progress_bar.show()

    def reset_progress(self) -> None:
        """Clear and hide the save progress percentage and indicator."""
        self.progress_bar.reset()
        self.progress_bar.hide()
        self.progress_label.clear()
        self.progress_label.hide()

    def _on_font_loaded(self, result: object) -> None:
        """Populate the window from a newly loaded font session."""
        session = cast("FontSession", result)
        self.grid.set_glyphs(session.display_glyphs, session.ascender, session.descender)
        self.controls.axes_panel.set_axes(session.axes)
        self.header.set_font_info(
            session.path.name,
            f"{session.font_format} · {session.units_per_em} UPM · {session.glyph_count} glyphs · "
            f"{len(session.island_glyphs)} with islands · {len(session.composites)} composites",
        )
        self.header.set_font_loaded(True)
        self.grid_stack.setCurrentWidget(self.grid)
        self.comparison.clear()
        self.direction_picker.clear()
        self._current_glyph = None
        if session.display_names:
            self.grid.select_glyph(session.display_names[0])
        self.statusBar().showMessage(f"Loaded {session.path.name}")

    def _on_glyph_selected(self, name: str) -> None:
        """Show bridge-direction controls for the selected displayed glyph."""
        session = self.controller.session
        if session is None:
            return
        self._current_glyph = name
        self.controls.axes_panel.set_location_applies(not session.is_composite(name))
        sources = session.direction_sources(name)
        direction = self.controller.direction_for(sources[0]) if sources else BridgeDirection.AUTO
        self.direction_picker.show_for(name, sources, direction)

    def _on_direction_chosen(self, value: str) -> None:
        """Apply a user's selected bridge direction to the current glyph."""
        if self._current_glyph is not None:
            self.controller.set_direction(self._current_glyph, BridgeDirection(value))

    def _on_direction_changed(self, name: str, value: str) -> None:
        """Update the direction marker and picker after a direction change."""
        self.grid.set_direction_marker(name, BridgeDirection(value))
        if name == self._current_glyph:
            self._on_glyph_selected(name)

    def _on_unbridged_changed(self, names: object) -> None:
        """Mark displayed glyphs for which bridge placement failed."""
        self.grid.set_unbridged(cast("frozenset[str]", names))

    def _on_preview_ready(self, result: object) -> None:
        """Show a preview using the current session's vertical metrics."""
        session = self.controller.session
        if session is not None:
            self.comparison.show_preview(
                cast("PreviewResult", result), session.ascender, session.descender
            )

    def _on_save_finished(self, result: object) -> None:
        """Clear progress and describe a completed save in the status bar."""
        self.reset_progress()
        output_path = self._output_path
        if output_path is not None:
            stats = cast("ProcessingStats", result)
            self.statusBar().showMessage(
                f"Saved {output_path.name}: {stats.processed_count} glyphs stencilized, "
                f"{stats.error_count} errors"
            )

    def _on_error(self, message: str) -> None:
        """Clear pending progress and present a controller error to the user."""
        self.reset_progress()
        QMessageBox.warning(self, "Stencilizer", message)
