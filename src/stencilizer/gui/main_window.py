"""Top-level application window for the stencilizer desktop GUI."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QSplitter, QWidget

from stencilizer.gui.controller import GuiController
from stencilizer.gui.controls import ControlPanel
from stencilizer.gui.glyph_grid import GlyphGrid
from stencilizer.gui.glyph_view import ComparisonView
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

        self.setWindowTitle("Stencilizer")
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.controls = ControlPanel(splitter)
        self.grid = GlyphGrid(splitter)
        self.comparison = ComparisonView(splitter)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.grid)
        splitter.addWidget(self.comparison)
        self.setCentralWidget(splitter)
        self.statusBar()

        self.controls.open_requested.connect(self.open_font_dialog)
        self.controls.save_requested.connect(self.save_font_dialog)
        self.controls.parameters_changed.connect(self._update_parameters)
        self.controls.workers_spin.valueChanged.connect(self._update_parameters)
        self.grid.glyph_selected.connect(self.controller.select_glyph)
        self.controller.font_loaded.connect(self._on_font_loaded)
        self.controller.preview_ready.connect(self._on_preview_ready)
        self.controller.save_progress.connect(self.controls.set_progress)
        self.controller.busy_changed.connect(self.controls.set_busy)
        self.controller.save_finished.connect(self._on_save_finished)
        self.controller.error.connect(self._on_error)
        self._update_parameters()

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

    def _on_font_loaded(self, result: object) -> None:
        """Populate the window from a newly loaded font session."""
        session = cast("FontSession", result)
        island_glyphs = session.island_glyphs
        self.grid.set_glyphs(island_glyphs, session.ascender, session.descender)
        self.controls.set_font_info(
            f"{session.path.name}\n{session.font_format}, {session.units_per_em} UPM\n"
            f"{session.glyph_count} glyphs, {len(island_glyphs)} with islands"
        )
        self.controls.set_font_loaded(True)
        self.comparison.clear()
        if island_glyphs:
            self.grid.select_glyph(island_glyphs[0].name)
        self.statusBar().showMessage(f"Loaded {session.path.name}")

    def _on_preview_ready(self, result: object) -> None:
        """Show a preview using the current session's vertical metrics."""
        session = self.controller.session
        if session is not None:
            self.comparison.show_preview(
                cast("PreviewResult", result), session.ascender, session.descender
            )

    def _on_save_finished(self, result: object) -> None:
        """Clear progress and describe a completed save in the status bar."""
        self.controls.reset_progress()
        output_path = self._output_path
        if output_path is not None:
            stats = cast("ProcessingStats", result)
            self.statusBar().showMessage(
                f"Saved {output_path.name}: {stats.processed_count} glyphs stencilized, "
                f"{stats.error_count} errors"
            )

    def _on_error(self, message: str) -> None:
        """Clear pending progress and present a controller error to the user."""
        self.controls.reset_progress()
        QMessageBox.warning(self, "Stencilizer", message)
