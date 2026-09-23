"""Coordinate font loading, previews, and saves for the desktop GUI."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QObject, Qt, QThreadPool, Signal

from stencilizer.config import (
    BridgeConfig,
    GeometryConfig,
    LoggingConfig,
    ProcessingConfig,
    StencilizerSettings,
)
from stencilizer.core import FontProcessor
from stencilizer.exceptions import StencilizerError
from stencilizer.gui.session import FontSession
from stencilizer.gui.tasks import BackgroundTask, ProgressFn

if TYPE_CHECKING:
    from stencilizer.utils import ProcessingStats


class GuiController(QObject):
    """Own the font session, the worker pool, and the current parameters."""

    font_loaded = Signal(object)
    preview_ready = Signal(object)
    save_progress = Signal(int, int)
    save_finished = Signal(object)
    error = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, log_file: Path, parent: QObject | None = None) -> None:
        """Create one processor and a private pool for this controller's tasks."""
        super().__init__(parent)
        settings = StencilizerSettings(
            logging=LoggingConfig(
                log_file=log_file,
                log_level="WARNING",
                file_log_level="INFO",
            )
        )
        self._processor = FontProcessor(settings)
        self._pool = QThreadPool(self)
        self._task: BackgroundTask | None = None
        self._session: FontSession | None = None
        self._selected_glyph: str | None = None
        self._bridge: BridgeConfig = BridgeConfig()
        self._max_workers: int | None = None

    @property
    def session(self) -> FontSession | None:
        """Return the currently loaded font session, if any."""
        return self._session

    @property
    def is_busy(self) -> bool:
        """Report whether a load or save task is awaiting its outcome."""
        return self._task is not None

    def _start(self, task: BackgroundTask, on_finished: Callable[[object], None]) -> None:
        """Connect a task to GUI-thread handlers before submitting it."""
        connection = Qt.ConnectionType.QueuedConnection
        task.signals.finished.connect(on_finished, connection)
        task.signals.failed.connect(self._on_failed, connection)
        task.signals.progress.connect(self._on_progress, connection)
        self._task = task
        self.busy_changed.emit(True)
        self._pool.start(task)

    def _finish_task(self) -> None:
        """Release the completed task before notifying observers."""
        self._task = None
        self.busy_changed.emit(False)

    def _on_font_loaded(self, result: object) -> None:
        """Publish a loaded session after clearing the busy state."""
        session = cast("FontSession", result)
        self._session = session
        self._selected_glyph = None
        self._finish_task()
        self.font_loaded.emit(session)

    def _on_save_finished(self, result: object) -> None:
        """Publish save statistics after clearing the busy state."""
        self._finish_task()
        self.save_finished.emit(cast("ProcessingStats", result))

    def _on_failed(self, message: str) -> None:
        """Publish a background task's user-facing error."""
        self._finish_task()
        self.error.emit(message)

    def _on_progress(self, completed: int, total: int) -> None:
        """Forward save progress from the worker to GUI observers."""
        self.save_progress.emit(completed, total)

    def open_font(self, path: Path) -> None:
        """Load and classify a font in this controller's worker pool."""
        if self.is_busy:
            self.error.emit("Busy: wait for the current operation to finish")
            return

        def load(_progress: ProgressFn) -> object:
            return FontSession.open(path, self._processor)

        self._start(BackgroundTask(load), self._on_font_loaded)

    def set_parameters(self, bridge: BridgeConfig, max_workers: int | None) -> None:
        """Store processing parameters and refresh the selected glyph."""
        self._bridge = bridge
        self._max_workers = max_workers
        self._refresh_preview()

    def select_glyph(self, name: str) -> None:
        """Select a glyph and refresh its preview."""
        self._selected_glyph = name
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Render the selected glyph synchronously when a font is loaded."""
        if self._session is None or self._selected_glyph is None:
            return
        try:
            result = self._session.preview(self._selected_glyph, self._bridge, GeometryConfig())
        except StencilizerError as error:
            self.error.emit(str(error))
        else:
            self.preview_ready.emit(result)

    def save(self, output_path: Path) -> None:
        """Save the loaded font using the current parameters."""
        if self.is_busy:
            self.error.emit("Busy: wait for the current operation to finish")
            return
        session = self._session
        if session is None:
            self.error.emit("No font loaded")
            return
        settings = StencilizerSettings(
            bridge=self._bridge,
            processing=ProcessingConfig(max_workers=self._max_workers),
            logging=self._processor.config.logging,
        )

        def save_font(progress: ProgressFn) -> object:
            def report(completed: int, total: int, _name: str, _success: bool) -> None:
                progress(completed, total)

            return session.save(output_path, settings, report)

        self._start(BackgroundTask(save_font), self._on_save_finished)

    def shutdown(self) -> None:
        """Wait for this controller's active pool work to finish."""
        self._pool.waitForDone()
