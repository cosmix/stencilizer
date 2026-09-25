"""Background work runners for the desktop GUI."""

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from stencilizer.exceptions import StencilizerError

ProgressFn = Callable[[int, int], None]


class TaskSignals(QObject):
    """Signals a BackgroundTask emits from its pool thread."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)


class BackgroundTask(QRunnable):
    """Run one work function on a QThreadPool thread."""

    def __init__(self, work: Callable[[ProgressFn], object]) -> None:
        """Create a runnable that reports work outcomes through its signals."""
        super().__init__()
        self.signals = TaskSignals()
        self._work = work
        self.setAutoDelete(False)

    def run(self) -> None:
        """Run work and translate errors into user-facing signals."""
        try:
            result = self._work(self.signals.progress.emit)
        except StencilizerError as error:
            self.signals.failed.emit(str(error))
        except Exception as error:
            self.signals.failed.emit(f"Unexpected error: {error}")
        else:
            self.signals.finished.emit(result)
