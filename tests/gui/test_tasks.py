"""Tests for GUI background tasks."""

from PySide6.QtCore import QObject, Qt, QThreadPool
from pytestqt.qtbot import QtBot

from stencilizer.exceptions import StencilizerError
from stencilizer.gui.tasks import BackgroundTask, ProgressFn


class ProgressReceiver(QObject):
    """Collect task progress delivered on the GUI thread."""

    def __init__(self) -> None:
        """Create an empty progress-event collection."""
        super().__init__()
        self.values: list[tuple[int, int]] = []

    def receive(self, completed: int, total: int) -> None:
        """Record one progress event."""
        self.values.append((completed, total))


def test_task_emits_finished_for_success(qtbot: QtBot) -> None:
    """A successful task reports its return value without reporting a failure."""
    pool = QThreadPool()
    task = BackgroundTask(lambda _progress: 42)

    try:
        with (
            qtbot.assertNotEmitted(task.signals.failed),
            qtbot.waitSignal(task.signals.finished, timeout=1_000) as blocker,
        ):
            pool.start(task)
        assert blocker.args == [42]
    finally:
        pool.waitForDone()


def test_task_emits_progress_in_order(qtbot: QtBot) -> None:
    """Progress events are queued to the GUI thread in their emitted order."""

    def work(progress: ProgressFn) -> object:
        progress(1, 3)
        progress(3, 3)
        return None

    pool = QThreadPool()
    task = BackgroundTask(work)
    receiver = ProgressReceiver()
    task.signals.progress.connect(receiver.receive, Qt.ConnectionType.QueuedConnection)

    try:
        with qtbot.waitSignal(task.signals.finished, timeout=1_000):
            pool.start(task)
        qtbot.waitUntil(lambda: len(receiver.values) == 2, timeout=1_000)
        assert receiver.values == [(1, 3), (3, 3)]
    finally:
        pool.waitForDone()


def test_task_emits_stencilizer_error(qtbot: QtBot) -> None:
    """A StencilizerError becomes its user-facing error message."""

    def work(_: ProgressFn) -> object:
        raise StencilizerError("bad input")

    pool = QThreadPool()
    task = BackgroundTask(work)

    try:
        with (
            qtbot.assertNotEmitted(task.signals.finished),
            qtbot.waitSignal(task.signals.failed, timeout=1_000) as blocker,
        ):
            pool.start(task)
        assert blocker.args == ["bad input"]
    finally:
        pool.waitForDone()


def test_task_emits_unexpected_error(qtbot: QtBot) -> None:
    """An unexpected exception gains the standard user-facing prefix."""

    def work(_: ProgressFn) -> object:
        raise ValueError("boom")

    pool = QThreadPool()
    task = BackgroundTask(work)

    try:
        with (
            qtbot.assertNotEmitted(task.signals.finished),
            qtbot.waitSignal(task.signals.failed, timeout=1_000) as blocker,
        ):
            pool.start(task)
        assert blocker.args == ["Unexpected error: boom"]
    finally:
        pool.waitForDone()


def test_task_disables_auto_delete() -> None:
    """The controller retains ownership of a task until it handles completion."""
    pool = QThreadPool()
    task = BackgroundTask(lambda _progress: None)

    try:
        assert task.autoDelete() is False
    finally:
        pool.waitForDone()
