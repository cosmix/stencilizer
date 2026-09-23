"""Tests for the GUI controller's asynchronous font workflow."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Qt, QThread
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig, GeometryConfig, StencilizerSettings
from stencilizer.core.processor import ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.gui.controller import GuiController
from stencilizer.gui.session import FontSession, PreviewResult
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats
from tests.gui.conftest import LOAD_TIMEOUT, SAVE_TIMEOUT, load_session


class ProgressRecorder(QObject):
    """Collect save progress delivered by the controller."""

    def __init__(self) -> None:
        """Create an empty progress collection."""
        super().__init__()
        self.values: list[tuple[int, int]] = []

    def record(self, completed: int, total: int) -> None:
        """Record one save progress update."""
        self.values.append((completed, total))


class BusyStateRecorder(QObject):
    """Capture the busy state observed by a font-loaded receiver."""

    def __init__(self, controller: GuiController) -> None:
        """Retain the controller whose state will be observed."""
        super().__init__()
        self._controller = controller
        self.states: list[bool] = []

    def record(self, _session: object) -> None:
        """Record the state visible while handling font_loaded."""
        self.states.append(self._controller.is_busy)


class ThreadRecorder(QObject):
    """Record the current thread for outward controller signals."""

    def __init__(self) -> None:
        """Create empty per-signal and combined thread collections."""
        super().__init__()
        self.threads: list[QThread] = []
        self.loaded_threads: list[QThread] = []
        self.progress_threads: list[QThread] = []
        self.finished_threads: list[QThread] = []

    def record_loaded(self, _session: object) -> None:
        """Record the thread that emits font_loaded."""
        thread = QThread.currentThread()
        self.loaded_threads.append(thread)
        self.threads.append(thread)

    def record_progress(self, _completed: int, _total: int) -> None:
        """Record the thread that emits save_progress."""
        thread = QThread.currentThread()
        self.progress_threads.append(thread)
        self.threads.append(thread)

    def record_finished(self, _stats: object) -> None:
        """Record the thread that emits save_finished."""
        thread = QThread.currentThread()
        self.finished_threads.append(thread)
        self.threads.append(thread)


def _saved_glyph(path: Path, name: str) -> Glyph:
    """Read one glyph from a saved output font."""
    with FontReader(path) as reader:
        glyph = reader.get_glyph(name)
    assert glyph is not None
    return glyph


def test_open_font_loads_session_and_toggles_busy(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Loading Roboto publishes its session after the busy state clears."""
    busy_values: list[bool] = []
    controller.busy_changed.connect(busy_values.append)

    session = load_session(controller, qtbot, roboto_path)

    assert len(session.island_glyphs) == 562
    assert busy_values == [True, False]
    assert controller.is_busy is False


def test_open_font_busy_guard(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """A second mutation while loading fails synchronously without starting work."""
    loaded: list[FontSession] = []
    controller.font_loaded.connect(loaded.append)
    with qtbot.waitSignal(controller.error) as blocker:
        controller.open_font(roboto_path)
        controller.open_font(roboto_path)
    assert blocker.args == ["Busy: wait for the current operation to finish"]
    qtbot.waitUntil(lambda: len(loaded) == 1, timeout=LOAD_TIMEOUT)

    output_path = tmp_path / "x.ttf"
    with qtbot.waitSignal(controller.error) as save_blocker:
        controller.open_font(roboto_path)
        controller.save(output_path)
    assert save_blocker.args == ["Busy: wait for the current operation to finish"]
    assert not output_path.exists()
    qtbot.waitUntil(lambda: len(loaded) == 2, timeout=LOAD_TIMEOUT)


def test_select_glyph_publishes_preview(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Selecting O synchronously publishes its stenciled preview."""
    load_session(controller, qtbot, roboto_path)

    with qtbot.waitSignal(controller.preview_ready) as blocker:
        controller.select_glyph("O")

    result = blocker.args[0]
    assert isinstance(result, PreviewResult)
    assert result.bridges_added == 1


def test_parameters_refresh_preview_with_new_width(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Changing bridge width refreshes the selected preview with new geometry."""
    load_session(controller, qtbot, roboto_path)
    controller.select_glyph("O")
    with qtbot.waitSignal(controller.preview_ready) as narrow_blocker:
        controller.set_parameters(BridgeConfig(width_percent=30.0), None)
    with qtbot.waitSignal(controller.preview_ready) as wide_blocker:
        controller.set_parameters(BridgeConfig(width_percent=110.0), None)

    narrow = narrow_blocker.args[0]
    wide = wide_blocker.args[0]
    assert isinstance(narrow, PreviewResult)
    assert isinstance(wide, PreviewResult)
    assert narrow.stenciled is not None
    assert wide.stenciled is not None
    assert narrow.stenciled.to_dict() != wide.stenciled.to_dict()


def test_parameters_refresh_preview_with_spanning_mode(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Changing spanning-bridge mode refreshes B with a different outline."""
    load_session(controller, qtbot, roboto_path)
    controller.select_glyph("B")
    with qtbot.waitSignal(controller.preview_ready) as spanning_blocker:
        controller.set_parameters(BridgeConfig(use_spanning_bridges=True), None)
    with qtbot.waitSignal(controller.preview_ready) as split_blocker:
        controller.set_parameters(BridgeConfig(use_spanning_bridges=False), None)

    spanning = spanning_blocker.args[0]
    split = split_blocker.args[0]
    assert isinstance(spanning, PreviewResult)
    assert isinstance(split, PreviewResult)
    assert spanning.stenciled is not None
    assert split.stenciled is not None
    assert spanning.stenciled.to_dict() != split.stenciled.to_dict()


def test_parameters_without_selection_do_not_publish_preview(
    controller: GuiController, qtbot: QtBot
) -> None:
    """Storing parameters alone does not create a preview."""
    with qtbot.assertNotEmitted(controller.preview_ready):
        controller.set_parameters(BridgeConfig(width_percent=30.0), None)


def test_save_writes_previewed_font(
    controller: GuiController,
    qtbot: QtBot,
    roboto_path: Path,
    tmp_path: Path,
    outlines_match: Callable[[Glyph, Glyph], bool],
) -> None:
    """Saving emits progress and writes outlines matching the synchronous preview."""
    load_session(controller, qtbot, roboto_path)
    controller.set_parameters(BridgeConfig(), 1)
    progress = ProgressRecorder()
    controller.save_progress.connect(progress.record)
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        controller.save(output_path)

    stats = blocker.args[0]
    assert isinstance(stats, ProcessingStats)
    assert stats.processed_count == 562
    assert stats.error_count == 0
    assert any(total == 562 for _, total in progress.values)
    session = controller.session
    assert session is not None
    expected = session.preview("O", BridgeConfig(), GeometryConfig())
    assert expected.stenciled is not None
    assert outlines_match(_saved_glyph(output_path, "O"), expected.stenciled)


def test_shutdown_waits_for_active_save(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Shutdown waits for an in-flight save without leaving partial output behind."""
    load_session(controller, qtbot, roboto_path)
    controller.set_parameters(BridgeConfig(), 1)
    output_path = tmp_path / "s.ttf"
    controller.save(output_path)
    controller.shutdown()

    assert output_path.exists()
    assert len(_saved_glyph(output_path, "O").contours) == 4
    with qtbot.waitSignal(controller.save_finished, timeout=LOAD_TIMEOUT) as blocker:
        pass
    stats = blocker.args[0]
    assert isinstance(stats, ProcessingStats)
    assert stats.error_count == 0
    assert controller.is_busy is False


def test_font_loaded_observes_idle_controller(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """font_loaded receivers observe the task released before notification."""
    recorder = BusyStateRecorder(controller)
    controller.font_loaded.connect(recorder.record)

    load_session(controller, qtbot, roboto_path)

    assert recorder.states == [False]


def test_signals_delivered_on_gui_thread(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Outward load and save signals originate from the controller's GUI thread."""
    recorder = ThreadRecorder()
    connection = Qt.ConnectionType.DirectConnection
    controller.font_loaded.connect(recorder.record_loaded, connection)
    controller.save_progress.connect(recorder.record_progress, connection)
    controller.save_finished.connect(recorder.record_finished, connection)

    load_session(controller, qtbot, roboto_path)
    controller.set_parameters(BridgeConfig(), 1)
    with qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT):
        controller.save(tmp_path / "out.ttf")

    assert recorder.loaded_threads
    assert recorder.progress_threads
    assert recorder.finished_threads
    assert recorder.threads
    assert all(thread == controller.thread() for thread in recorder.threads)


def test_save_uses_current_parameters(
    controller: GuiController,
    qtbot: QtBot,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings built for a save retain the controller's current parameters."""
    load_session(controller, qtbot, roboto_path)
    captured: list[StencilizerSettings] = []

    def recorder(
        _session: FontSession,
        _output_path: Path,
        settings: StencilizerSettings,
        _progress: ProgressCallback | None = None,
    ) -> ProcessingStats:
        """Record the controller-created settings and finish immediately."""
        captured.append(settings)
        return ProcessingStats()

    monkeypatch.setattr(FontSession, "save", recorder)
    controller.set_parameters(BridgeConfig(width_percent=30.0), 1)
    with qtbot.waitSignal(controller.save_finished, timeout=LOAD_TIMEOUT):
        controller.save(tmp_path / "p.ttf")

    assert captured[0].bridge.width_percent == 30.0
    assert captured[0].processing.max_workers == 1


def test_single_processor_per_controller(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Loads and saves reuse the processor created with the controller."""
    initial_handler_count = len(logging.getLogger().handlers)
    load_session(controller, qtbot, roboto_path)
    controller.set_parameters(BridgeConfig(), 1)
    with qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT):
        controller.save(tmp_path / "out.ttf")
    load_session(controller, qtbot, roboto_path)

    assert len(logging.getLogger().handlers) == initial_handler_count
