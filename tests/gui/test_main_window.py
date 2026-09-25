"""Tests for the top-level Stencilizer GUI window."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.domain import Glyph, GlyphMetadata
from stencilizer.gui.controller import GuiController
from stencilizer.gui.main_window import MainWindow
from stencilizer.gui.session import PreviewResult
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats

LOAD_TIMEOUT = 30_000
SAVE_TIMEOUT = 120_000


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot) -> Iterator[MainWindow]:
    """Create a window backed by a controller that is shut down after each test."""
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _load_font(window: MainWindow, qtbot: QtBot, path: Path) -> None:
    """Load ``path`` and wait until the automatic initial preview is visible."""
    with qtbot.waitSignal(window.controller.font_loaded, timeout=LOAD_TIMEOUT):
        window.load_font(path)
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)


def _saved_glyph(path: Path, name: str) -> Glyph:
    """Read one saved glyph while asserting the output contains it."""
    with FontReader(path) as reader:
        glyph = reader.get_glyph(name)
    assert glyph is not None
    return glyph


def _assert_roboto_save(stats: object, output_path: Path) -> None:
    """Check a successful complete Roboto save and reopen its stencilized outline."""
    assert isinstance(stats, ProcessingStats)
    assert stats.processed_count == 562
    assert stats.error_count == 0
    assert len(_saved_glyph(output_path, "O").contours) == 4


def _record_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the modal warning box with a recorder and return its messages."""
    messages: list[str] = []

    def record_warning(_parent: QMainWindow, _title: str, message: str) -> None:
        """Collect a warning instead of showing a blocking modal dialog."""
        messages.append(message)

    monkeypatch.setattr(QMessageBox, "warning", record_warning)
    return messages


def test_load_font_populates_window_and_selects_first_glyph(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """Loading Roboto enables saving and previews the first island glyph."""
    _load_font(window, qtbot, roboto_path)

    current_item = window.grid.currentItem()
    assert window.grid.count() == 1027
    assert window.header.save_button.isEnabled()
    assert current_item is not None
    assert current_item.text() == ".notdef"
    assert window.comparison.after_canvas.glyph is not None


def test_width_changes_refresh_selected_preview(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """Changing the bridge-width slider produces a new preview for O."""
    _load_font(window, qtbot, roboto_path)
    assert window.grid.select_glyph("O")

    window.controls.width_slider.setValue(30)
    narrow = window.comparison.after_canvas.glyph
    assert narrow is not None
    narrow_outline = narrow.to_dict()

    window.controls.width_slider.setValue(110)
    wide = window.comparison.after_canvas.glyph
    assert wide is not None
    assert wide.to_dict() != narrow_outline


def test_save_font_reports_completed_stencilization(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Saving reports the complete Roboto outcome in the status bar."""
    _load_font(window, qtbot, roboto_path)
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font(output_path)

    _assert_roboto_save(blocker.args[0], output_path)
    assert (
        window.statusBar()
        .currentMessage()
        .startswith("Saved out.ttf: 562 glyphs stencilized, 0 errors")
    )


def test_saved_font_matches_preview(
    window: MainWindow,
    qtbot: QtBot,
    roboto_path: Path,
    tmp_path: Path,
    outlines_match: Callable[[Glyph, Glyph], bool],
) -> None:
    """Saving with a 30-percent bridge writes the previewed O outline."""
    _load_font(window, qtbot, roboto_path)
    assert window.grid.select_glyph("O")
    window.controls.workers_slider.setValue(1)
    window.controls.width_slider.setValue(30)
    expected = window.comparison.after_canvas.glyph
    assert expected is not None
    output_path = tmp_path / "w30.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font(output_path)

    _assert_roboto_save(blocker.args[0], output_path)
    assert outlines_match(_saved_glyph(output_path, "O"), expected)


def test_second_save_while_busy_keeps_first_output_path(
    window: MainWindow,
    qtbot: QtBot,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save requested while another is in flight does not steal the status path."""
    _load_font(window, qtbot, roboto_path)
    window.controls.workers_slider.setValue(1)
    messages = _record_warnings(monkeypatch)
    path_a = tmp_path / "a.ttf"
    path_b = tmp_path / "b.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT):
        window.save_font(path_a)
        window.save_font(path_b)

    assert len(messages) == 1
    assert messages[0].startswith("Busy")
    assert window.statusBar().currentMessage().startswith("Saved a.ttf")
    assert not path_b.exists()
    assert path_a.exists()


def test_save_finished_without_requested_save_does_not_report_saved(window: MainWindow) -> None:
    """A save_finished signal a window save call never triggered leaves the status untouched."""
    window.controller.save_finished.emit(ProcessingStats())

    assert not window.statusBar().currentMessage().startswith("Saved")


def test_preview_ready_without_session_leaves_comparison_empty(window: MainWindow) -> None:
    """A preview_ready signal without a loaded session leaves the after canvas empty."""
    original = Glyph(GlyphMetadata("empty", None, 100, 0), [])
    result = PreviewResult(
        glyph_name="empty",
        original=original,
        stenciled=None,
        bridges_added=0,
        error=None,
        duration_ms=0.0,
    )

    window.controller.preview_ready.emit(result)

    assert window.comparison.after_canvas.glyph is None


def test_invalid_font_shows_load_warning(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable font presents the controller's load error in a warning."""
    messages = _record_warnings(monkeypatch)
    invalid_path = tmp_path / "invalid.ttf"
    invalid_path.write_bytes(b"not a font")

    window.load_font(invalid_path)
    qtbot.waitUntil(lambda: bool(messages), timeout=LOAD_TIMEOUT)

    assert messages[0].startswith("Failed to load font")


def test_unsupported_font_is_rejected(
    window: MainWindow,
    qtbot: QtBot,
    cff2_font_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CFF2 input is rejected before it can become a saveable session."""
    messages = _record_warnings(monkeypatch)

    window.load_font(cff2_font_path)
    qtbot.waitUntil(lambda: bool(messages), timeout=LOAD_TIMEOUT)

    assert messages[0].startswith("Failed to load font")
    assert "CFF2 outlines are not supported" in messages[0]
    assert window.controller.session is None
    assert window.grid.count() == 0
    assert not window.header.save_button.isEnabled()


def test_open_font_dialog_loads_selection_and_ignores_cancel(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open dialog loads its selected path and does nothing on cancellation."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(roboto_path), ""),
    )
    with qtbot.waitSignal(window.controller.font_loaded, timeout=LOAD_TIMEOUT):
        window.open_font_dialog()
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)

    busy_values: list[bool] = []
    window.controller.busy_changed.connect(busy_values.append)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    with qtbot.assertNotEmitted(window.controller.busy_changed):
        window.open_font_dialog()

    assert not busy_values


def test_save_font_dialog_saves_selected_path(
    window: MainWindow,
    qtbot: QtBot,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save dialog forwards its selected path to the saving workflow."""
    _load_font(window, qtbot, roboto_path)
    window.controls.workers_slider.setValue(1)
    output_path = tmp_path / "dialog.ttf"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args: (str(output_path), ""),
    )

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font_dialog()

    _assert_roboto_save(blocker.args[0], output_path)


def test_save_font_dialog_without_session_does_nothing(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The save dialog guard skips the file chooser entirely when nothing is loaded."""
    calls: list[tuple[object, ...]] = []

    def record_dialog(*args: object) -> tuple[str, str]:
        """Record a getSaveFileName call that should never happen."""
        calls.append(args)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", record_dialog)
    busy_values: list[bool] = []
    window.controller.busy_changed.connect(busy_values.append)

    window.save_font_dialog()

    assert calls == []
    assert busy_values == []


def test_save_font_dialog_cancelled_starts_no_save(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the save dialog after a load starts no background save."""
    _load_font(window, qtbot, roboto_path)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: ("", ""))

    with qtbot.assertNotEmitted(window.controller.save_finished):
        window.save_font_dialog()

    assert window.controller.is_busy is False


def test_close_after_load_returns_without_hanging(
    window: MainWindow, qtbot: QtBot, roboto_path: Path
) -> None:
    """A loaded idle window closes cleanly through the controller shutdown path."""
    _load_font(window, qtbot, roboto_path)

    assert window.close()


def test_save_busy_state_disables_and_restores_actions(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """A save immediately disables actions and restores them after completion."""
    _load_font(window, qtbot, roboto_path)
    window.controls.workers_slider.setValue(1)
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font(output_path)
        assert not window.header.save_button.isEnabled()
        assert not window.header.open_button.isEnabled()

    _assert_roboto_save(blocker.args[0], output_path)
    assert window.header.save_button.isEnabled()
    assert window.header.open_button.isEnabled()
    assert not window.progress_bar.isVisibleTo(window)


def test_close_while_busy_is_refused(
    window: MainWindow, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Closing during a save is refused until the asynchronous work finishes."""
    _load_font(window, qtbot, roboto_path)
    window.controls.workers_slider.setValue(1)
    window.show()
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.save_font(output_path)
        assert not window.close()
        assert window.statusBar().currentMessage() == "Wait for the current operation to finish"

    _assert_roboto_save(blocker.args[0], output_path)
    assert window.close()


def test_workers_slider_updates_controller_parameters(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the worker limit reaches the controller with that limit."""
    calls: list[tuple[BridgeConfig, int | None]] = []

    def recorder(bridge: BridgeConfig, max_workers: int | None) -> None:
        """Record parameters the control panel sends to the controller."""
        calls.append((bridge, max_workers))

    monkeypatch.setattr(window.controller, "set_parameters", recorder)
    window.controls.workers_slider.setValue(1)

    assert calls[-1][1] == 1
