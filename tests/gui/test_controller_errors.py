"""Tests for the GUI controller's error and busy-guard paths."""

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from pytestqt.qtbot import QtBot

from stencilizer.gui.controller import GuiController
from tests.gui.conftest import LOAD_TIMEOUT, SAVE_TIMEOUT, load_session


def test_open_font_reports_invalid_file(
    controller: GuiController, qtbot: QtBot, tmp_path: Path
) -> None:
    """An invalid font file produces a load error without a session."""
    invalid_path = tmp_path / "invalid.ttf"
    invalid_path.write_bytes(b"not a font")

    with qtbot.waitSignal(controller.error, timeout=LOAD_TIMEOUT) as blocker:
        controller.open_font(invalid_path)

    assert blocker.args[0].startswith("Failed to load font")
    assert controller.session is None
    assert controller.is_busy is False


def test_open_font_rejects_cff2(
    controller: GuiController, qtbot: QtBot, cff2_font_path: Path
) -> None:
    """CFF2 fonts are rejected before a session can be published."""
    with (
        qtbot.assertNotEmitted(controller.font_loaded),
        qtbot.waitSignal(controller.error, timeout=LOAD_TIMEOUT) as blocker,
    ):
        controller.open_font(cff2_font_path)

    assert blocker.args[0].startswith("Failed to load font")
    assert "CFF2 outlines are not supported" in blocker.args[0]
    assert controller.session is None
    assert controller.is_busy is False


def test_select_missing_glyph_emits_error_without_raising(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Selecting a name absent from the font reports an error instead of raising."""
    load_session(controller, qtbot, roboto_path)

    with qtbot.waitSignal(controller.error) as blocker:
        controller.select_glyph("does-not-exist")

    assert "does-not-exist" in blocker.args[0]


def test_save_requires_loaded_font(controller: GuiController, qtbot: QtBot, tmp_path: Path) -> None:
    """Saving before a load reports the controller's explicit error."""
    with qtbot.waitSignal(controller.error) as blocker:
        controller.save(tmp_path / "out.ttf")
    assert blocker.args == ["No font loaded"]


def test_save_refuses_loaded_font_path(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """The controller preserves a loaded source when saving onto its own path."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    original_bytes = source.read_bytes()
    load_session(controller, qtbot, source)

    with qtbot.waitSignal(controller.error, timeout=SAVE_TIMEOUT) as blocker:
        controller.save(source)

    assert blocker.args[0].startswith("Failed to save font")
    assert source.read_bytes() == original_bytes


def test_failure_signals_delivered_on_gui_thread(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """Outward error and busy signals originate from the controller's GUI thread on failure."""
    error_on_gui_thread: list[bool] = []
    busy_on_gui_thread: list[bool] = []
    connection = Qt.ConnectionType.DirectConnection

    def check_error(_message: str) -> None:
        """Record, at emission time, whether error fired on the controller's own thread."""
        error_on_gui_thread.append(QThread.currentThread() == controller.thread())

    def check_busy(_busy: bool) -> None:
        """Record, at emission time, whether busy_changed fired on the controller's own thread."""
        busy_on_gui_thread.append(QThread.currentThread() == controller.thread())

    controller.error.connect(check_error, connection)
    controller.busy_changed.connect(check_busy, connection)

    invalid_path = tmp_path / "invalid.ttf"
    invalid_path.write_bytes(b"not a font")
    with qtbot.waitSignal(controller.error, timeout=LOAD_TIMEOUT):
        controller.open_font(invalid_path)

    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    load_session(controller, qtbot, source)
    with qtbot.waitSignal(controller.error, timeout=SAVE_TIMEOUT):
        controller.save(source)

    assert len(error_on_gui_thread) == 2
    assert busy_on_gui_thread
    assert all(error_on_gui_thread)
    assert all(busy_on_gui_thread)
