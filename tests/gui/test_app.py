"""Tests for the stencilizer GUI application entry point."""

import importlib.metadata
import multiprocessing
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from stencilizer.gui import app
from stencilizer.io import FontReader

DRIVER = '''"""Drive stencilizer.gui.app.main: load a font, save it, report, quit."""

import multiprocessing
import sys
from pathlib import Path


def main() -> int:
    """Run app.main with create_window wrapped to save once the font loads."""
    from PySide6.QtWidgets import QApplication

    from stencilizer.gui import app, main_window
    from stencilizer.utils import ProcessingStats

    font, out = Path(sys.argv[1]), Path(sys.argv[2])
    original = app.create_window

    def report_error(_parent: object, _title: str, message: str) -> None:
        print(f"error={message}", flush=True)
        QApplication.exit(3)

    def create_window(font_arg: Path | None, _log_file: Path) -> main_window.MainWindow:
        window = original(font_arg, out.parent / "gui.log")
        controller = window.controller

        def done(stats: ProcessingStats) -> None:
            method = multiprocessing.get_start_method()
            print(
                f"start-method={method} processed={stats.processed_count} "
                f"errors={stats.error_count}",
                flush=True,
            )
            window.close()
            QApplication.quit()

        controller.font_loaded.connect(lambda _session: window.save_font(out))
        controller.save_finished.connect(done)
        return window

    main_window.QMessageBox.warning = report_error
    app.create_window = create_window
    return app.main([str(font)])


if __name__ == "__main__":
    sys.exit(main())
'''


def test_console_script_is_registered() -> None:
    """The GUI command resolves to the application's main function."""
    entry_points = importlib.metadata.entry_points(group="console_scripts", name="stencilizer-gui")

    assert any(entry.value == "stencilizer.gui.app:main" for entry in entry_points)


def test_build_parser_accepts_an_optional_font() -> None:
    """The optional launch font is parsed as a path."""
    parser = app.build_parser()

    assert parser.parse_args([]).font is None
    assert parser.parse_args(["x.ttf"]).font == Path("x.ttf")


def test_main_help_exits_before_starting_qt(capsys: pytest.CaptureFixture[str]) -> None:
    """Help output is available without starting a QApplication."""
    with pytest.raises(SystemExit) as error:
        app.main(["--help"])

    output = capsys.readouterr().out
    assert error.value.code == 0
    assert "stencilizer-gui" in output
    assert "font" in output


def test_create_window_loads_font(qtbot: QtBot, roboto_path: Path, tmp_path: Path) -> None:
    """A launch font populates the glyph grid and its first preview."""
    window = app.create_window(roboto_path, tmp_path / "gui.log")
    qtbot.addWidget(window)

    with qtbot.waitSignal(window.controller.font_loaded, timeout=30_000):
        pass

    assert window.grid.count() == 1027
    assert window.comparison.after_canvas.glyph is not None
    window.controller.shutdown()


def test_create_window_without_font_has_no_session(qtbot: QtBot, tmp_path: Path) -> None:
    """A window created without a launch font starts without a session."""
    window = app.create_window(None, tmp_path / "gui.log")
    qtbot.addWidget(window)

    assert window.controller.session is None
    window.controller.shutdown()


def _stub_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[str], list[Path | None]]:
    """Replace Qt and window creation with recorders and keep main's log file in tmp_path."""
    events: list[str] = []
    created_fonts: list[Path | None] = []

    class ApplicationStub:
        """Record QApplication construction without starting Qt."""

        def __init__(self, arguments: list[str]) -> None:
            """Record construction arguments."""
            self.arguments = arguments
            events.append("application")

        def exec(self) -> int:
            """Return a successful application exit code."""
            return 0

    class WindowStub:
        """Record that the main window was displayed."""

        def show(self) -> None:
            """Record the show request."""
            events.append("show")

    def create_window(font: Path | None, _log_file: Path) -> WindowStub:
        """Record the optional launch font and return a displayable stub."""
        created_fonts.append(font)
        return WindowStub()

    def skip_theme(_application: object) -> None:
        """Leave the application stub unthemed: theming needs a real QApplication."""

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(app, "QApplication", ApplicationStub)
    monkeypatch.setattr(app, "apply_theme", skip_theme)
    monkeypatch.setattr(app, "create_window", create_window)
    return events, created_fonts


def test_default_log_file_is_private_and_unique(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each run gets its own freshly created log file that only its owner can read."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    first = app.default_log_file()
    second = app.default_log_file()

    assert first != second
    for log_file in (first, second):
        assert log_file.is_file()
        assert log_file.parent == tmp_path
        assert log_file.match("stencilizer-gui-*.log")
        if os.name == "posix":
            assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


def test_main_sets_spawn_and_shows_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The production startup path configures spawn before creating Qt widgets."""
    events, created_fonts = _stub_startup(monkeypatch, tmp_path)
    methods: list[str] = []

    def set_start_method(method: str) -> None:
        events.append("start-method")
        methods.append(method)

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_kwargs: None)
    monkeypatch.setattr(multiprocessing, "set_start_method", set_start_method)

    assert app.main(["x.ttf"]) == 0
    assert methods == ["spawn"]
    assert events.index("start-method") < events.index("application")
    assert created_fonts == [Path("x.ttf")]
    assert events[-1] == "show"


def test_main_keeps_an_existing_start_method(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A start method chosen before main runs is left in place."""
    events, created_fonts = _stub_startup(monkeypatch, tmp_path)
    methods: list[str] = []

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_kwargs: "spawn")
    monkeypatch.setattr(multiprocessing, "set_start_method", methods.append)

    assert app.main(["x.ttf"]) == 0
    assert methods == []
    assert created_fonts == [Path("x.ttf")]
    assert events == ["application", "show"]
    assert len(list(tmp_path.glob("stencilizer-gui-*.log"))) == 1


def test_main_subprocess_saves_with_spawn(roboto_path: Path, tmp_path: Path) -> None:
    """A fresh interpreter loads and saves a font using multiprocessing spawn."""
    driver = tmp_path / "drive_main.py"
    output_path = tmp_path / "out.ttf"
    driver.write_text(DRIVER)

    result = subprocess.run(
        [sys.executable, str(driver), str(roboto_path), str(output_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0
    assert "start-method=spawn processed=562 errors=0" in result.stdout
    assert "Traceback" not in result.stderr
    assert len(list(tmp_path.glob("stencilizer-gui-*.log"))) == 1
    with FontReader(output_path) as reader:
        glyph = reader.get_glyph("O")
    assert glyph is not None
    assert len(glyph.contours) == 4
