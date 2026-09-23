# W9: app.py (wave 4, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for `app.py` is
binding. `main_window.py` and `controller.py` exist: read them with
`cat src/stencilizer/gui/main_window.py src/stencilizer/gui/controller.py`.

## Files owned

- `src/stencilizer/gui/app.py`
- `tests/gui/test_app.py`

Read-only anchors: `MainWindow` (main_window.py), `GuiController` (controller.py), the
`[project.scripts]` table in `pyproject.toml` (`stencilizer-gui = "stencilizer.gui.app:main"`).

## Steps

1. Module top: import PySide6 inside `try:` / `except ImportError as error:` that raises
   `SystemExit(f"stencilizer-gui needs the 'gui' extra (from a source checkout: uv pip install
   -e '.[gui]') ({error})")` from `error`, with `# pragma: no cover` on the except line (a missing system
   library behind PySide6 must stay visible). `build_parser()`: `argparse` with
   `prog="stencilizer-gui"`, a description, and one optional positional `font` (`type=Path`,
   `nargs="?"`, help "Font file (TTF/OTF) to open on launch"). `default_log_file()` returns
   `Path(tempfile.gettempdir()) / f"stencilizer-gui-{getpass.getuser()}.log"` (per user: another
   user's file would raise PermissionError in the appending FileHandler).
2. `create_window(font, log_file)`: `window = MainWindow(GuiController(log_file))`; when `font`
   is not None call `window.load_font(font)`; return the window.
3. `main(argv=None)`: parse `argv` FIRST (so `--help` exits before Qt starts); if
   `multiprocessing.get_start_method(allow_none=True)` is None, call
   `multiprocessing.set_start_method("spawn")` (the save runs a process pool from a worker
   thread; forking a multi-threaded Qt process can deadlock the child); create
   `QApplication(sys.argv[:1])`; `window = create_window(args.font, default_log_file())`;
   `window.show()`; return `app.exec()`.

## Tests (`tests/gui/test_app.py`, `qtbot`)

- The console script is registered: `importlib.metadata.entry_points(group="console_scripts",
  name="stencilizer-gui")` contains an entry whose `value == "stencilizer.gui.app:main"`.
- `build_parser().parse_args([]).font is None`; `parse_args(["x.ttf"]).font == Path("x.ttf")`.
- `main(["--help"])` raises `SystemExit` with code 0 and the captured stdout (`capsys`) contains
  `"stencilizer-gui"` and `"font"`.
- `window = create_window(roboto_path, tmp_path / "gui.log")`; `qtbot.addWidget(window)`; then
  `with qtbot.waitSignal(window.controller.font_loaded, timeout=30000): pass` (safe: the load
  result arrives through a queued connection that only runs inside waitSignal's event loop). The
  grid holds 562 glyphs and `comparison.after_canvas.glyph` is not None once the first preview
  arrives. Call `window.controller.shutdown()` at the end.
- `create_window(None, tmp_path / "gui.log")` returns a window whose `controller.session` is None.
- `test_main_sets_spawn_and_shows_window`: in the `stencilizer.gui.app` namespace monkeypatch
  `multiprocessing.get_start_method` -> `lambda allow_none=False: None` (required: after any
  in-process pool it returns `"fork"`), `multiprocessing.set_start_method` -> a recorder,
  `QApplication` -> a stub class whose `exec()` returns 0 and whose construction is recorded,
  and `create_window` -> a stub recording its `font` argument and returning an object with
  `show()`. Assert `main(["x.ttf"]) == 0`, the start-method recorder holds `["spawn"]` recorded
  before the QApplication construction, `create_window` got `Path("x.ttf")`, and `show()` was
  called.
- `test_main_subprocess_saves_with_spawn`: the production start path in a fresh interpreter
  (the autouse spawn fixture and pytest's own process state do not reach it). Write this driver
  to `tmp_path / "drive_main.py"` (all code in functions; the only top-level code is the
  `__main__` guard, because spawn children re-import `__main__`):

```python
"""Drive stencilizer.gui.app.main: load a font, save it, report, quit."""

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

    main_window.QMessageBox.warning = report_error  # a real warning would block offscreen
    app.create_window = create_window
    return app.main([str(font)])


if __name__ == "__main__":
    sys.exit(main())
```

  `tests/gui/test_app.py` holds the driver as a module-level string constant `DRIVER` and
  writes it with `write_text`; pytest never collects it (the name does not start with `test_`)
  and ruff/mypy see only the string. Run
  `subprocess.run([sys.executable, str(driver), str(roboto_path), str(tmp_path / "out.ttf")],
  env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True,
  timeout=180, check=False)`. Assert `returncode == 0`, stdout contains
  `"start-method=spawn processed=562 errors=0"`, stderr has no `"Traceback"`, and `with
  FontReader(tmp_path / "out.ttf") as reader:` the saved `O` has 4 contours. The lambda
  connections are safe here: `font_loaded` and `save_finished` are emitted by the controller on
  the GUI thread.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/app.py tests/gui/test_app.py && .venv/bin/ruff check src/stencilizer/gui/app.py tests/gui/test_app.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_app.py
```
