# W9: app.py (wave 4, codex gpt-6-luna)

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
   `SystemExit("stencilizer-gui needs the 'gui' extra: pip install 'stencilizer[gui]'")` from
   `error`, with `# pragma: no cover` on the except line. `build_parser()`: `argparse` with
   `prog="stencilizer-gui"`, a description, and one optional positional `font` (`type=Path`,
   `nargs="?"`, help "Font file (TTF/OTF) to open on launch"). `default_log_file()` returns
   `Path(tempfile.gettempdir()) / "stencilizer-gui.log"`.
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
- `create_window(roboto_path, tmp_path / "gui.log")` inside
  `qtbot.waitSignal(window.controller.font_loaded, timeout=30000)` (create the window, register it
  with `qtbot.addWidget`, then wait): the grid holds 562 glyphs and `comparison.after_canvas.glyph`
  is not None once the first preview arrives. Call `window.controller.shutdown()` at the end.
- `create_window(None, tmp_path / "gui.log")` returns a window whose `controller.session` is None.

## Proof command

```bash
uv run pytest --no-cov -q tests/gui/test_app.py && uv run mypy src/stencilizer/gui/app.py tests/gui/test_app.py && uv run ruff check src/stencilizer/gui/app.py tests/gui/test_app.py
```
