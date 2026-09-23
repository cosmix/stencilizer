# Gui

> GUI package layout, threading model, save safety

## GUI package layout

`stencilizer-gui = "stencilizer.gui.app:main"` (pyproject.toml:20) launches `src/stencilizer/gui/`. The CLI is untouched: nothing under cli/core/io/config imports `stencilizer.gui`, the package `__init__` imports nothing, and PySide6 lives only in the optional `gui` extra.

| Module | Role |
| --- | --- |
| `app.py` | `build_parser`, `default_log_file`, `create_window`, `main` (sets the `spawn` start method, app.py:57) |
| `session.py` | `FontSession`: Qt-free open, island list, `preview`, `save`; `unsupported_reason` rejects `fvar` and CFF2 with `FontLoadError` |
| `controller.py` | `GuiController`: one `FontProcessor`, one `QThreadPool`, busy guards, signals |
| `tasks.py` | `BackgroundTask` (`QRunnable`, `setAutoDelete(False)`) and `TaskSignals` finished/failed/progress |
| `main_window.py` | `MainWindow`: horizontal `QSplitter` of controls, glyph grid, comparison view |
| `controls.py`, `glyph_grid.py`, `glyph_view.py`, `outline.py` | Control panel; thumbnail `QListWidget`; Original and Stencilized `GlyphCanvas` sharing one union frame; glyph to `QPainterPath` via `QtPen(None, path=path)` |

## Threading model

Open and save run as `BackgroundTask`s on the controller's `QThreadPool`. `TaskSignals` connect to bound controller methods with `Qt.ConnectionType.QueuedConnection` (controller.py:65), so handlers run on the GUI thread. The controller keeps `self._task` until its handler runs, then emits `busy_changed(False)`. `open_font` and `save` refuse a second request while busy (controller.py:102, 135). Previews run synchronously on the GUI thread (one glyph takes at most 8.3 ms). `shutdown()` waits on the pool; `MainWindow.closeEvent` refuses while busy.

The save's `ProcessPoolExecutor` must not fork from the multi-threaded Qt process: `app.main` sets `multiprocessing.set_start_method("spawn")` when none is set, and `tests/gui/conftest.py` patches `stencilizer.core.processor.ProcessPoolExecutor` with a spawn context (tests never call `set_start_method`, it is process-global).

## Save safety

`FontSession.save` (session.py:171-204) refuses the input path (resolve/samefile), a missing output folder, and a source whose digest changed since open (`source_digest`, checked again after the write). `FontWriter` reopens its target by path after the whole glyph run, and `TTFont.save` follows symlinks, so the font is written into a private 0700 `tempfile.TemporaryDirectory` (session.py:189) and published by `_publish`: an `O_CREAT|O_EXCL` sibling of the output written through its descriptor, then rename (session.py:31-45). An output that is a symlink is replaced by a regular file, never written through. `default_log_file` uses `tempfile.mkstemp(prefix="stencilizer-gui-", suffix=".log")` (app.py:40) because `FileHandler` appends through links and a predictable shared-tmp name can be pre-planted.

Accepted residual risks: a non-`OSError` save failure passes `str(error)` to the user; the staging dir follows `TMPDIR`; the digest check cannot stop an input swap-and-revert between its two checks (needs write access to the input).

## Deliberate non-handling

`GuiController._refresh_preview` catches only `StencilizerError`: `process_glyph` already converts transform exceptions into an error dict.
