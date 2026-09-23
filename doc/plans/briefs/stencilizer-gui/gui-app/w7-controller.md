# W7: controller.py (wave 2, codex gpt-6-sol)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for
`controller.py` is binding. `session.py` and `tasks.py` were written in wave 1: read them with
`cat src/stencilizer/gui/session.py src/stencilizer/gui/tasks.py` (the source graph does not
show them).

## Files owned

W7 (wave 2) writes `src/stencilizer/gui/controller.py` from Design decisions and Steps. W7T
(wave 3) writes `tests/gui/test_controller.py` from Tests, reading the finished `controller.py`
with `cat`.

Read-only anchors: `FontSession`, `PreviewResult` in `src/stencilizer/gui/session.py`;
`BackgroundTask`, `ProgressFn` in `src/stencilizer/gui/tasks.py`; `FontProcessor.__init__` in
`src/stencilizer/core/processor.py` (it calls `configure_logging`, which ADDS root-logger
handlers on every construction); `StencilizerSettings`, `LoggingConfig`, `ProcessingConfig`,
`GeometryConfig` in `src/stencilizer/config/settings.py`.

## Design decisions (settled; do not redesign)

- ONE `FontProcessor` per controller, built in `__init__` from
  `StencilizerSettings(logging=LoggingConfig(log_file=log_file, log_level="WARNING",
  file_log_level="INFO"))`. Never build another: each construction stacks more logging
  handlers.
- An owned `QThreadPool(self)` (never `globalInstance()`), so `shutdown()` waits only for this
  controller's work.
- Busy means `self._task is not None`. The guard lives in `open_font` and `save` themselves (the
  mutating entry points), not only in disabled buttons: a busy call emits
  `error("Busy: wait for the current operation to finish")` and returns without side effects.
- Task signals connect to bound methods of the controller with
  `Qt.ConnectionType.QueuedConnection`. The controller keeps `self._task` referenced until the
  finished/failed handler runs, then sets it to None, then emits `busy_changed(False)`, then the
  outward signal.
- Previews run synchronously on the calling (GUI) thread: a glyph transforms in at most 8.3 ms.
  No debounce timer, no preview thread.
- Parameters are stored as `self._bridge: BridgeConfig` (default `BridgeConfig()`) and
  `self._max_workers: int | None`; the geometry is always `GeometryConfig()` (the GUI does not
  expose it).

## Steps

1. `__init__`, `session`, `is_busy`, a private `_start(task, on_finished)`: connect
   `task.signals.finished` -> `on_finished`, `task.signals.failed` -> `self._on_failed` and
   `task.signals.progress` -> `self._on_progress` (loads never emit progress), all with
   `Qt.ConnectionType.QueuedConnection`, BEFORE anything starts; then store the task, emit
   `busy_changed(True)`, and `self._pool.start(task)`. A signal emitted before its connection
   exists is dropped: a missing font fails at once, and a lost `failed` would leave the
   controller busy forever. `open_font(path)`: busy guard, then `_start` a `BackgroundTask`
   whose work calls `FontSession.open(path, self._processor)`. On finished: store the session,
   reset the selected glyph to None, then emit `font_loaded(session)`. On failed: emit
   `error(message)`.
2. `set_parameters(bridge, max_workers)` stores both and refreshes the preview; `select_glyph(
   name)` stores the name and refreshes. The refresh does nothing without a session or a
   selection; otherwise it calls `session.preview(name, self._bridge, GeometryConfig())` and emits
   `preview_ready(result)`, or `error(str(error))` on `StencilizerError`.
3. `save(output_path)`: busy guard; without a session emit `error("No font loaded")` and return.
   Build `StencilizerSettings(bridge=self._bridge, processing=ProcessingConfig(max_workers=
   self._max_workers), logging=self._processor.config.logging)` and `_start` a task whose work is
   `session.save(output_path, settings, <adapter>)`, the adapter mapping the processor's
   `(completed, total, name, success)` callback onto the task's `progress(completed, total)`.
   `_on_progress` relays task progress to `save_progress`. On finished emit
   `save_finished(stats)`; on failed emit `error(message)`. `shutdown()` calls
   `self._pool.waitForDone()`.

## Tests (`tests/gui/test_controller.py`, `qtbot`)

Fixture: `controller = GuiController(tmp_path / "gui.log")`, yielded, then
`controller.shutdown()` in teardown. Use `timeout=30000` for loads and `timeout=120000` for saves.

- `open_font(roboto_path)`: `font_loaded` arrives with a `FontSession` of 562 island glyphs;
  `busy_changed` emitted True then False; `is_busy` is False afterwards.
- `open_font` on a file containing `b"not a font"`: `error` arrives with a message starting
  `"Failed to load font"`; `session` stays None.
- `open_font(cff2_font_path)`: `error` arrives with a message starting `"Failed to load font"`
  and containing `"CFF2"`; no `font_loaded`; `session` stays None; `is_busy` is False.
- `test_open_font_busy_guard`: call `open_font(roboto_path)` twice in a row without waiting. The
  second call emits `error` synchronously with the busy message; exactly one `font_loaded`
  follows; also `open_font(roboto_path)` then immediately `save(tmp_path / "x.ttf")` emits
  `error` with the busy message and `x.ttf` does not exist.
- After loading, `select_glyph("O")` emits `preview_ready` synchronously with `bridges_added == 1`.
- `set_parameters(BridgeConfig(width_percent=30.0), None)` then `width_percent=110.0` with `O`
  selected: two `preview_ready` emissions with different `stenciled.to_dict()`.
- With `B` selected, `use_spanning_bridges` True vs False gives different `stenciled.to_dict()`.
- `set_parameters` before any selection emits no `preview_ready` (`qtbot.assertNotEmitted`).
- `save(tmp_path / "out.ttf")` after `set_parameters(BridgeConfig(), 1)`: `save_finished` arrives
  with `processed_count == 562` and `error_count == 0`; at least one `save_progress` emission
  has total 562; `with FontReader(out) as reader:` the saved `O` matches
  (`outlines_match`) `controller.session.preview("O", BridgeConfig(), GeometryConfig())
  .stenciled`.
- `test_shutdown_waits_for_active_save`: after a load, `set_parameters(BridgeConfig(), 1)`,
  `save(tmp_path / "s.ttf")`, then immediately `controller.shutdown()`. When `shutdown()`
  returns, `s.ttf` exists and reads back through `FontReader` with a 4-contour `O` (the pool
  finished the task: no half-written output, no hang); then `qtbot.waitSignal(
  controller.save_finished, timeout=30000)` delivers stats with `error_count == 0` and
  `is_busy` is False afterwards. `shutdown()` has no deadline of its own: the save's bounded
  duration (about 1 s) bounds it, and pytest's failure would be a hang the orchestrator sees.
- `save` before any load emits `error("No font loaded")`.
- `save` onto the loaded font's own path (load a copy in `tmp_path`): `error` message starts with
  `"Failed to save font"` and the copy's bytes are unchanged.
- Handler order: a receiver connected to `font_loaded` records `controller.is_busy`; after the
  load the recorded list is `[False]`.
- `test_signals_delivered_on_gui_thread`: connect `font_loaded`, `save_progress` and
  `save_finished` with `Qt.ConnectionType.DirectConnection` to a recorder storing
  `QThread.currentThread()`; after a load and a save (`set_parameters(BridgeConfig(), 1)`),
  every recorded thread `== controller.thread()` (a handler wired through a lambda would run in
  the pool thread and fail this).
- Parameters reach the save: `monkeypatch.setattr(FontSession, "save", recorder)` with a
  recorder `(self, output_path, settings, progress=None)` that stores `settings` and returns
  `ProcessingStats()`; after `set_parameters(BridgeConfig(width_percent=30.0), 1)` and
  `save(tmp_path / "p.ttf")`, once `save_finished` arrives, `settings.bridge.width_percent ==
  30.0` and `settings.processing.max_workers == 1`.
- `test_single_processor_per_controller`: record `len(logging.getLogger().handlers)` right
  after building the controller; after open, save, and a second open it is unchanged.

## Proof command

W7 (module unit):

```bash
.venv/bin/mypy src/stencilizer/gui/controller.py && .venv/bin/ruff check src/stencilizer/gui/controller.py
```

W7T (test unit):

```bash
.venv/bin/mypy src/stencilizer/gui/controller.py tests/gui/test_controller.py && .venv/bin/ruff check tests/gui/test_controller.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_controller.py
```
