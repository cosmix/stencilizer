# Gui

> GUI package layout, threading model, save safety

## GUI package layout

`stencilizer-gui = "stencilizer.gui.app:main"` (pyproject.toml:20) launches `src/stencilizer/gui/`. The CLI is untouched: nothing under cli/core/io/config imports `stencilizer.gui`, the package `__init__` imports nothing, and PySide6 lives only in the optional `gui` extra.

| Module | Role |
| --- | --- |
| `app.py` | `build_parser`, `default_log_file`, `create_window`, `main` (sets the `spawn` start method, app.py:57) |
| `session.py` | `FontSession`: Qt-free open, `display_glyphs` (island glyphs plus bridged composites), `direction_sources`, `preview`, `unbridged`, `save`; `unsupported_reason` rejects `fvar` and CFF2 with `FontLoadError` |
| `composites.py` | Qt-free composite discovery and composition over the fontTools glyph set: `find_bridged_composites`, `load_component_outlines`, `compose` |
| `controller.py` | `GuiController`: one `FontProcessor`, one `QThreadPool`, busy guards, per-glyph directions, debounced survey, signals |
| `tasks.py` | `BackgroundTask` (`QRunnable`, `setAutoDelete(False)`) and `TaskSignals` finished/failed/progress |
| `main_window.py` | `MainWindow`: horizontal `QSplitter` of controls, glyph grid, comparison view |
| `direction_picker.py` | `DirectionPicker`: Auto / Vertical / Horizontal combo; for a composite it is disabled and reads "Follows <base>" (direction_picker.py:41) |
| `controls.py`, `glyph_grid.py`, `glyph_view.py`, `outline.py` | Control panel; thumbnail `QListWidget` (red mark via `UNBRIDGED_ROLE`, direction marker); Original and Stencilized `GlyphCanvas` sharing one union frame; glyph to `QPainterPath` via `QtPen(None, path=path)` |

## Composites in the grid

The domain reader drops components (`fonttools_glyph_to_domain` records outline segments only), so `classify_glyphs` files a composite as an empty glyph and `session.island_glyphs` (= `classification.glyphs_to_process`) never lists one, yet the saved font bridges it through the glyphs it references. `FontSession.display_glyphs` therefore adds every composite that draws an island glyph, found by `find_bridged_composites` over the fontTools glyph set (Roboto: 562 island glyphs + 465 composites = 1027; Lato 447 + 370; CommitMono 467 + 0). A composite is composed from leaf outlines with fontTools `Transform` (child first, then parent), matching `DecomposingRecordingPen` output for all Roboto composites. `direction_sources(name)` returns the island glyphs a name follows: `(name,)` for an island glyph, the referenced islands for a composite, `()` otherwise. A composite has no direction of its own: `preview` runs each source under that source's direction and composes the result. `_component_parts` bounds depth and part count and raises `ValueError` on a component cycle, which `FontSession.open` wraps as `FontLoadError`; without the bound a doubling component DAG hung open.

## Threading model

Open and save run as `BackgroundTask`s on the controller's `QThreadPool`. `_start` connects `TaskSignals` to bound controller methods with `Qt.ConnectionType.QueuedConnection` (controller.py:79-86), so handlers run on the GUI thread. The controller keeps `self._task` until its handler runs, then emits `busy_changed(False)`. `open_font` and `save` refuse a second request while busy (controller.py:120, 183). Previews run synchronously on the GUI thread (one glyph takes at most 8.3 ms). `shutdown()` stops the survey timer and waits on the pool; `MainWindow.closeEvent` refuses while busy.

**Directions.** The controller holds per-glyph choices in `_directions` (`direction_for`, `set_direction`, controller.py:136-158); `AUTO` removes the entry. `set_direction` refuses a composite or non-island name with an `error` signal. Every change refreshes the preview and schedules a survey; `save` passes a copy of the dict through `functools.partial(session.save, ..., directions=dict(self._directions))`.

**Unbridged survey.** `_schedule_survey` bumps `_survey_generation` and restarts a `SURVEY_DELAY_MS = 250` timer (controller.py:27); `_run_survey` runs `session.unbridged(bridge, geometry, directions)` for every displayed glyph as its own `BackgroundTask` kept in `_survey_task`, never touching `_task` or `busy_changed`, so a save is not blocked by it. Only one survey runs at a time; a request during a run sets `_survey_pending`. `_on_survey_finished` emits `unbridged_changed` only when the result's generation equals the current one, then starts the pending survey. `_on_survey_failed` mirrors that: it clears the task, reports only a failure of the current generation, and drains `_survey_pending`. A failure handler must repeat the success path's pending and generation handling.

The save's `ProcessPoolExecutor` must not fork from the multi-threaded Qt process: `app.main` sets `multiprocessing.set_start_method("spawn")` when none is set, and `tests/gui/conftest.py` patches `stencilizer.core.processor.ProcessPoolExecutor` with a spawn context (tests never call `set_start_method`, it is process-global).

## Save safety

`FontSession.save` (session.py:280-315) refuses the input path (resolve/samefile), a missing output folder, and a source whose digest changed since open (`source_digest`, checked again after the write; `_assert_source_unchanged`). `FontWriter` reopens its target by path after the whole glyph run, and `TTFont.save` follows symlinks, so the font is written into a private 0700 `tempfile.TemporaryDirectory` (session.py:299) and published by `_publish`: an `O_CREAT|O_EXCL` sibling of the output written through its descriptor, then rename (session.py:38-51). An output that is a symlink is replaced by a regular file, never written through. `default_log_file` uses `tempfile.mkstemp(prefix="stencilizer-gui-", suffix=".log")` (app.py:40) because `FileHandler` appends through links and a predictable shared-tmp name can be pre-planted. `save(..., directions=None)` forwards the per-glyph directions to `FontProcessor.process`; nothing else in it depends on them.

Accepted residual risks: a non-`OSError` save failure passes `str(error)` to the user; the staging dir follows `TMPDIR`; the digest check cannot stop an input swap-and-revert between its two checks (needs write access to the input).

## Deliberate non-handling

`GuiController._refresh_preview` catches only `StencilizerError`: `process_glyph` already converts transform exceptions into an error dict.
