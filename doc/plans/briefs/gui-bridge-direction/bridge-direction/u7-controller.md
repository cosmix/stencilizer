# U7: controller keeps per-glyph directions and runs the unbridged survey (gpt-5.6-terra)

Read `_shared.md` in this directory first ("gui/controller.py additions") and
doc/loom/knowledge/architecture/gui.md ("Threading model").

## Files owned

- `src/stencilizer/gui/controller.py`

Read-only: `src/stencilizer/gui/session.py` (U6, in the worktree, read it with `cat`:
`preview(..., directions)`, `unbridged(...)`, `direction_sources`, `save(..., directions=)`),
`src/stencilizer/gui/tasks.py` (`BackgroundTask`: `setAutoDelete(False)`, so the controller
must hold a reference until its handler runs).

## Steps

1. Directions. `__init__` adds `self._directions: dict[str, BridgeDirection] = {}`.
   `_on_font_loaded` resets it to `{}` before emitting `font_loaded`. `direction_for(name)`
   returns `self._directions.get(name, BridgeDirection.AUTO)`. `set_direction(name, direction)`:
   return when no session; when `session.direction_sources(name) != (name,)` emit
   `error` with `f"The bridge direction of '{name}' follows {', '.join(sources)}"` (or
   `f"'{name}' has no islands to bridge"` when sources is empty) and return; otherwise store it
   (AUTO removes the key), emit `direction_changed(name, direction.value)`, refresh the preview
   and schedule a survey. `_refresh_preview` passes `self._directions` to `session.preview`;
   `save` passes `directions=dict(self._directions)` (a copy: the save runs on a pool thread).
2. Survey. `__init__` adds a single-shot `QTimer(self)` with interval `SURVEY_DELAY_MS` whose
   `timeout` connects to `self._run_survey`, plus `self._survey_generation = 0`,
   `self._survey_task: BackgroundTask | None = None`, `self._survey_pending = False`.
   `_schedule_survey()` increments `_survey_generation` and (re)starts the timer; call it at the
   end of `_on_font_loaded`, in `set_parameters`, and in `set_direction`. `_run_survey()`: return
   without a session; if `_survey_task` is not None set `_survey_pending = True` and return;
   otherwise capture `generation`, `session`, `self._bridge` and `dict(self._directions)`, build
   a `BackgroundTask` whose work returns `(generation, session.unbridged(bridge,
   GeometryConfig(), directions))`, connect `finished` to `self._on_survey_finished` and
   `failed` to `self._on_survey_failed` with `Qt.ConnectionType.QueuedConnection`, keep it in
   `_survey_task`, and start it on `self._pool`. The survey never touches `self._task` or
   `busy_changed`: open and save stay available while it runs.
3. `_on_survey_finished(result)`: clear `_survey_task`; emit `unbridged_changed(names)` only when
   the result's generation equals `_survey_generation`; then, if `_survey_pending`, clear it and
   call `_run_survey()`. `_on_survey_failed(message)`: clear `_survey_task` and emit `error`.
   `shutdown()` stops the timer before `self._pool.waitForDone()`.

Constraint: the class must stay under 300 lines and each method under 50 lines.

## Done when

After a font loads, `unbridged_changed` arrives once with the session's unbridged names; a result
whose generation is stale is dropped; `set_direction` on a composite emits `error` and changes
nothing.

## Proof (run once, report the output)

    .venv/bin/ruff check src/stencilizer/gui/controller.py && .venv/bin/mypy src/stencilizer/gui/controller.py
