# W3: tasks.py (wave 1, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for `tasks.py`
is binding.

## Files owned

- `src/stencilizer/gui/tasks.py`
- `tests/gui/test_tasks.py`

Read-only anchors: `StencilizerError` in `src/stencilizer/exceptions.py`.

## Steps

1. `TaskSignals(QObject)` with the three class-level signals exactly as in `_shared.md`.
2. `BackgroundTask.__init__(work)`: `super().__init__()`, `self.signals = TaskSignals()`,
   store `work`, and call `self.setAutoDelete(False)`. The owner (the controller) keeps a Python
   reference until it has handled `finished` or `failed`; auto-delete would free the C++
   runnable while Python still holds it.
3. `run()`: call `self._work(self.signals.progress.emit)`. On `StencilizerError` emit
   `failed(str(error))`; on any other `Exception` emit `failed(f"Unexpected error: {error}")`;
   otherwise emit `finished(result)`. Exactly one of `finished`/`failed` fires per run. Never let
   an exception escape `run()` (it would be lost on the pool thread).

## Tests (`tests/gui/test_tasks.py`, use `qtbot` and a local `QThreadPool()`)

- Success: a work function returning 42 -> `qtbot.waitSignal(task.signals.finished)` delivers
  `[42]`; `failed` is not emitted (`qtbot.assertNotEmitted`).
- Progress: a work function calling `progress(1, 3)`, `progress(3, 3)` -> both pairs arrive in
  order (collect them through a receiver QObject's bound method, connected with
  `Qt.ConnectionType.QueuedConnection`, as production code does).
- `StencilizerError("bad input")` -> `failed` delivers `["bad input"]`; `finished` is not
  emitted (`qtbot.assertNotEmitted(task.signals.finished)`).
- `ValueError("boom")` -> `failed` delivers `["Unexpected error: boom"]`; `finished` is not
  emitted (`qtbot.assertNotEmitted(task.signals.finished)`).
- `BackgroundTask(...).autoDelete()` is False.
- Call `pool.waitForDone()` at the end of each test.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/tasks.py tests/gui/test_tasks.py && .venv/bin/ruff check src/stencilizer/gui/tasks.py tests/gui/test_tasks.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_tasks.py
```
