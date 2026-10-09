# Variable Fonts

> Pool start-method side effects, CFF2 delta checks, analyzer blind spots

## Changing the pool start method dropped worker logging

**What happened**: Switching the processor's `ProcessPoolExecutor` to a spawn context removed 437 fork `DeprecationWarning`s, but spawned workers started with no handlers on the `stencilizer` logger, so `--log-level DEBUG --log-file` lost the merge-failure records forked children had inherited.
**Why**: The change was judged by the warning count and the existing tests, none of which read worker-side log output.
**Prevention**: When changing a multiprocessing start method, list what the children inherited under fork (logging handlers, globals, env) and recreate it in a pool initializer; add a test that a worker-side record reaches the log file.
**Fix**: `utils/logging.py` `worker_pool_options()` passes `init_worker_logging` through `core/pool.py`; `tests/unit/test_worker_logging.py` runs a real spawned pool.

## Writer round-trip was the wrong check for CFF2 deltas

**What happened**: CFF2 deltas were first snapped to the 16.16 grid so validation saw the stored geometry. Cantarell `o`, `O`, `zero` and `B` still regained an island after `instantiateVariableFont` (1-6 units of drift between coincident cut edges, even at masters).
**Why**: fontTools' CFF2 instancer rounds every blended relative operand, so fractional deltas break coincidence in its output. The check covered only the engine's own evaluation (`vg.instance`) and a save/reopen.
**Prevention**: Validate a stored format by running it through the consumer that will evaluate it (the fontTools instancer for CFF2, FreeType for glyf), not only through the engine that wrote it; store values the consumer's rounding cannot change.
**Fix**: `round_variable_glyph` stores integer default and integer deltas for CFF2 as for gvar (`rounding._round_deltas`).

## The island count cannot say a counter is open

**What happened**: Stenciled Cantarell `ampersand`, Inter `A` at wght 900 and the static Cantarell `g` and `q` at wght 700 passed every `GlyphAnalyzer` island check while their counters stayed closed by a hairline of ink or a bow-tie; a pathops union found them. Keeping each vertex's default offset from its bridge line per master (the first fix after "Fixed-parameter replay breaks bridge-cut coincidence across masters") left the same hairline when the vertex and the line rounded to different integers.
**Why**: The analyzer decides by contour nesting with exact float comparisons; a wall one unit thick, a one-point pinch or a self-intersection is not a nesting violation.
**Prevention**: Treat a counter as open only when the non-zero union encloses no hole (`variable/holes.py` `enclosed_counters`, both contour directions, pinch points split) and every point of a bridge line sits exactly on it; never accept a replay design on `island_count` alone. A recurrence on the static path is tracked in concerns.md "Static bridges can lean off the bridge line".
**Fix**: `variable/align.py` puts line members on the line, `variable/crossings.py` settles slid crossings, and `validate()` rejects any location with more enclosed counters than allowed.
