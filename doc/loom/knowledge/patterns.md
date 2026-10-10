# Architectural Patterns

> Discovered patterns in the codebase that help agents understand how things work.
> Keep current: correct or delete entries the code no longer supports.

(Add patterns as you discover them)

## Bridge algorithm (summary)

Analyzer finds islands (holes whose on-curve points sit inside an outer contour, src/stencilizer/core/analyzer.py:202); `ContourMerger` cuts notches into the outer contour rather than adding contours. Horizontal and vertical bridges share one implementation parameterized by `Axis` (core/axis.py); horizontal_bridge.py, vertical_bridge.py, multi_island.py and horizontal_multi_island.py are thin wrappers. Font-unit thresholds scale with UPM through `GeometryConfig`. `BridgeConfig.direction` forces an axis per glyph, and `bridges_added` counts islands that left the output (0 = no bridge placed). Full detail, including the direction mapping table and the Θ and ⑧ multi-island cases: [patterns/bridge-algorithm](patterns/bridge-algorithm.md).

## Winding normalization

Internally all contours use TrueType winding: clockwise = outer (negative signed area), counter-clockwise = hole (`GlyphAnalyzer.analyze`, src/stencilizer/core/analyzer.py:94; `signed_area`, src/stencilizer/core/geometry_polygon.py:8). CFF and CFF2 use the opposite, so the converter reverses their point order on read and again on write (src/stencilizer/io/converter.py:33-36, `_store_cff_charstring`). The root CLAUDE.md states the reverse ("TrueType: CCW=outer, CW=inner"); that claim is wrong, the code above is authoritative.

## Exception hierarchy

`StencilizerError` roots `FontError` (`FontLoadError`, `FontSaveError`, `FontFormatError`), `GlyphError` (`GlyphNotFoundError`, `GlyphProcessingError`, `VariationDataError`), `FontProcessingError`, `GeometryError`, and `ProcessingCancelledError` (src/stencilizer/exceptions.py). The CLI catches `FontLoadError`, `FontSaveError`, then `StencilizerError`, then a catch-all. Worker errors return as `{"error", "traceback"}` dictionaries rather than raised exceptions. `VariationDataError` (variation data the variable engine cannot use) never escapes a glyph: the glyph is left unchanged and counted. An invalid `--instance` raises `InstanceSpecError` (a `FontFormatError`, io/instance.py) and exits 1.

## Settings and logging

`StencilizerSettings` composes pydantic `BridgeConfig`, `GeometryConfig`, `ProcessingConfig`, `LoggingConfig` with `Field(ge=, le=)` bounds (src/stencilizer/config/settings.py). `configure_logging()` sets up structlog with an always-on file log, auto-named `stencilizer_<timestamp>.log` when `--log-file` is absent (src/stencilizer/utils/logging.py); `ProcessingStats` tracks counts and per-glyph timings. Spawned pool workers start without handlers, so `worker_pool_options()` passes the `init_worker_logging` initializer that re-attaches the parent's file handler (architecture.md "Process-pool IPC"). Tests that build a `FontProcessor` or run the CLI standard path pass a `tmp_path` log file.

## Variable fonts (summary)

A font with an `fvar` table is stenciled by running the normal default-master pipeline once and replaying it on a master at every `gvar` support peak (or CFF2 region), then re-solving the deltas against the original supports. Overlaps are removed once on the default and the merge is replayed; each bridge line is re-placed per master so coincident cut edges stay coincident; the rounded result is validated at the default, every peak and a grid of axis locations, and a glyph that fails any check is left byte for byte unchanged and counted as unbridged. Modules are under `src/stencilizer/variable/`. Full detail, measured failure rates and checking traps: [patterns/variable-replay](patterns/variable-replay.md).
