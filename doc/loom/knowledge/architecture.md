# Architecture

> High-level component relationships, data flow, and module dependencies.
> Keep current: correct or delete entries the code no longer supports.

(Add architecture diagrams and component relationships as you discover them)

## Processing pipeline

`cli/app.py:stencilize()` validates flags (option definitions in `cli/options.py`), builds `StencilizerSettings`, resolves width scaling (`handlers.resolve_width_scaling`), optionally pins a variable font to a static instance (`--instance`, `io/instance.py`; see entry-points.md "CLI"), classifies glyphs once via `FontProcessor.classify_glyphs(reader)` for the island list, and passes that `GlyphClassification` (core/processor.py:28) to `FontProcessor.process(..., classification=...)`. With `--instance` on a glyf variable font in proportional mode it instead stencils the variable font first and runs a static second pass over the pinned instance (`cli/app.py` `_run_stencil_first`, `handlers.stencil_pinned`; [patterns/variable-replay](patterns/variable-replay.md) "Width scaling"). The `--list-islands` and `--dry-run` handlers live in `cli/handlers.py`.

`FontProcessor.process()` classifies (or reuses the classification), dispatches selected glyphs to worker processes with `BridgeConfig` and `GeometryConfig` dictionaries, then writes through `FontWriter` using `FontReader.font`. It does not measure a font-wide stroke width. When `is_variable(reader.font)` (an `fvar` table), `classify_glyphs` and `process` delegate to `variable/processing.py` (`classify_variable_glyphs`, `process_variable_font`), which reuse `_process_glyphs_parallel` with `process_variable_glyph` as the worker and `_save_font(variable=True)`; glyphs the engine cannot use are skipped with reason "unsupported variation data" and their default islands counted as unbridged. The variable path always skips composites and ignores `ProcessingConfig.skip_composite`.

Per glyph: `GlyphAnalyzer` → `GlyphTransformer.transform()` → `ContourMerger` → axis-generic contour builders. The default master's bridge width is `width_percent / 100 * (upm * REFERENCE_STROKE_FRACTION)` (config/settings.py, 0.1 of UPM). `BridgeConfig.width_scaling` (`fixed` default, `proportional`) with `scaling_strength` and `min_width_percent` decides the gap in the other masters of a variable font: `fixed` repeats the default master's gap, `proportional` scales it by the stroke each bridge cuts (`variable/bridge_width.py`); static fonts ignore the three fields. Algorithm detail: [patterns/bridge-algorithm](patterns/bridge-algorithm.md); variable glyphs replay the default-master surgery on every master: [patterns/variable-replay](patterns/variable-replay.md).

## Process-pool IPC

`process_glyph()` is module-level so `ProcessPoolExecutor` can pickle it. Production workers receive `Glyph.to_dict()`, `BridgeConfig.model_dump()`, UPM, and `GeometryConfig.model_dump()` as a keyword argument. The fourth positional argument remains accepted for frozen callers but is ignored. The worker rebuilds the glyph and configs, constructs an analyzer and transformer, and returns `{"glyph", "bridges_added", "duration_ms"}` or `{"error", "traceback"}`; the variable worker `process_variable_glyph` also returns `unbridged_count`. KeyboardInterrupt cancels pending futures. `Glyph`, `Contour`, `Point`, `GlyphMetadata` and `VariableGlyph` serialize for worker IPC.

Every pool starts its workers with a spawn context: `core/pool.py` `pool_options()` returns `mp_context=multiprocessing.get_context("spawn")` plus `utils/logging.py` `worker_pool_options()`, whose initializer `init_worker_logging` rebuilds the `stencilizer` logger's handlers (append-mode file handler, `delay=True`) because a spawned child inherits none of the parent's. Forking would run from a multi-threaded parent in the CLI (the rich progress bar runs a refresh thread) and the GUI. Entry points call `multiprocessing.freeze_support()`. Lesson: [mistakes/variable-fonts.md "Changing the pool start method dropped worker logging"](mistakes/variable-fonts.md).

## Font format I/O

Read: `fonttools_glyph_to_domain()` records any outline with `RecordingPen`; contours are point-reversed to TrueType winding when `"CFF " in font or "CFF2" in font` (src/stencilizer/io/converter.py:33-36). Write: `domain_glyph_to_fonttools()` (converter.py:52) checks `"glyf"` first → `_update_truetype_glyph`, then `"CFF "` → `_update_cff_glyph`, then `"CFF2"` → `_update_cff2_glyph`; anything else raises `GlyphProcessingError`. The CFF writers share `_store_cff_charstring` (draws the contours reversed back to CFF winding) and differ in table key, private dict and width: `_fd_private_dict` resolves the private dict through FDSelect for CID-keyed CFF (no top-level Private; a font without FDSelect uses FDArray[0]), `_update_cff_glyph` subtracts `private.nominalWidthX` from the advance because `T2CharStringPen(width=w)` stores `w` verbatim, and CFF2 charstrings carry no width. `FontReader.format` labels CFF and CFF2 "OpenType".

Fonts with an `fvar` table take a second route: `FontWriter.update_glyph` rejects them and `update_variable_glyph` writes through `variable/write_gvar.py` (glyf plus rebuilt gvar tuples) or `variable/write_cff2.py` (charstring with `blend` operands). Detail: [patterns/variable-replay](patterns/variable-replay.md).

## GUI (summary)

`stencilizer-gui` is a PySide6 window over the unchanged core: `FontSession` (Qt-free) opens, previews and saves; the grid lists island glyphs plus composites that draw one (`gui/composites.py`); `GuiController` runs open and save as `QRunnable`s with queued signals, keeps per-glyph bridge directions, and runs a debounced unbridged-glyph survey on the pool; saves stage in a private temp dir and never write through the output directory. `MainWindow` is a header bar over a splitter of sidebar, glyph grid and preview pane, with save progress in the status bar; `gui/theme.py` styles it in light or dark and follows the system scheme. Layout, theme, threading and save-safety detail: [architecture/gui](architecture/gui.md).

## Font format and error boundary

`FontReader.load` opens TrueType, static CFF, static CFF2 and variable fonts. `FontWriter.update_glyph` raises on an `fvar` font and `update_variable_glyph` writes it; an `fvar` font with `CFF ` outlines loads but fails at save with "unsupported variable outline format" once a glyph needs writing. Glyph conversion errors are raised instead of silently omitting glyphs, and a failed write publishes no output font.

## Processing outcomes

Operational glyph worker, conversion, and write failures abort font publication. Geometrically unbridgeable islands are explicit incomplete outcomes: count only confirmed hole connections as bridges, retain an unbridged count, and warn in the CLI rather than converting an expected geometric limitation into a worker crash.

## Preserve no-op font glyphs

A worker outcome with zero confirmed bridges must not enqueue a glyph rewrite. Keeping the original font glyph avoids dropping TrueType hint instructions or changing encoding merely because analysis found an unbridgeable island. Its unbridged count remains visible in processing statistics.
