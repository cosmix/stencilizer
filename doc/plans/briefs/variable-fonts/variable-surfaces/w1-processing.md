# W1: variable processing and processor dispatch (stage variable-surfaces, wave 1)

Tier: sonnet (`loom-software-engineer`). Never run git.

You own:

- `src/stencilizer/variable/processing.py` (create)
- `src/stencilizer/core/processor.py`
- `src/stencilizer/io/reader.py`
- `tests/unit/test_variable_processing.py` (create)
- `tests/unit/test_review_io.py`
- `tests/unit/test_processor.py`
- `tests/unit/test_processor_more.py`

Read-only:

- `src/stencilizer/variable/transform.py` (`process_variable_glyph`, `transform_variable_glyph`, `VariableOutcome`);
- `variable/reader.py` (`read_variable_glyph`, `is_variable`), `variable/flatten.py` (`flatten_compatible`), `variable/overlaps.py` (`remove_overlaps_compatible`);
- `src/stencilizer/io/writer.py` (`FontWriter.update_variable_glyph`; `update_glyph` raises `FontFormatError` on a font with `fvar`).

Never edit the frozen `tests/unit/test_variable_surface_contracts.py` or `tests/gui/test_variable_gui_contracts.py`, `tests/regression/**` or `tests/unit/test_refactor_contracts.py`.

## Current code (core/processor.py, 372 lines; limit 400)

- `FontProcessor.classify_glyphs(reader) -> GlyphClassification` (line 133). It skips empty, composite and island-free glyphs, recording the reasons in `skipped_reasons`.
- `FontProcessor.process(font_path, output_path=None, max_workers=None, progress_callback=None, classification=None, directions=None) -> ProcessingStats` (line 161; 46 of 50 effective lines, so add nothing there). It creates `stats`, opens a `FontReader` and calls `_process_loaded_font(reader, output_path, max_workers, stats, progress_callback, classification, directions)` (line 209), which fills `stats` in place; `process` stamps `start_time`/`end_time` around it.
- `_process_glyphs_parallel(glyphs, upm, max_workers, stats, progress_callback, directions)` (line 244) submits `process_glyph(glyph_dict, config, upm, geometry_dict=geometry_dict)` to `ProcessPoolExecutor`, reports progress, and cancels pending futures on `KeyboardInterrupt`.
- `_collect_result(future, name, stats, processed)` (line 288) stores a glyph only `if bridges_added` (knowledge `architecture.md` "Preserve no-op font glyphs").
- `_save_font(reader, output_path, processed)` (line 332) writes a temp file beside the output through `FontWriter`, raises `FontSaveError` on any update or save failure, then `Path.replace`s it.
- `tests/unit/test_review_processing.py` calls `_save_font(reader, output, {...})` and `_collect_result(future, name, stats, processed)` positionally; `tests/gui/conftest.py:31-37` and `tests/integration/test_processor_directions.py:22` patch `stencilizer.core.processor.ProcessPoolExecutor` with a spawn context.

## Steps

1. **`io/reader.py`:** delete the remaining `fvar` rejection in `FontReader.load` (lines 54-57 at HEAD 770240f; stage `cff2-static` removed the CFF2 half, so find it by the `"fvar"` check, not the line). In `tests/unit/test_review_io.py`, the reader-rejection test has no case left: rewrite it, keeping its name, into a positive test that `FontReader` loads `tests/fixtures/variable/Ubuntu-VF-subset.ttf` and `is_variable(reader.font)` is True. The stage's test-integrity check fails when test declarations or assertions disappear, so never just delete a test.

2. **Generalize the static loop instead of copying it** (`core/processor.py`). The pool must stay `stencilizer.core.processor.ProcessPoolExecutor`, so the tests' spawn patch also covers variable saves (the GUI saves from a `QThreadPool` thread and must not fork from the multi-threaded Qt process, knowledge `architecture/gui.md` "Threading model").
   - `_process_glyphs_parallel` gains keyword-only `worker: Callable[..., dict[str, Any]] = process_glyph`, and accepts any items with `.name` and `.to_dict()` (`Glyph` or `VariableGlyph`; type it with a small `Protocol`). It submits `worker(item_dict, _config_for_glyph(config_dict, directions, name), upm, geometry_dict=geometry_dict)`; `process_variable_glyph`'s 4th parameter is named `geometry_dict`, so the keyword call fits both.
   - `_collect_result` gains keyword-only `rebuild: Callable[[dict[str, Any]], Any] = Glyph.from_dict`, and `_process_glyphs_parallel` gains the same keyword and passes it through.
   - `_save_font` gains keyword-only `variable: bool = False`; when True it calls `writer.update_variable_glyph(vg)` per entry instead of `writer.update_glyph`. An update failure already raises `FontSaveError` and never publishes, which is the rule for variable fonts too: a failed `update_variable_glyph` can leave glyf replaced and gvar stale.
   - Static calls keep their current positional shapes and defaults, so `tests/regression` and `test_review_processing.py` stay green. Keep every function ≤50 effective lines (`tests/regression/test_code_structure.py`) and the file ≤400 lines.

3. **`variable/processing.py`:**
   - `GlyphClassification` (`core/processor.py`) gains `unsupported_islands: dict[str, int] = field(default_factory=dict)`; static classification leaves it empty.
   - Import `read_variable_glyph` into `variable/processing.py` by name at module level (`from stencilizer.variable.reader import read_variable_glyph`): the frozen contract `test_unsupported_glyph_counted` monkeypatches `stencilizer.variable.processing.read_variable_glyph`.
   - `classify_variable_glyphs(processor: FontProcessor, reader: FontReader) -> tuple[GlyphClassification, dict[str, VariableGlyph]]`. For every glyph in `reader.font.getGlyphOrder()`, per glyph:
     - `read_variable_glyph` returning `None` → skipped `"empty glyph"` or `"composite glyph"` (composite when the glyf entry `isComposite()`);
     - `read_variable_glyph` raising `VariationDataError` → skipped `"unsupported variation data"`; read the static default with `fonttools_glyph_to_domain(name, reader.font.getGlyphSet()[name], reader.font)`, analyze it, and when it has islands record their count in `unsupported_islands[name]`;
     - otherwise `flatten_compatible(vg, upm)`, then `remove_overlaps_compatible`; when that returns `None`, analyze the flattened default instead; when flattening raises `VariationDataError`, analyze `vg.default` (the analyzer accepts curves). Call `processor.analyzer.analyze(default, upm)` (`core/analyzer.py` `GlyphAnalyzer.analyze`), exactly as `transform_variable_glyph` counts, so classification and the worker's no-op outcome agree. No island → `"no islands"`;
     - else append the original `vg.default` (curved, for display) to `glyphs_to_process` and keep `vg` in the dict. A glyph whose flattening raised stays in: the worker returns the no-op outcome and its islands reach `unbridged_count`;
     - one glyph's variation data must never abort the font: `VariationDataError` is a `StencilizerError`, which the CLI maps to exit 1 and the GUI to a failed open. Catch only `VariationDataError`: any other exception is font data fontTools cannot decode and stays a font-level failure, as in the static pipeline.
     - Log skips through `processor.processing_logger.log_glyph_skipped`, as `classify_glyphs` does.
   - `variable_island_counts(reader: FontReader) -> list[tuple[str, int]]`: the same per-glyph path, returning `(name, island count)` for glyphs with islands. Stage W2 calls it from the CLI's `_scan_islands` so `--list-islands` and `--dry-run` show overlap-built counters (Inter `A D P R e 4 &`), which the raw default glyphs do not.
   - `process_variable_font(processor: FontProcessor, reader: FontReader, output_path: Path, max_workers: int | None, stats: ProcessingStats, progress_callback: ProgressCallback | None, classification: GlyphClassification | None, directions: Mapping[str, BridgeDirection] | None) -> None`. It fills the `stats` object `FontProcessor.process` created, in place, like `_process_loaded_font` (which returns `None`; `process` stamps the times and returns that same object). Never rebind `stats` or return a new `ProcessingStats`:
     - with `classification` given (the CLI and the GUI save always pass one), read `read_variable_glyph` only for the names in `classification.glyphs_to_process`, with no second analysis; otherwise call `classify_variable_glyphs`;
     - `stats.skipped_count = selected.skipped_count` and `stats.unbridged_count += sum(selected.unsupported_islands.values())`;
     - `processed = processor._process_glyphs_parallel(list(vgs), upm, max_workers, stats, progress_callback, directions, worker=process_variable_glyph, rebuild=VariableGlyph.from_dict)`; with no glyph to process, skip the pool and save the unchanged font, as `_process_loaded_font` does;
     - `if stats.error_count: raise FontProcessingError(stats.errors)`;
     - `processor._save_font(reader, output_path, processed, variable=True)`.
   - Import from `stencilizer.core.processor` at module level here; `core/processor.py` must import `stencilizer.variable.processing` only inside functions (below).

4. **Dispatch in `core/processor.py`.** `core/__init__.py:34` imports `processor` eagerly, so a module-level import of `variable.processing` is a guaranteed cycle (planning reproduced `ImportError: cannot import name 'FontProcessor' from partially initialized module`). Always import inside the function:
   - `classify_glyphs`: when `is_variable(reader.font)`, return `classify_variable_glyphs(self, reader)[0]`.
   - `_process_loaded_font`: right after logging "Font loaded", when `is_variable(reader.font)`, call `process_variable_font(self, reader, output_path, max_workers, stats, progress_callback, classification, directions)` and return. Write that call literally (the stage wiring check greps `process_variable_font\(` in core/processor.py).
   - Verify all three entry points import: `uv run python -c "import stencilizer.cli.app"`, `"import stencilizer.variable.processing"`, `"import stencilizer.gui.session"`.

5. **Existing mock tests.** `tests/unit/test_processor.py:116-122, 154-160` and `tests/unit/test_processor_more.py:36-42, 85-91, 124-130, 269-275` build `mock_reader = Mock()` with no `.font`; `"fvar" in Mock()` raises `TypeError`, so the dispatch breaks 8 tests (planning simulated it: 8 failed, 30 passed). Add `mock_reader.font = MagicMock()` to each (a `MagicMock` membership test returns False). Change no assertion.

6. **Tests** in `tests/unit/test_variable_processing.py`:
   - on Inter-VF-subset, `classify_glyphs` lists `P` (an overlap-built counter) and skips `l`; `variable_island_counts` lists `P`;
   - a glyph whose `read_variable_glyph` raises `VariationDataError` (monkeypatch it for one name) is skipped with "unsupported variation data" and the rest still classify;
   - `process` on Ubuntu-VF-subset writes a font that keeps `fvar` and `gvar`, and the returned stats have `bridges_added > 0`, `processed_count == len(glyphs_to_process)`, `skipped_count == classification.skipped_count`, `error_count == 0`, `unbridged_count` equal to the sum of the workers' `unbridged_count` values, and `duration_seconds > 0`;
   - `process` on the fvar-only Roboto (Roboto plus fvar, no gvar, built as `tests/gui/conftest.py` fixture `variable_font_path` does) succeeds, bridges `O`, and writes no `gvar`;
   - flattening that raises for one glyph during classification (monkeypatch `flatten_compatible` in `variable.processing` for `o`) keeps `o` in `glyphs_to_process`; the worker side (no-op outcome, islands counted) is the engine's own test, because the worker runs in a spawned process where a monkeypatch does not reach;
   - `progress_callback` fires once per entry of `glyphs_to_process`, with `total == len(glyphs_to_process)` (the CLI progress bar uses that total, cli/app.py:255-266);
   - `process` with a `classification` from `classify_glyphs` analyzes no glyph a second time in the parent (count `GlyphAnalyzer.analyze` calls with a wrapper);
   - a glyph whose outcome has zero bridges is not rewritten (compare glyf bytes and gvar tuples). Find one at test time by running `transform_variable_glyph` over the fixture (Ubuntu `four` had an island but no bridge in the planning spike) and assert at least one exists.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_variable_processing.py tests/unit/test_review_io.py tests/unit/test_processor.py tests/unit/test_processor_more.py tests/unit/test_review_processing.py tests/regression/test_code_structure.py
```

The verifier runs the full gate.
