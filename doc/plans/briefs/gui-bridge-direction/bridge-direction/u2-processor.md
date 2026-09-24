# U2: per-glyph directions and a truthful bridge count in the processor, with tests (gpt-5.6-terra)

Read `_shared.md` in this directory first ("core/processor.py" contract, "Measured facts").

## Files owned

- `src/stencilizer/core/processor.py`
- `tests/integration/test_processor_directions.py` (new; under 400 lines)

Read-only: `src/stencilizer/core/analyzer.py` (`ContourHierarchy.get_islands` returns contour
indices), `src/stencilizer/domain/contour.py` (`Contour.to_dict`),
`src/stencilizer/config/settings.py` (`BridgeDirection`, `BridgeConfig.direction`),
`tests/integration/conftest.py` (`FIXTURES_DIR`), `tests/gui/conftest.py`
(`spawn_process_pool`: the pattern to copy).

## Steps

1. `_transform_glyph`: count islands actually bridged. Before calling `transformer.transform`,
   snapshot `before = [glyph.contours[idx].to_dict() for idx in
   analyzer.analyze(glyph).get_islands()]`; after it, `after = [c.to_dict() for c in
   transformed.contours]`; return `sum(1 for island in before if island not in after)` as the
   count. Take the snapshot BEFORE the transform. An island left unbridged is appended verbatim by
   `GlyphTransformer.transform`; a bridged one is cut into new contours. `process_glyph` keeps its
   signature and its "bridges_added" key.
2. `FontProcessor.process` gains the trailing keyword parameter
   `directions: Mapping[str, BridgeDirection] | None = None` and passes it through
   `_process_loaded_font` (new trailing parameter) to `_process_glyphs_parallel` (new trailing
   keyword parameter `directions: Mapping[str, BridgeDirection] | None = None`). Existing callers
   pass positionals up to `classification`; keep every existing parameter's position. In
   `_process_glyphs_parallel` submit each glyph with the shared `config_dict` when it is absent
   from `directions`, else `{**config_dict, "direction": directions[name]}`. Keep the function
   at or under 50 lines (docstring excluded); a small private helper for the per-glyph dict is
   fine. Import `Mapping` from `collections.abc`, `BridgeDirection` from
   `stencilizer.config.settings`.
3. Write `tests/integration/test_processor_directions.py`, with a module-level autouse fixture
   copying `spawn_process_pool` (monkeypatch `stencilizer.core.processor.ProcessPoolExecutor`
   with the spawn context) and every `FontProcessor` built with
   `LoggingConfig(log_file=tmp_path / "p.log")`:
   - `test_unbridgeable_glyph_reports_zero_bridges`: through `process_glyph` with default
     settings, Roboto `four` and `AE` report `bridges_added == 0` and output contours equal to the
     input; `O` reports 1, `B` and `eight` report 2.
   - `test_process_applies_per_glyph_directions`: process Roboto into `tmp_path` twice with
     `max_workers=1`, once with `directions={"O": BridgeDirection.HORIZONTAL}` and once without;
     `error_count == 0` both times; read `O` back with `FontReader(...).get_glyph("O")`: the
     directed save has no contour bbox spanning the input O's centre y (728), the plain one none
     spanning its centre x (703.5); `D` reads back identical in both saves.
   No `skip`/`xfail`/`importorskip` in this file.

The CLI never passes `directions`; its only change is that `stats.bridges_added` no longer counts
islands that got no bridge.

## Done when

A glyph whose output keeps all its islands verbatim reports `bridges_added == 0`, `process(...,
directions=...)` gives only the named glyph its direction, and the proof command is clean.

## Proof (run once, report the output; never run the tests)

    .venv/bin/ruff check src/stencilizer/core/processor.py tests/integration/test_processor_directions.py && .venv/bin/mypy src/stencilizer/core/processor.py tests/integration/test_processor_directions.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/integration/test_processor_directions.py
