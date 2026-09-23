# W1: session.py (wave 1, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for `session.py`
is binding.

## Files owned

- `src/stencilizer/gui/__init__.py` (docstring only, see `_shared.md`)
- `src/stencilizer/gui/session.py`
- `tests/gui/test_session.py`

Read-only anchors: `FontProcessor.classify_glyphs`, `FontProcessor.process`, `process_glyph` and
`GlyphClassification` in `src/stencilizer/core/processor.py`; `_classify_font` in
`src/stencilizer/cli/app.py` (the error-wrapping pattern to mirror); `FontReader` in
`src/stencilizer/io/reader.py`; `tests/gui/conftest.py` fixtures.

## Steps

1. `FontSession.open(path, processor)`: inside `with FontReader(path) as reader:` read
   `reader.format`, `reader.units_per_em`, `reader.glyph_count`,
   `int(reader.font["hhea"].ascent)`, `int(reader.font["hhea"].descent)`, and
   `processor.classify_glyphs(reader)`. Wrap ANY exception (including a missing file) as
   `FontLoadError(str(path), str(error))` raised `from error`, exactly as `_classify_font` does.
   `island_glyphs` returns `classification.glyphs_to_process`. `glyph(name)` returns the island
   glyph with that name or None (build a name index once, lazily or in `__post_init__`).
2. `preview(name, bridge, geometry)`: raise `GlyphNotFoundError(name)` if `glyph(name)` is None.
   Call `process_glyph(glyph.to_dict(), bridge.model_dump(), self.units_per_em,
   geometry_dict=geometry.model_dump())` in-process (keyword `geometry_dict`: the fourth
   positional parameter is ignored). An `"error"` key gives `PreviewResult(stenciled=None,
   bridges_added=0, error=result["error"], ...)`; otherwise `stenciled=Glyph.from_dict(result
   ["glyph"])`, `bridges_added=int(result["bridges_added"])`, `error=None`. Always carry
   `duration_ms=float(result["duration_ms"])` and `original=glyph`.
3. `save(output_path, settings, progress)`: if `output_path.resolve() == self.path.resolve()`
   raise `FontSaveError(str(output_path), "output would overwrite the input font")` BEFORE
   anything else (fontTools reads the input lazily while saving; overwriting it corrupts the
   font). Then set `self.processor.config = settings` and return `self.processor.process(
   font_path=self.path, output_path=output_path, max_workers=settings.processing.max_workers,
   progress_callback=progress, classification=self.classification)`. Re-raise
   `StencilizerError` unchanged; wrap any other exception as `FontSaveError(str(output_path),
   str(error))` from error. The classification is reused because it depends only on
   `skip_composite`, which the GUI never changes.

## Tests (`tests/gui/test_session.py`, plain pytest, no Qt)

- Open Roboto: `font_format == "TrueType"`, `units_per_em == 2048`, `ascender == 2146`,
  `descender == -555`, `len(island_glyphs) == 562`, `glyph("O")` is not None, `glyph("space")`
  is None.
- Open CommitMono: `font_format == "OpenType"`, `len(island_glyphs) == 467`, previewing its
  first island glyph gives a non-None `stenciled`.
- Opening a file containing `b"not a font"` and opening a missing path both raise
  `FontLoadError`.
- `preview("O", BridgeConfig(), GeometryConfig())`: `bridges_added == 1`, `error is None`,
  `len(original.contours) == 2`, `len(stenciled.contours) == 4`.
- Width 30.0 vs 110.0 on `O` give different `stenciled.to_dict()`; `use_spanning_bridges`
  True vs False on `B` give different `stenciled.to_dict()`.
- `preview("space", ...)` raises `GlyphNotFoundError`.
- `save(tmp_path / "out.ttf", StencilizerSettings(processing=ProcessingConfig(max_workers=1),
  logging=LoggingConfig(log_file=tmp_path / "save.log")))`: the file exists,
  `stats.processed_count + stats.error_count == 562`, and
  `TTFont(out)["name"].getDebugName(1)` ends with `" Stenciled"`.
- `save` onto the input path (copy Roboto into `tmp_path` first and open the copy) raises
  `FontSaveError` and leaves the copy's bytes unchanged.

## Proof command

```bash
uv run pytest --no-cov -q tests/gui/test_session.py && uv run mypy src/stencilizer/gui/__init__.py src/stencilizer/gui/session.py tests/gui/test_session.py && uv run ruff check src/stencilizer/gui/session.py tests/gui/test_session.py
```
