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

1. `FontSession.open(path, processor)`, all inside one `try`: first `source_sha256 =
   source_digest(path)` (the pinned revision, see `_shared.md`); then inside
   `with FontReader(path) as reader:` call
   `unsupported_reason(reader.font)` and, when it returns a reason, raise
   `FontLoadError(str(path), reason)` BEFORE classifying; otherwise read `reader.format`,
   `reader.units_per_em`, `reader.glyph_count`, `int(reader.font["hhea"].ascent)`,
   `int(reader.font["hhea"].descent)`, and `processor.classify_glyphs(reader)`. Re-raise
   `FontLoadError` unchanged; wrap ANY other exception (including a missing file) as
   `FontLoadError(str(path), str(error))` raised `from error`, exactly as `_classify_font` does.
   `unsupported_reason(font)` checks, in this order: `"fvar" in font` -> `"variable fonts (fvar
   table) are not supported"`; `"CFF2" in font` -> `"CFF2 outlines are not supported"`; neither
   `"glyf"` nor `"CFF "` in font -> `"no supported outline table (glyf or CFF)"`; else None.
   The core writer handles only `glyf` and `CFF ` (`domain_glyph_to_fonttools` in
   `io/converter.py`) and `FontProcessor._save_font` logs a failed glyph write without counting
   it, then saves and renames the font anyway, so an unsupported font must never become a
   session.
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
   or (`output_path.exists()` and `output_path.samefile(self.path)`), raise
   `FontSaveError(str(output_path), "output would overwrite the input font")` BEFORE anything
   else. This is a product rule (never replace the user's original font; symlinks and hard
   links to it count), not a corruption guard: `TTFont.save` serializes to memory before
   opening the destination. Next the source-revision check: when `source_digest(self.path)`
   raises `OSError` or differs from `self.source_sha256`, raise `FontSaveError(str(output_path),
   f"input font '{self.path}' changed on disk since it was opened; reopen it")`. Then set
   `self.processor.config = settings` and `stats = self.processor.process(font_path=self.path,
   output_path=output_path, max_workers=settings.processing.max_workers,
   progress_callback=progress, classification=self.classification)`. After it returns, run the
   source-revision check again; on a mismatch `output_path.unlink(missing_ok=True)` and raise
   the same `FontSaveError` (the output could mix the old outlines with the new file's tables).
   Return `stats`. Re-raise `StencilizerError` unchanged; wrap any other exception as
   `FontSaveError(str(output_path), str(error))` from error. Put the check in one private
   helper so `save` stays under 50 lines. The classification is reused because it holds the
   outlines of the pinned revision (the preview shows the same ones) and depends otherwise only
   on `skip_composite`, which the GUI never changes; the two checks guarantee the reopened file
   is still that revision.

## Tests (`tests/gui/test_session.py`, plain pytest, no Qt)

- Open Roboto: `font_format == "TrueType"`, `units_per_em == 2048`, `ascender == 2146`,
  `descender == -555`, `len(island_glyphs) == 562`, `glyph("O")` is not None, `glyph("space")`
  is None.
- Open CommitMono: `font_format == "OpenType"`, `len(island_glyphs) == 467`, previewing its
  first island glyph gives a non-None `stenciled`.
- Opening a file containing `b"not a font"` and opening a missing path both raise
  `FontLoadError`.
- `test_open_rejects_unsupported_fonts`: `FontSession.open(cff2_font_path, processor)` raises
  `FontLoadError` whose message contains `"CFF2"`; `variable_font_path` raises one containing
  `"variable"`. `unsupported_reason(TTFont())` contains `"glyf or CFF"`;
  `unsupported_reason(TTFont(roboto_path))` and `unsupported_reason(TTFont(commit_mono_path))`
  are None. A mocked `processor.classify_glyphs` (`monkeypatch.setattr`, recording calls) is
  never called for the two rejected files.
- `FontSession.open(roboto_path, processor).source_sha256 ==
  hashlib.sha256(roboto_path.read_bytes()).hexdigest()`.
- `preview("O", BridgeConfig(), GeometryConfig())`: `bridges_added == 1`, `error is None`,
  `len(original.contours) == 2`, `len(stenciled.contours) == 4`.
- Width 30.0 vs 110.0 on `O` give different `stenciled.to_dict()`; `use_spanning_bridges`
  True vs False on `B` give different `stenciled.to_dict()`.
- `preview("space", ...)` raises `GlyphNotFoundError`.
- `test_save_writes_stenciled_outlines`, parametrized over Roboto (562 island glyphs, output
  `out.ttf`) and CommitMono (467, `out.otf`): `save(tmp_path / <out>,
  StencilizerSettings(processing=ProcessingConfig(max_workers=1),
  logging=processor.config.logging))` returns `stats.processed_count == <island count>` and
  `stats.error_count == 0`; `TTFont(out)["name"].getDebugName(1)` ends with `" Stenciled"`;
  `with FontReader(out) as reader:` the saved `O` has 4 contours and
  `outlines_match(saved_O, session.preview("O", BridgeConfig(), GeometryConfig()).stenciled)`
  is True. A renamed copy of the input, or a save whose glyph writes all failed, keeps the
  2-contour `O` and fails this.
- `test_save_uses_given_settings`: save to `tmp_path / "a.ttf"` with
  `BridgeConfig(width_percent=30.0)` and to `tmp_path / "b.ttf"` with `110.0`, then to
  `c.ttf` with `BridgeConfig(use_spanning_bridges=True)` and to `d.ttf` with `False` (all
  `ProcessingConfig(max_workers=1)`, each `stats.error_count == 0`). Read back with
  `FontReader`: saved `O` of a and b differ, and each matches (`outlines_match`) the preview of
  `O` at its own width; saved `B` of c and d differ, and each matches the preview of `B` with
  its own toggle (a `save` that never sets `processor.config` fails this).
- `test_save_refuses_changed_source`: copy Roboto into `tmp_path`, open the copy, then overwrite
  the copy with the bytes of `tests/fixtures/Lato-Black.ttf`. `save(tmp_path / "out.ttf", ...)`
  raises `FontSaveError` whose message contains `"changed on disk"`, and `out.ttf` does not
  exist. Deleting the copy instead gives the same error. Second case (change during the save):
  open a fresh copy, `monkeypatch.setattr(processor, "process", fake)` where `fake(**kwargs)`
  writes `b"partial"` to `kwargs["output_path"]`, overwrites the source copy with Lato's bytes
  and returns `ProcessingStats()`; `save` raises `FontSaveError` containing `"changed on
  disk"` and the output file no longer exists.
- `test_save_refuses_input_path`: copy Roboto into `tmp_path` and open the copy. Saving to the
  copy's path, to a symlink to it (`Path.symlink_to`), to a hard link to it (`os.link`), and to
  `tmp_path / "sub" / ".." / copy.name` each raise `FontSaveError`, and the copy's bytes are
  unchanged after all four.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/__init__.py src/stencilizer/gui/session.py tests/gui/test_session.py && .venv/bin/ruff check src/stencilizer/gui/session.py tests/gui/test_session.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_session.py
```
