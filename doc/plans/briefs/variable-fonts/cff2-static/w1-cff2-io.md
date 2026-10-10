# W1: CFF2 read/write and rejection tests (stage cff2-static)

Tier: sonnet (`loom-software-engineer`). You own exactly these files:

- `src/stencilizer/io/converter.py`
- `src/stencilizer/io/reader.py`
- `src/stencilizer/io/writer.py`
- `src/stencilizer/gui/session.py`
- `pyproject.toml`
- `uv.lock`
- `tests/unit/test_io.py`
- `tests/unit/test_review_io.py`
- `tests/gui/conftest.py`
- `tests/gui/test_session.py`
- `tests/gui/test_main_window.py`
- `tests/gui/test_controller_errors.py`

Never touch the frozen contract files `tests/unit/test_cff2_contracts.py` and `tests/gui/test_cff2_session_contracts.py`; they must pass when you are done. Never run git.

## Goal

Static CFF2 fonts load with TrueType winding, are stenciled, and save as CFF2. Variable fonts (`fvar`) remain rejected everywhere in this stage.

## Steps

1. **Read path.** `fonttools_glyph_to_domain` (`io/converter.py`) reverses contour points only when `"CFF " in font` (the `is_cff` check near line 32). Make the condition true for `"CFF2"` as well.

2. **Write path.** `domain_glyph_to_fonttools` (converter.py:51-72) dispatches `"glyf"` → `_update_truetype_glyph`, `"CFF "` → `_update_cff_glyph`, and everything else to `NotImplementedError`. Add an `elif "CFF2" in font:` branch whose body is exactly `_update_cff2_glyph(glyph, original_glyph, font)` (the stage's wiring check greps that literal call), calling a new `_update_cff2_glyph(glyph: Glyph, _: Any, font: TTFont) -> None` modelled on `_update_cff_glyph` (converter.py:232) with these differences:
   - The table is `font["CFF2"]` and the top dict is `cff_table.cff.topDictIndex[0]`. CFF2 has no `top_dict.Private` (a converted CommitMono has one FDArray entry and no FDSelect). Find the private dict with `_, fd_index = top_dict.CharStrings.getItemAndSelector(glyph.name)` and `private = top_dict.FDArray[fd_index or 0].Private`; this handles FDSelect fonts without an O(n) glyph-order lookup. Keep `charstrings = top_dict.CharStrings` and `global_subrs = cff_table.cff.GlobalSubrs`.
   - Use `T2CharStringPen(width=None, glyphSet=font.getGlyphSet(), CFF2=True)`. CFF2 charstrings carry no width (the pen asserts if one is passed with `CFF2=True`); advance widths live in hmtx.
   - Reverse the points back to CFF winding exactly as `_update_cff_glyph` does.
   - Call `getCharString(private=private, globalSubrs=global_subrs, optimize=False)`. This deliberately mirrors the CFF path (knowledge `patterns/bridge-algorithm.md` "Preserve CFF preview vertices").
   - Before you rely on the `T2CharStringPen` parameter names, confirm them with `uv run python -c "import inspect; from fontTools.pens.t2CharStringPen import T2CharStringPen as P; print(inspect.signature(P.__init__))"`.

3. **Rejections.** Keep the `fvar` rejection and remove the CFF2-only one:
   - `FontReader.load` (`io/reader.py` lines 54-57) must still raise `FontFormatError` for `"fvar"`.
   - `_check_supported_format` (`io/writer.py` lines 25-29) loses its CFF2 branch.
   - `gui/session.py` `unsupported_reason` (lines 61-69) loses its CFF2 branch. Its last check must accept `glyf`, `CFF ` or `CFF2`; the message becomes "no supported outline table (glyf, CFF or CFF2)".
   - `_check_supported_format` keeps only its `fvar` branch here; stage `variable-writers` moves that guard into `FontWriter.update_glyph`.

4. **Repair the stale test** that is red at 4254909: `tests/unit/test_io.py::TestCffGlyphUpdate::test_update_cff_glyph_passes_private_and_global_subrs`. It must expect `optimize=False` in the `getCharString` call. If your base already has that, there is nothing to do.

   **Add pytest-xdist:** run `uv add --dev pytest-xdist`, never a hand edit. Integration-verify runs the whole suite with `--numprocesses=16`: the serial run takes about 600 s, past the 300 s acceptance cap, and planning measured 69 s with 16 workers and 406 passing. Add a `TestCff2GlyphUpdate` class with these unit tests (mock style, as in `TestCffGlyphUpdate`):
   - the FDArray private is used;
   - `CFF2=True` is passed to the pen;
   - the points are reversed.

5. **Update the tests that assert the old CFF2 rejection.** The stage's test-integrity check fails on fewer test declarations or assertions in a file, so every rejection assertion you drop is replaced by a positive assertion in the same file, never just deleted:
   - `tests/unit/test_review_io.py`: the two `@pytest.mark.parametrize("table", ["fvar", "CFF2"])` tests (lines 26-50) become fvar-only. Add one positive test to that file: a converted CFF2 CommitMono (built as `tests/gui/conftest.py:69-76` does) loads in `FontReader`, and `FontWriter(font, tmp_path / "out.otf").save()` writes a file that reopens with a `CFF2` table.
   - `tests/gui/test_session.py`: lines 77-78 (CFF2 rejected) become "opening `cff2_font_path` succeeds and lists `O`"; keep the fvar assertion at :79. Line 83 asserts `"glyf or CFF" in ...`; update it to the new message text below. The same test monkeypatches `classify_glyphs` to record calls (:70-75) and asserts `calls == []` (:86): open the CFF2 font in a separate test, so the `calls == []` assertion keeps covering only rejected fonts.
   - `tests/gui/test_main_window.py` (`test_unsupported_font_is_rejected`, :203-219) and `tests/gui/test_controller_errors.py` (`test_open_font_rejects_cff2`, :28-41): garbage-bytes tests already exist beside them (test_main_window.py:189-200, test_controller_errors.py:13-25), so do not add a third. Convert each into a positive test that opening `cff2_font_path` loads: the controller emits `font_loaded` and no `error`, the grid is non-empty, and save is enabled. Rename them `test_cff2_font_opens` and `test_open_font_accepts_cff2`.

6. **Force offscreen Qt in `tests/gui/conftest.py`.** Line 24 is `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`. The host desktop session exports `QT_QPA_PLATFORM=wayland;xcb`, so the default never applies and every GUI test opens real windows on the user's screen (it happened during planning). Change it to `os.environ["QT_QPA_PLATFORM"] = "offscreen"`. Also update the `cff2_font_path` and `variable_font_path` docstrings (lines 71 and 81), which say "unsupported by the core": after this stage CFF2 is supported, and variable fonts become supported in `variable-surfaces`.

## Proof (run once, report output)

```bash
QT_QPA_PLATFORM=offscreen uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_io.py tests/unit/test_review_io.py tests/gui/test_session.py tests/gui/test_main_window.py tests/gui/test_controller_errors.py
```

The orchestrator's verifier runs the full gate (whole suite, lint, format, mypy) after you return.

`tests/regression` must stay green: it pins static TrueType and CFF output bit for bit. Size limits: files ≤400 lines, functions ≤50 effective lines (`tests/regression/test_code_structure.py`). `converter.py` is 257 lines.
