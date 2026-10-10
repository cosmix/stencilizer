# W3: FontWriter variable dispatch and name records (stage variable-writers, codex unit)

Tier: codex gpt-5.6-terra, effort xhigh. Do not run git. Do not touch `.loom/`.

You own exactly:

- `src/stencilizer/io/writer.py`
- `tests/unit/test_variable_writer_names.py` (create)
- `tests/unit/test_review_io.py`: only its writer-rejection test `test_writer_rejects_unsupported_fonts_without_output` (lines 41-50). Rewrite it; never delete it (the stage's test-integrity check fails when test declarations or assertions disappear). Leave the reader test alone.

Read-only anchors:

- `FontWriter` (writer.py:93; constructor `FontWriter(font, output_path)` at :105), `update_font_names` (:32), `_updated_font_name` (:68), and `_check_supported_format` (:25, called at :127 in `update_glyph` and :147 in `save`) in `src/stencilizer/io/writer.py`;
- `VariableGlyph` in `src/stencilizer/variable/model.py` (`vg.name` is `vg.default.name`).

Two functions are written in parallel by other workers. Import them by these exact signatures; do not implement them:

```python
from stencilizer.variable.write_gvar import write_truetype_variable_glyph  # (font: TTFont, vg: VariableGlyph) -> None
from stencilizer.variable.write_cff2 import write_cff2_variable_glyph      # (font: TTFont, vg: VariableGlyph) -> None
```

## Steps

1. **Guard the static path; stop rejecting at save.** Replace `_check_supported_format` with a guard in `update_glyph` only: when `"fvar" in font`, raise `FontFormatError(str(self._output_path), "variable fonts must use update_variable_glyph")` (use whatever attribute `__init__` stores the output path in). `save()` no longer rejects anything. Planning showed why the guard must stay: `domain_glyph_to_fonttools` on a variable TrueType glyph writes glyf and leaves gvar stale, so `save` raises a bare `AssertionError` that escapes `FontSaveError` (writer.py:151 catches only `OSError`), or the saved font crashes when drawn at a non-default location; for CFF2 the glyph silently stops varying.
2. Add `FontWriter.update_variable_glyph(self, vg: VariableGlyph) -> None`. Import the two writer functions INSIDE this method, not at module top: `stencilizer.io` → `writer` → `stencilizer.variable` → `reader` → `stencilizer.io.converter` would otherwise be a circular import, and the other workers' files may not exist yet when you run your check. Import `VariableGlyph` only under `if TYPE_CHECKING:`. The method:
   - raises `ValueError` if `vg.name` is not in `font.getGlyphOrder()`;
   - calls `write_truetype_variable_glyph(self._font, vg)` when `"glyf" in font` (the wiring check greps `write_truetype_variable_glyph\(` in writer.py);
   - calls `write_cff2_variable_glyph(self._font, vg)` when `"CFF2" in font`;
   - else raises `FontFormatError(<output path>, "unsupported variable outline format")`.
   Use whatever attribute name `__init__` stores the font in.
3. In `update_font_names`, build the set of PostScript-rule name IDs once, before the loop: `{6, 25}` plus, when `"fvar" in font`, every fvar instance `postscriptNameID` that is not `None` or `0xFFFF`. Apply the existing nameID 6 rule (suffix without spaces inserted before the first hyphen, appended when there is none) to every record whose nameID is in that set, exactly once per record. A second pass after the nameID 6 loop would double-suffix a record shared by two instances or by an instance and nameID 6, which the spec allows. Do not touch `subfamilyNameID` records, STAT or axis names.
4. In `_updated_font_name` (or a helper beside it), for fonts with `fvar` only: nameID 4 (full name) whose text starts with the family name (nameID 16 if present, else nameID 1) followed by a space or end of string gets the suffix inserted right after that family name. Inter's nameID 4 is "Inter Variable" with family "Inter Variable"; the existing rsplit rule (writer.py:75-77) would make it "Inter Stenciled Variable" while nameID 1 becomes "Inter Variable Stenciled". Fonts without `fvar` keep today's rule unchanged: tests/regression pins static output.
5. Tests in `tests/unit/test_variable_writer_names.py`, using `tests/fixtures/variable/Inter-VF-subset.ttf` (nameID 25 "InterVariable"; nine instances with postscriptNameIDs 280..296, e.g. 280 "InterVariable-Thin"; none points at 6 or 25):
   - `FontWriter(TTFont(path), tmp_path / "out.ttf").save()`, then nameID 25 == "InterVariableStenciled", 280 == "InterVariableStenciled-Thin", and nameID 4 == "Inter Variable Stenciled";
   - a font whose two instances share one postscriptNameID gets that record suffixed once;
   - `update_variable_glyph` raises `ValueError` for an unknown glyph name (build the `VariableGlyph` with `Glyph(metadata=GlyphMetadata("nope", None, 500, 0), contours=[])`, `supports=()`, `masters=()`, `axis_tags=("wght",)`);
   - `update_glyph` on a font with `fvar` raises `FontFormatError`.
6. Rewrite `test_writer_rejects_unsupported_fonts_without_output` in `tests/unit/test_review_io.py`: keep its name, parametrize `["fvar"]`, assert `update_glyph` raises `FontFormatError` and that no output file exists, and add an assertion that `save()` on that font writes the output.

Test-writing rules units have missed before: quote `cast()` type arguments (ruff TC006), prefix unused stub parameters with `_` (ARG001), annotate `dict.fromkeys` results, keep `ruff format` clean.

## Done

One static check, run once (codex's sandbox has no network and a read-only uv cache, so `uv run` fails there; call the worktree venv directly):

```bash
.venv/bin/ruff check src/stencilizer/io/writer.py tests/unit/test_variable_writer_names.py tests/unit/test_review_io.py && .venv/bin/mypy src/stencilizer/io/writer.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/unit/test_variable_writer_names.py
```

The orchestrator runs the tests. `writer.py` stays ≤400 lines (172 now), and every function stays ≤50 lines.
