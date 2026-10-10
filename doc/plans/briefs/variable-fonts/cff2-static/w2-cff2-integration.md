# W2: CFF2 integration tests (stage cff2-static, codex unit)

Tier: codex gpt-5.6-terra, effort xhigh. Do not run git. Do not touch `.loom/`.

You own exactly one file, which you create: `tests/integration/test_cff2_static.py`.

You may read:

- `tests/integration/test_stencilization_formats.py`: the pattern to follow;
- `tests/gui/conftest.py`: the `cff2_font_path` fixture, lines 70-77;
- `src/stencilizer/core/processor.py`: `FontProcessor.process`;
- `src/stencilizer/io/reader.py`: `FontReader`.

## Context

W1 (in parallel) adds CFF2 read-winding normalization and a `_update_cff2_glyph` writer to `src/stencilizer/io/converter.py`. Your tests exercise the result end to end. They will fail until W1 finishes; that is expected.

## Steps

1. Add a module-level helper `_cff2_font(tmp_path) -> Path`. It loads `tests/fixtures/CommitMono-Cosmix-700-Regular.otf` with `TTFont`, calls `fontTools.cffLib.CFFToCFF2.convertCFFToCFF2(font)`, saves to `tmp_path / "commitmono-cff2.otf"`, and returns the path.
2. Write these tests:
   - `test_cff2_process_keeps_cff2_table`: runs `FontProcessor(StencilizerSettings()).process(font_path=src, output_path=tmp_path / "out.otf")`. Asserts that `TTFont(out)` has `"CFF2"`, has no `"CFF "`, and that `stats.bridges_added > 0`.
   - `test_cff2_output_glyphs_have_no_islands`: for `O`, `zero` and `B` in the output, loads through `FontReader(out)` and asserts `GlyphAnalyzer().analyze(glyph, upm).get_islands() == []`.
   - `test_cff2_untouched_glyph_charstring_unchanged`: glyph `l` has no island. Assert that its decompiled charstring program equals the input's (`charstring.decompile(); charstring.program`).
3. Import style and typing as in `tests/integration/test_stencilization_formats.py`. Use the `tmp_path` fixture only.

## Done

One static check, run once (codex's sandbox has no network and a read-only uv cache, so `uv run` fails there; call the worktree venv directly):

```bash
.venv/bin/ruff check tests/integration/test_cff2_static.py && .venv/bin/mypy tests/integration/test_cff2_static.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/integration/test_cff2_static.py
```

It collects 3 tests. The orchestrator runs the tests after W1 returns. Lint rules units have missed before: quote `cast()` type arguments (ruff TC006), prefix unused stub parameters with `_` (ARG001), and keep `ruff format` clean.
