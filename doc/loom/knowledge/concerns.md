# Concerns & Technical Debt

> Technical debt, warnings, issues, and improvements needed.
> Lists only extant issues: delete an entry once it is fixed.

## Unsupported font formats

CFF2 write and variable fonts are unsupported; see stack.md "Supported font formats". Nothing in the core rejects them: `FontReader.load` only checks the file exists and `FontReader.format` labels CFF2 "OpenType", so `classify_glyphs` succeeds (26 island glyphs on a CFF2 copy of CommitMono) and every glyph write raises `NotImplementedError` in `domain_glyph_to_fonttools` (`io/converter.py`). Callers must check for `fvar`/`CFF2` themselves (the GUI plan does so in `FontSession.open`).

## Oversized test module

tests/unit/test_io.py is 490 lines, over the 400-line file limit. src/ is within limits (enforced by tests/regression/test_code_structure.py, which checks src/ only).

## Unknown config fields are silently ignored

`BridgeConfig` and `GeometryConfig` (src/stencilizer/config/settings.py) use pydantic's default `extra="ignore"`, so a removed or misspelled field such as `BridgeConfig(min_bridges=1)` is accepted and has no effect.

## Ignored `process_glyph` parameter

`process_glyph` (src/stencilizer/core/processor.py) still accepts a fourth `reference_stroke_width` parameter and ignores it, because the regression helpers in tests/regression/_golden.py pass it positionally. Drop it together with that call site.

## Native-UPM output change not visually reviewed

UPM scaling made fonts not at 1000 UPM behave as the 1000-UPM tuning scaled. For Roboto (2048) and Lato (2000), 75 of 1009 island glyphs changed output (17 gained contours, 5 lost contours, 53 kept the count). The changes are intended but nobody has inspected the rendered glyphs.

## Swallowed glyph-write failures and unchecked classification reuse

`FontProcessor._save_font` (`core/processor.py`) logs a failed `writer.update_glyph` and continues: the failure is not counted in `ProcessingStats.error_count`, and `writer.save()` still writes a font renamed "Stenciled". A caller-supplied `classification` passed to `process` is used without checking it came from the file `process` reopens, so an edited source mixes old outlines with new tables. Tests of a save must read the output back rather than trust the stats.
