# Concerns & Technical Debt

> Technical debt, warnings, issues, and improvements needed.
> Lists only extant issues: delete an entry once it is fixed.

## Unsupported font formats

CFF2 write and variable fonts are unsupported; see stack.md "Supported font formats". `FontReader.load` raises `FontFormatError` for an `fvar` or `CFF2` table (src/stencilizer/io/reader.py:54-57), `FontWriter` repeats the check in `_check_supported_format` (src/stencilizer/io/writer.py:25-29), and the GUI rejects both in `unsupported_reason` (src/stencilizer/gui/session.py:63-66, called from `FontSession.open`). Past those guards nothing handles them: `fonttools_glyph_to_domain` reverses winding only when `"CFF " in font` (io/converter.py:32), so a CFF2 outline would read inside-out, and `domain_glyph_to_fonttools` raises `NotImplementedError` for any font without `glyf` or `CFF ` (io/converter.py:65-72).

## Oversized test module

tests/unit/test_io.py is 490 lines, over the 400-line file limit. tests/regression/test_code_structure.py enforces the 50-line function limit on src/ only, so nothing fails when a test file passes 400 lines (tests/gui/test_session.py and test_controller.py did during integration and were split by hand). Check `wc -l tests/gui/*.py` by hand.

## Unknown config fields are silently ignored

`BridgeConfig` and `GeometryConfig` (src/stencilizer/config/settings.py) use pydantic's default `extra="ignore"`, so a removed or misspelled field such as `BridgeConfig(min_bridges=1)` is accepted and has no effect.

## Ignored `process_glyph` parameter

`process_glyph` (src/stencilizer/core/processor.py) still accepts a fourth `reference_stroke_width` parameter and ignores it, because the regression helpers in tests/regression/_golden.py pass it positionally. Drop it together with that call site.

## Native-UPM output change not visually reviewed

UPM scaling made fonts not at 1000 UPM behave as the 1000-UPM tuning scaled. For Roboto (2048) and Lato (2000), 75 of 1009 island glyphs changed output (17 gained contours, 5 lost contours, 53 kept the count). The changes are intended but nobody has inspected the rendered glyphs.

## Unchecked classification reuse

A caller-supplied `classification` passed to `FontProcessor.process` is used without checking it came from the file `process` reopens (core/processor.py:226), so an edited source mixes old outlines with new tables; only the GUI guards this, with its source digest check (architecture/gui.md "Save safety"). Tests of a save must read the output back rather than trust the stats. A failed `writer.update_glyph` in `_save_font` raises `FontSaveError` and publishes nothing (core/processor.py:349-355).

## Tests write log files into the working directory

`setup_logging` names the log `stencilizer_%Y%m%d_%H%M%S.log` in the cwd when `log_file` is None (src/stencilizer/utils/logging.py:69-71), and tests that build `FontProcessor(settings)` without a `log_file` hit it (tests/unit/test_processor.py, tests/integration/test_stencilization*.py, test_e2e_output.py, tests/regression/_golden.py:195). One full `uv run pytest` leaves about 12 such files at the repo root; `*.log` is gitignored (.gitignore:65). Fix: give those fixtures a `tmp_path` log file, as the tests/gui `processor` fixture does.

## Composite glyphs read without contours

`fonttools_glyph_to_domain` (io/converter.py) records only outline segments, so a composite glyph (Roboto `Aacute` = `A` + `acute`) has zero contours and `classify_glyphs` files it as "empty glyph". The converter reads `fonttools_glyph._glyph`, which fontTools' `_TTGlyphGlyf` glyph-set objects lack, so `Glyph.is_composite()` is always False and `ProcessingConfig.skip_composite` never fires (dead setting). The saved font still bridges composites through the glyph they reference. The GUI no longer depends on the reader: `gui/composites.py` resolves composites from the fontTools glyph set and `FontSession.display_glyphs` lists them (see architecture/gui.md "Composites in the grid"); the core and CLI still see them as empty.

## Fork warnings when GUI and pool tests share a process

Measured at 111d115 with coverage on: a single `uv run pytest` printed 42 `DeprecationWarning` "multi-threaded, use of fork()" (Qt threads from tests/gui alive when later tests fork `ProcessPoolExecutor` workers). At the integration-verify commit, `uv run pytest --no-cov -q -p no:cacheprovider tests` in one process (Python 3.13.1, 313 tests, 224 s) printed none, and a short gui-then-pool mix printed none. Not re-measured with coverage on, so the warnings may still appear there.

## CommitMono .notdef renders as a solid box

Under Auto, CommitMono `.notdef` (frame, hole, inverted "404" digits) stencils into overlapping full-width outer rectangles, so preview and saved glyph render as a solid black box (20 contours, output identical to the base before per-glyph directions; sha256 prefix c1f9a7e72f56d125). The defect is in the inverted-island path (`core/surgery_nested.py`) and is outside the direction work because Auto output is pinned. Repro: `process_glyph` on that glyph with `BridgeConfig()`. Which real glyphs reach `_process_inverted` is unknown; the synthetic filled encircled digit never does (patterns/bridge-algorithm.md).

## Full test suite near the acceptance time cap

`uv run pytest --no-cov -q -p no:cacheprovider` ran 366 tests in 240 s pytest / 241 s wall on the gui-beautify tree (364 tests took 265 s in an earlier run: run-to-run spread is about 25 s), against loom's 300 s limit per acceptance command; it was 319 tests in 225 s at c888109. pytest-xdist is not installed, so the suite cannot be parallelised. The margin is 35-60 s: the next plan that adds GUI tests should split `tests/gui` across acceptance commands or ask for a higher cap.

## Duplicated GUI test fixtures

The `window` fixture (`GuiController` plus `MainWindow`, tests/gui/conftest.py:143) is repeated across five `tests/gui` files and `_load_font` across three; one of them is the frozen `test_beautify_contracts.py`, which nothing may edit. Move both into `conftest.py` in a plan that owns those files.

## Untested theme details

`apply_theme`'s `app.setStyle("Fusion")` (theme.py:320) has no direct test, because `style().name()` is `''` once a stylesheet is set. `HeaderBar` tooltips, the placeholder `font_details_label` text and the cursor shape are untested (cosmetic). The direction picker's source label is blank before a font loads (an empty-state placeholder was deferred because `direction_picker.py` was out of scope). Card frames use `$border` while interactive controls use `$border_strong` on purpose (theme.py:79): grouping chrome is quieter than control edges; keep that distinction if either changes. The grid has no `::item:focus` rule: the selection tint doubles as the focus indicator (theme.py:206). `accent_pressed` and `focus` in `_derived_colors` (theme.py:274) share one `_blend(accent, text, 0.3)` expression and are equal in both themes; they are separate roles, so deriving one from the other would couple them.

## Sandbox fingerprint and reviewer hand-back need loom fixes

Two plan stages hit the same two gate failures (mistakes/review-and-completion-gates.md), so prose rules are not enough. Proposals: the finish and contract-freeze fingerprint should ignore non-regular files in the worktree root, or those commands should run outside the sandbox; the review-harvest hook should read a review block from a hand-back message, or the reviewer agent definition should drop that tool. Until then a version 2 stage run from a sandboxed session stalls on the review gate.

## GUI tests honour an inherited QT_QPA_PLATFORM

`tests/gui/conftest.py:24` uses `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`, so in a desktop session that exports `QT_QPA_PLATFORM=wayland;xcb` every GUI test opens real windows (and in a sandbox without a display socket Qt cannot start). Run GUI tests as `QT_QPA_PLATFORM=offscreen uv run pytest ...` until conftest assigns the variable outright; see mistakes.md "GUI tests open real windows when QT_QPA_PLATFORM is set".

## Sequential bridges cut through sibling counters

When an island group has no clear gap and no spanning bridge (`surgery_groups.arrangement` falls back to vertical, `_spanning` fails because the islands share no x range), `_sequential` (core/surgery_groups.py:108-135) bridges each island in turn, and every bridge band crosses the sibling counters of the same parent without splitting them. Holes are never obstructions (`geometry_crossings.is_bridge_path_clear`, core/geometry_crossings.py:112-119), `bridge_nested._nested_status` splits only the bridged island's own children, `SurgeryContext.merge` passes the original contours rather than the current pieces, and `_containing_piece` matches later islands by bbox centre, so a triangle whose centre lands in a gap is appended unchanged. Roboto `.notdef` (a box with four triangular counters, the first glyph the GUI previews for Roboto, tests/gui/test_main_window.py:66-78) comes out with 0 islands but stray holes crossing the bridge gaps, which render as black wedges. A prototype that splits overlapped sibling holes, restricted to siblings so nested rings (◎, ℗) are untouched, fixed `.notdef`, `naira`, `won`, `uni20BB`, `dollar`, `uni0E3F`, `uni2318` and `uni25A9`, and reduced glyphs with leftover islands or crossing contours from 69 to 59 in Roboto, 76 to 70 in Lato and 71 to 70 in CommitMono. Fixing it changes the static goldens, so it needs its own plan; the variable-font engine reuses this surgery on the default master and inherits the defect.
