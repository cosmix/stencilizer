
## PLAN-stencilizer-gui pressure test (2026-09-23)

- Codex workers were told to verify with `uv run pytest/mypy/ruff`; codex's workspace-write sandbox has no network, a read-only uv cache and an unwritable /tmp, and its preamble forbids verification. Every proof command would have failed. Now `.venv/bin/` static proofs; the orchestrator runs tests per wave.
- `loom stage complete`'s unwired-file scan cannot see dotted Python imports or top-level `tests/`; the plan had no wiring memory note, so completion would fail.
- Integration-verify was told to write PNGs to a mktemp dir, where the Read tool is blocked; moved to the session scratchpad with a spawn-safe `main()` guard.
- GUI saves in tests forked from a threaded process (DeprecationWarning, deadlock risk) and the production spawn path was never exercised; added an autouse spawn-context fixture.
- The rationale "saving onto the input corrupts it" was false (`TTFont.save` buffers in memory); the guard is kept as a product rule and extended to hard links (`samefile`).
- Closing the window mid-save would block the GUI thread in `waitForDone()`; close is now refused while busy.
- Prose-only promises: the `gui` extra, the spawn switch, the offscreen launch, the README update, and "one FontProcessor per controller" had no gate; wiring/acceptance entries and named tests added.
- gui-app ran a scoped pytest though it edits the shared pytest config; now the full `uv run pytest`.
- Weak tests: save tests used only default settings, the overwrite test only the literal path, no check that signals are delivered on the GUI thread, no busy-state reflection in the window, `main()` never ran past `--help`.
- W7/W8 module+test pairs risk the 540 s codex deadline; pre-split into module and test units.
- Smaller: incomplete import blocks, a wrong `@Slot` rationale, a float `drawLine` that fails mypy, a shared `/tmp/stencilizer-gui.log` that breaks for a second user, DEBUG log volume.

## PLAN-stencilizer-gui second review (codex, 2026-09-23)

- Assumed `FontSession` could reuse its classification at save time because only `skip_composite` matters; it also holds the outlines read at open, and `process` reopens the path, so a source edited or replaced after open would produce a mixed font. Now pinned by SHA-256, checked before and after the save.
- Treated "CFF2 and variable fonts are non-goals" as enforced; nothing rejected them, `FontReader` calls CFF2 "OpenType", and `_save_font` swallows the resulting write errors and still saves a renamed font. `FontSession.open` now rejects `fvar`/`CFF2`/no-outline fonts.
- Save tests trusted `processed_count + error_count == N`, which an all-failure save passes; swallowed glyph-write errors are not even counted. Every save test now asserts zero errors and reads the saved `O`/`B` back against the preview.
- The spawn fixture covered pytest saves, but no test ran `app.main`'s real start path through a save; added a fresh-interpreter subprocess test and a mid-save `shutdown()` test.
- README had string checks but no artifact, and the checked string `stencilizer[gui]` was not the real from-source install command (`uv pip install -e ".[gui]"`); now a section-aware check plus the artifact entry.

## Pressure test 2026-10-09: PLAN-variable-fonts.md

Omissions, mistakes and bad assumptions found in the plan as written, each grounded in code or a probe. All are folded into the plan and its briefs.

### Blocked execution

- The plan, briefs and spike were untracked, and main had uncommitted edits (pyproject.toml, uv.lock, tests/unit/test_io.py, README.md, knowledge files) to files the stages write. Worktrees branch from commits, so workers would not find their briefs. The plan claimed "All spike scripts are committed". Added a "Preconditions before loom init" section.
- No test-integrity plan. cff2-static, variable-writers and variable-surfaces rewrite or delete rejection tests, which raises TI events. The briefs said "delete it". Added a "Test integrity in this plan" table; every rejection test now becomes a positive test instead.
- variable-surfaces' dispatch `is_variable(reader.font)` breaks 8 tests that use plain `Mock()` readers (tests/unit/test_processor.py, test_processor_more.py). Those files were in acceptance but not in `files:`.
- W3 overlap removal matched pathops output at "1e-4 after rounding". pathops goes through float32, which is off by up to 1.2e-4 at 2048 UPM. The spike passed only because it used 8 subdivisions, which float32 stores exactly. With W2's tolerance-chosen counts, every Inter probe glyph was unmappable. W3 also could not test against the real flattener, so flattening moved to wave 1.
- The solver contract compared domain points to glyf points 1:1. The converter rotates contours and appends a closing duplicate, so Ubuntu 'o' has 34 domain points against 32 glyf points.
- gvar writer: rounding each delta separately gave a 2-unit error at the Ubuntu corner tuples, against a 1-unit contract. optimize(tolerance=0.5) can also reopen bridge gaps.
- CFF2 blend writer: the blend arguments lacked the trailing count of 1 (specializer asserts). The default preserveTopology=False drops points. Rounding relative arguments drifted by 13 units.
- A font with fvar and no gvar (the GUI fixture `variable_font_path`) made the reader raise KeyError.
- One VariationDataError in classification aborted a whole font (CLI exit 1, GUI cannot open it).

### Shipped broken without failing a gate

- `--instance` kept overlaps, so overlap-built counters (Inter A D P R e 4 &) were never bridged; the contract checked only advance width. Fix: `overlap=OverlapMode.REMOVE`, plus a contract check that cannot pass vacuously.
- gvar writer ignored hmtx lsb while xMin changed, so the outline shifts by the xMin change.
- gvar-keeps-phantom-deltas compared `.width`, which comes from HVAR, so a zero-phantom writer passed.
- Removing the writer's fvar check opened the static path to variable fonts (stale gvar, AssertionError on save, CFF2 losing blends). The guard moved into `update_glyph`.
- validation_locations was the full 3^n product: 1.6M analyzer runs per glyph for 13 axes, on the GUI thread. No contract pinned it, and the fixtures cannot tell peaks-only from a full grid.
- Variable process pool created in variable/processing.py escaped the tests' spawn patch, so it would fork from Qt threads. Cancellation and progress were unspecified. Now the static loop is generalized and reused.
- GUI normalization took no avar input (Inter wght 700 is 0.6 without avar, 0.54 with it). Composite previews ran the static pipeline. The AxisPanel would be unstyled: QWidget role=card gets no style, and QDoubleSpinBox gets no rule.
- `--list-islands` and `--dry-run` scanned raw default glyphs and missed overlap-built counters.
- Support.scalar reimplemented the OpenType rules instead of delegating to fontTools' supportScalar. The CFF2 vsindex scan read only the top-level program, but Cantarell's blends live in subrs.
- Several contracts could pass a no-op engine (masters-share-structure) or depended on unpinned behaviour.

### Under-specified

- Every stage gate ran a test subset. Writers and surfaces ran no goldens or integration tests, and surfaces skipped most GUI files. Now every gate runs the full suite with xdist.
- Wiring regexes: `_update_cff2_glyph\(` matches its own def line; `set_location\(` rejects the repo's direct-connect style.
- The core -> variable import cycle is certain (core/__init__.py imports the processor eagerly). The brief called it conditional.
- process_variable_font's signature dropped `classification` and `stats`.
- cli/app.py has no room: 393/400 lines, and `stencilize` is at 46/50 effective lines.
- Codex units were told to run `uv run`, which fails in codex's sandbox. Contract sessions were not told the lint and typing rules frozen files must meet. The help check depended on terminal width and color. The plan path gets renamed during the run.
- Knowledge claims assumed "correct at HEAD", but HEAD's concerns.md said nothing in the core rejects fvar/CFF2 (false), and "Swallowed glyph-write failures" was stale too.

### Process mistake during the pressure test

- A teammate's GUI timing run opened real windows on the user's desktop. tests/gui/conftest.py only setdefaults QT_QPA_PLATFORM, and the session exports `wayland;xcb`. The plan now forces offscreen in conftest and in every acceptance command; recorded in mistakes.md.
- Stage lint and type gates checked `src tests` only, but CI (`checks.yml`) also covers `packaging/` (tracked since 46b29d9), so stages could merge lint or type failures there. Every stage gate now uses CI's scope.
- The `.notdef` the user saw stencil badly (Roboto, four triangular counters) comes from a static surgery defect that the variable engine inherits. It is recorded in concerns.md "Sequential bridges cut through sibling counters" and declared out of scope in the plan.

## PLAN-variable-fonts second review (codex, 2026-10-09)

Several findings targeted the pre-pressure-test draft and were already folded in (solver point mapping, blend count, fvar-only reader, avar input, in-place stats, subroutine-following vsindex, shared spawn pool, full-suite gates with `packaging`). The rest were real:

- Goals promised "every bridged glyph free of islands at every location" while the engine brief allowed partial success (`allowed_islands=unbridged_count`), which is the static pipeline's actual rule. The goal now states partial success explicitly, and a deterministic partial case is tested.
- The no-op outcome counted 0 unbridged islands when overlap removal failed before the hierarchy step, and a glyph skipped for unsupported variation data lost its counter from the stats. Counts now come from the overlap-merged, flattened or raw default, and classification carries `unsupported_islands`.
- Validation ran on unrounded geometry while the writers rounded afterwards, so the saved font was never the validated one. Rounding moved into the engine (`variable/rounding.py`); the writers store its values; an end-to-end contract rereads every modified glyph of all three fixtures and checks the preserved tables.
- The gvar writer assigned into `font["gvar"]` unconditionally, so an fvar-only font failed at save even though the reader accepted it. Only open was tested.
- `instantiate_static` turned on `updateFontNames` whenever STAT existed; Inter wght=650 (in range, unnamed in STAT) raises `ValueError`. Naming now falls back; the contract only tested 700.
- `gvar-keeps-phantom-deltas` could pass on all-zero phantom deltas; it now asserts a non-zero input delta (Ubuntu `o` pp2: −129).
- CFF2 reading allowed several `vsindex` values per glyph to go unnoticed, and the only fixture had one VarData, so a VarData-0 assumption could not fail. Synthetic non-zero index, Private-default and no-blend cases added; the writer handles `supports=()`.
- Composite previews ignoring the sliders was documented only in a docstring. It is now a visible UI rule (disabled sliders plus a note) with a contract, and the fixtures gained a composite (Á).
- The preview cache key omitted `GeometryConfig`, had no size bound, and the survey would have filled it with every glyph's outlines.
- Artifact lists omitted most implementing modules; the CLI axes line had no test.
- Line anchors pointed into files earlier stages edit; the briefs now name symbols and tests.
- The overview stated island sets "cannot differ across masters" as a format fact; it was a measurement on three fonts.
- Not adopted: running stage gates with coverage. Coverage has no threshold in pyproject or CI, and it tripled the xdist run (198 s against 69 s), leaving no room under the 300 s cap. CI's Python 3.11 job was uncovered at runtime; integration-verify now runs the suite under 3.11.
