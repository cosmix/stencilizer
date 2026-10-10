# Plan: Variable font and CFF2 support

## Overview

Stencilizer rejects variable fonts (`fvar`) and CFF2 outlines today (`io/reader.py` `FontReader.load`, `io/writer.py` `_check_supported_format`, `gui/session.py` `unsupported_reason`). This plan adds:

- static CFF2 read and write;
- a variable-font engine that runs the unchanged static bridge surgery once, on the default master, and replays it in every master so all masters share one point structure;
- gvar and CFF2-blend writers that rebuild variation deltas against each glyph's original regions;
- CLI processing of variable fonts, a `--instance` option that pins a static instance, and GUI open, save and axis-slider preview.

The design in the untracked `variable-fonts.md` (4-point insertion, `DeltaGenerator`, "conservative union" of islands across masters) is superseded: surgery cuts notches and flattens curves, so it never inserts a fixed point count. Its island union is also unnecessary for the probed fonts, whose island sets were identical at every axis extreme (probe below). That is a measured property of those fonts, not a format invariant: gvar fixes point structure, but island membership depends on winding and containment (`core/analyzer.py` `_classify_contours`, `_find_parents`, `_strictly_contains`), which moving points can change. The engine therefore validates every glyph at every validation location instead of assuming it.

## Goals

- A variable TrueType or CFF2 font in, a variable stenciled font out, with fvar, avar, STAT, HVAR and MVAR untouched.
- Partial success follows the static pipeline's rule. Static surgery can bridge some islands of a glyph and leave others (`core/surgery.py` `transform_with_outcome` returns `bridge_count` and `unbridged_count` together, and the static writer stores that glyph). The engine replays what the default-master surgery bridged, and the glyph is written only when, at every validation location, its instance has at most `unbridged_count` islands, i.e. no more than the default surgery left. A glyph with no unbridged island is therefore free of islands everywhere.
- Glyphs whose replay fails (unmappable, invalid at any location, a `VariationDataError` in any step) are written unchanged and their default-master islands are counted in `ProcessingStats.unbridged_count`. They are never written half-replayed.
- Validation runs on the exact values the writers store. The engine rounds the replayed glyph the way the target format will (`variable/rounding.py`) before validating it, and the writers store those values, so a glyph cannot pass validation and then gain an island in the saved font. An end-to-end contract (`variable-output-valid-everywhere`) rereads every modified glyph of all three fixtures from the saved file and checks it at every validation location.
- Static TrueType and CFF output is unchanged. `tests/regression` pins it bit for bit.
- Glyphs the engine does not bridge keep their original glyf and gvar data, or CFF2 charstring, byte for byte. Variation data the engine cannot handle in one glyph (singular supports, incompatible masters, curves past the subdivision cap, an unsupported CFF2 vsindex) raises `VariationDataError`, which leaves that glyph unchanged, counts its default-master islands as unbridged, and never aborts the font. Font data fontTools itself cannot decode stays a font-level failure, as in the static pipeline.
- Composite glyphs in a variable GUI session preview at the default axis location, and the window says so (see `variable-surfaces`).
- Non-goals: avar2-specific handling; composite-glyph surgery (composites keep referencing their bridged bases, as in static fonts); and restoring curves on contours that surgery leaves untouched in a bridged glyph. The engine flattens every contour of a glyph before surgery, so those contours come out as polygons, while the static pipeline re-appends them with their curves (`core/surgery.py:75-77`). Planning measured this on 3 Ubuntu glyphs (uni0221, uni0247, uni2116) and 22 Inter glyphs (Ohorn, ohorn, Q_rthook, ...).
- Also out of scope: static surgery defects that the engine inherits, because it runs the unchanged static surgery on the default master. The known one is knowledge `concerns.md` "Sequential bridges cut through sibling counters" (Roboto `.notdef`: stray holes crossing the bridge gaps). Fixing it changes the static goldens, so it needs its own plan. Validation still rejects a variable replay whose instances keep islands.

## Evidence gathered while planning

The spike scripts live under `doc/plans/briefs/variable-fonts/spike/` (reference only, not production code). They, the briefs and this plan must be committed before `loom init` (see "Preconditions before loom init").

**Island sets matched across masters in the probed fonts** (`spike/probe.py`). At the extremes of every axis, the analyzer found identical island sets for every probed character in Inter (opsz, wght), Ubuntu (wdth, wght) and Monaspace Neon (wght, wdth, slnt). This is evidence about these fonts only; validation still checks every location.

**Overlap-built counters** (`spike/probe.py`). At the default master, Inter's A, D, P, R, e, 4 and & and Monaspace's A B D P R a b d e g p q 4 6 8 9 & @ show no island: their counters come from overlapping or self-overlapping contours. Ubuntu shows islands for all of them.

**CFF2 winding** (`spike/probe.py`). Cantarell-VF (CFF2) shows no islands even on `O`, because `fonttools_glyph_to_domain` reverses contours only when `"CFF " in font` (`io/converter.py`, the `is_cff` check). CFF2 needs the same reversal.

**Mapping surgery output back to its input** (`spike/vlib.py` `mapping`). On Ubuntu[wdth,wght]'s default master, uniformly pre-flattened (8 lines per curve segment), 129,868 surgery output vertices equal input vertices and 4,861 lie on input edges. 4 vertices in 2 glyphs mapped to neither.

**Fixed-t replay fails** (`spike/vlib2.py` `replay2` without realignment; see `doc/loom/knowledge/mistakes.md` "Fixed-parameter replay breaks bridge-cut coincidence across masters"). Keeping each cut point at the same edge parameter t in every master left islands at some axis extreme in 554 of 561 bridged Ubuntu glyphs. Surgery output makes cut hole pieces touch their outer piece along the bridge line, and fixed t moves those points off one line.

**Per-master realignment works** (`spike/vlib3.py` `replay3`, `spike/vlib4.py` `lines`, `spike/run4.py`). Each bridge line's coordinate is recomputed per master, the cut point is re-intersected with the polyline within ±12 edges of the default edge, and same-contour vertices on the wrong side are projected onto the line. Measured at the 6 locations {each axis at ±1, all axes at +1, all axes at −1}:

| Font | Bridged glyphs | Failing at some location | Unmappable |
| --- | --- | --- | --- |
| Ubuntu[wdth,wght] | 561 | 23 | 1 |
| InterVariable | 576 | 27 | 0 |

These counts include composites, which the spike decomposed. The engine never processes composites (the converter reads them with 0 contours and `read_variable_glyph` returns `None`): simple glyphs with an island at the default number 229 in Ubuntu and 162 in Inter, of which the static surgery bridges 223 and 162.

**Compatible overlap removal works** (`spike/ovl.py`). Flattened default polygons are merged with `pathops.simplify(path, clockwise=True)` (skia-pathops 0.9.2 installs for Python 3.13.1). Each output vertex maps to an input vertex or to the crossing of two input edges, which is recomputed per master, and the realigned surgery replay follows. Results on the character set `ADPRe4&BOabdgopq0689`:

- Every glyph is bridged and valid at all locations (Inter and Ubuntu: every combination of {−1, 0, +1} on each axis, excluding the default; Cantarell: wght ±1), except:
  - `ampersand` fails at 3 of 8 locations in Inter and in Ubuntu;
  - Cantarell `e` fails at wght=+1;
  - `four` gets an island but no bridge from the static surgery in all three fonts.
- Inter `A`, `D`, `P`, `R` and `e` become bridged at 8 of 8 locations.

**fontTools 4.66.0 APIs used** (introspected):

| API | Signature or fact |
| --- | --- |
| `TTFont.getGlyphSet` | `(preferCFF=True, location=None, normalized=False, recalcBounds=True)` |
| `TupleVariation.optimize` | `(origCoords, endPts, tolerance=0.5, isComposite=False)` |
| `TupleVariation.calcInferredDeltas` | `(origCoords, endPts)` |
| `fontTools.varLib.instancer.instantiateVariableFont` | `(varfont, axisLimits, inplace=False, optimize=True, overlap=..., updateFontNames=False, *, downgradeCFF2=False, static=False)` |
| Ubuntu `o` gvar | 5 tuples, including two wdth+wght corner tuples (wdth −1 × wght −1 and wdth −1 × wght +1), with sparse (`None`) deltas |
| Cantarell CFF2 | one VarData over 2 regions (wght (−1, −1, 0) and (0, 1, 1)), one FDArray entry, no FDSelect, subroutinized charstrings whose `blend` operators sit only inside subrs; 12 of 1,322 glyphs never blend |
| `fontTools.varLib.models.supportScalar` | OpenType rules (`ot=True`): an axis is ignored when peak is 0, start > peak or peak > end, or start < 0 < end (`models.py:180-187`); the glyph sets that produce masters use it |
| `TTFont.getGlyphSet(normalized=True)` | skips avar (`ttFont.py:1321-1322`), so supports and locations share post-avar space; a glyf glyph draws at offset `hmtx lsb − xMin` (`ttGlyphSet.py:240-247`); with HVAR present `.width` comes from HVAR, not phantom points |
| `fontTools.cffLib.specializer` | a blended argument is `[default, d_0, …, d_n−1, 1]` (trailing blend count, asserted at `specializer.py:498`); `preserveTopology=False` (default) drops zero-length segments |
| `T2CharString.draw(pen, blender)` | the blender is called as `blender(vsIndex, deltas)` once per blended operand, following subrs; the decompiler's `vsIndex` starts at 0 and ignores `Private.vsindex`, while `PrivateDict.getNumRegions()` does default to `Private.vsindex` (`psCharStrings.py:338, 497-520`, `cffLib/__init__.py:2735-2742`), so a glyph relying on a non-zero Private default is read inconsistently by fontTools itself |
| skia-pathops 0.9.2 | `cp310-abi3` wheels for Linux x86_64 and macOS arm64; `simplify(path, fix_winding=True, keep_starting_points=True, clockwise=False)`; coordinates round-trip through float32 (up to 1.2e-4 off at 2048 UPM); no `py.typed` |
| `instantiateVariableFont` | needs a `TTFont`; clamps out-of-range values silently and raises a bare `KeyError` for an unknown axis; keeps overlaps unless `overlap=OverlapMode.REMOVE`; keeps CFF2 unless `downgradeCFF2=True`; with `updateFontNames=True` it raises `ValueError: Cannot find Axis Values {'wght': 650}` for an in-range value that no STAT AxisValue names (Inter: wght=700 succeeds, wght=650 raises; `varLib/instancer/names.py:73-77, 123-124, 161-164`) |
| Ubuntu `o` phantom deltas | non-zero: after `calcInferredDeltas`, pp2 moves by −129 (wdth −1), −11 (wght −1), +22 (wght +1), −5 and +8 (the two corner tuples); pp1, pp3 and pp4 are 0 |
| Fixture tables | Inter has MVAR and HVAR; Ubuntu has HVAR and no MVAR; Cantarell has MVAR and HVAR. Ubuntu and Inter `Aacute` are composite glyphs; Cantarell's `Aacute` is a plain CFF2 outline |

**Domain points versus glyf points.** `fonttools_glyph_to_domain` rotates each contour to its first on-curve point, appends a copy of point 0 when the last segment is a curve, and prepends an implied on-curve midpoint when a contour has no on-curve point (`io/converter.py:75-128`). Ubuntu `o` has 34 domain points against 32 glyf points. Structure is still identical across masters (0 mismatches over 1,023 simple Inter glyphs), and solved deltas mapped back to glyf points equal the gvar deltas after `calcInferredDeltas` within 2.8e-14.

**Gate baseline at HEAD 4254909** (run from the main checkout):

| Command | Result |
| --- | --- |
| `uv run pytest --no-cov -q -p no:cacheprovider -W error::DeprecationWarning tests/regression tests/unit tests/integration tests/test_domain_models.py` | 205 passed, 1 failed, 154 s |
| `uv run pytest --no-cov -q -p no:cacheprovider tests/gui` | 200 passed, 429 s while another pytest run competed for CPU |
| `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy` | exit 0 |
| `loom knowledge check --strict --baseline doc/loom/knowledge/check-baseline.txt` | clean |
| `uv run pytest --no-cov -q -p no:cacheprovider` (whole suite, serial) | 405 passed, 1 failed, 601 s |
| `uv run --with pytest-xdist pytest -n 16 --no-cov -q -p no:cacheprovider` | 406 passed, 69 s (working tree with the uncommitted test fix below) |
| `env QT_QPA_PLATFORM=offscreen uv run --with pytest-xdist pytest --numprocesses=16 -q -p no:cacheprovider` (pytest's configured coverage on; HEAD 770240f, second review) | 406 passed, 198 s, 87 % line coverage |

The one failure is `tests/unit/test_io.py::TestCffGlyphUpdate::test_update_cff_glyph_passes_private_and_global_subrs`. It expects `getCharString(private=..., globalSubrs=...)`, but `_update_cff_glyph` deliberately passes `optimize=False` (knowledge `patterns/bridge-algorithm.md` "Preserve CFF preview vertices").

An uncommitted edit to `tests/unit/test_io.py` in the main checkout, made outside this planning session, already adds `optimize=False`. Stage `cff2-static` owns the repair either way: it is a no-op if that edit is committed before `loom init`.

The serial suite exceeds the 300 s acceptance cap, so `cff2-static` adds pytest-xdist and every later gate runs the suite with `--numprocesses=16`.

## Preconditions before loom init

Loom worktrees branch from a commit, so anything untracked or uncommitted in the main checkout is absent from every stage. At the end of the pressure test (HEAD 770240f, which already carries `packaging/`, `.github/`, the `pyproject.toml` build group, `uv.lock`, `README.md` and the `optimize=False` repair in `tests/unit/test_io.py`), `git status` showed:

- untracked: this plan and `doc/plans/briefs/variable-fonts/` (every brief the stage tables point to, and the spike);
- modified: `doc/loom/knowledge/{INDEX,architecture,concerns,mistakes}.md`. Only the working tree has the mistake entries "Fixed-parameter replay breaks bridge-cut coincidence across masters" and "GUI tests open real windows when QT_QPA_PLATFORM is set", the corrected `concerns.md` "Unsupported font formats" (HEAD still says nothing in the core rejects these fonts), and the new concerns "GUI tests honour an inherited QT_QPA_PLATFORM" and "Sequential bridges cut through sibling counters".

Commit them on main before `loom init`. Knowledge-bootstrap and knowledge-distill edit the knowledge files, and a merge into a dirty main refuses on overlapping files. Other untracked files (root `CLAUDE.md`, `.claude/`, `cff2.md`, `variable-fonts.md`, `tests/fixtures/Lato-Black-Stenciled.ttf`, `doc/plans/codex-PLAN-variable-fonts.md`) are the user's call; nothing in this plan needs them.

CI (`.github/workflows/checks.yml`) lints and type-checks `packaging/` too, so every stage gate runs `ruff check src tests packaging`, `ruff format --check src tests packaging` and `mypy src/stencilizer tests packaging` (pyproject's mypy `files` omits `packaging`).

CI's test job runs `uv run pytest` serially on Python 3.11 and 3.13. The stage gates run the same suite (every test file, nothing scoped) with two deliberate differences:

- `--numprocesses=16`: the serial suite takes about 600 s, past the 300 s acceptance cap.
- `--no-cov`: pytest's `addopts` turn coverage on, but neither pyproject nor CI sets a coverage threshold, so coverage gates nothing. With coverage the xdist run measured 198 s against 69 s without, which leaves no room under the cap for the tests this plan adds.

Python 3.11 is covered statically on every stage by ruff `target-version = "py311"` and mypy `python_version = "3.11"`, and at runtime by one full suite run in integration-verify (`UV_PROJECT_ENVIRONMENT=.venv/py311`, inside the git-ignored `.venv/`; CPython 3.11.10 is installed under `~/.local/share/uv/python`).

## Test integrity in this plan

Completion compares each stage's test files with its base and raises an event when test declarations or assertions fall, or an assertion line present at base is removed or changed. This plan rewrites rejection tests whose subject stops being true, so expect `TI-edit` events, and only those, in these places:

| Stage | Test | Change |
| --- | --- | --- |
| cff2-static | `tests/unit/test_review_io.py` `test_reader_rejects_unsupported_fonts_before_exposing_font`, `test_writer_rejects_unsupported_fonts_without_output` | `["fvar", "CFF2"]` parameters become `["fvar"]` |
| cff2-static | `tests/gui/test_session.py` `test_open_rejects_unsupported_fonts` | CFF2 rejection becomes a successful open; message text gains "CFF2" |
| cff2-static | `tests/gui/test_main_window.py` `test_unsupported_font_is_rejected`, `tests/gui/test_controller_errors.py` `test_open_font_rejects_cff2` | CFF2 rejection becomes "CFF2 opens" |
| cff2-static | `tests/unit/test_io.py` `TestCffGlyphUpdate.test_update_cff_glyph_passes_private_and_global_subrs` | expects `optimize=False` (only if the precondition commit did not include it) |
| variable-writers | `tests/unit/test_review_io.py` `test_writer_rejects_unsupported_fonts_without_output` | writer rejection moves from `save` to `update_glyph` |
| variable-surfaces | `tests/unit/test_review_io.py` `test_reader_rejects_unsupported_fonts_before_exposing_font` | reader rejection becomes a successful variable load |
| variable-surfaces | `tests/gui/test_session.py` `test_open_rejects_unsupported_fonts` | fvar rejection becomes a successful open with one `wght` axis |
| variable-surfaces | `tests/unit/test_processor.py`, `tests/unit/test_processor_more.py` | `mock_reader.font = MagicMock()` added; no assertion changes |

Line numbers in this plan and its briefs are taken at HEAD 770240f. A file an earlier stage edits has moved by the time a later stage reads it, so every worker locates code by the symbol or test name given beside the number and treats the number as a hint.

Every rewritten test keeps its declaration and replaces each dropped assertion with a positive one in the same file. A `TI-assert` or `TI-decl` event means an assertion or test was lost: restore it, never dispute it. Dispute the `TI-edit` events of the rows above together, once, with `loom stage dispute-integrity <stage-id> --event <id> ... --reason "rejection tests rewritten because the plan makes <format> supported; see the plan's Test integrity table"`.

## Execution Diagram

```mermaid
graph LR
    knowledge-bootstrap --> cff2-static
    cff2-static --> variable-engine
    variable-engine --> variable-writers
    variable-writers --> variable-surfaces
    variable-surfaces --> integration-verify
    integration-verify --> knowledge-distill
```

The chain is serial because each stage needs the previous one merged (Q1 per stage below).

## Stages

### 0. knowledge-bootstrap: short audit

The knowledge base already describes this codebase (`loom knowledge check` is clean at HEAD), so this is a light audit. It does three things:

- checks the sections this plan contradicts once it lands (`architecture.md` "Font format I/O" and "Font format and error boundary", `concerns.md`'s first entry, `stack.md` "Supported font formats") against the tree. The pressure test rewrote the first three to match the tree (reader and writer reject fvar and CFF2 at reader.py:54-57 and writer.py:25-29); the audit confirms they arrived with the precondition commit;
- records the variable-font facts from the planning probe in `concerns.md`;
- confirms the mistake entry "Fixed-parameter replay breaks bridge-cut coincidence across masters" is present.

Model override to sonnet: a short audit needs no opus.

### 1. cff2-static: static CFF2 read and write

**Why a stage:**

- Q1. `variable-engine`'s reader reads CFF2 outlines through `fonttools_glyph_to_domain` and needs this stage's CFF2 winding reversal merged. If the engine reversed contours itself, it would double-reverse once this stage merged. `variable-writers` builds the CFF2 blend writer on this stage's CFF2 table handling.
- Q2. This stage and `variable-surfaces` both edit `io/reader.py`, `gui/session.py` and `tests/gui/test_session.py`.

Work:

- Reverse CFF2 contours on read, as is done for `CFF `.
- Add `_update_cff2_glyph` beside `_update_cff_glyph` and dispatch to it; the private dict comes from `top_dict.CharStrings.getItemAndSelector(name)` and `FDArray[fd or 0].Private` (CFF2 has no top-level Private).
- Lift the CFF2-only rejection in reader, writer and GUI session. fvar stays rejected until `variable-writers` (writer) and `variable-surfaces` (reader, GUI).
- Rewrite the tests that assert the CFF2 rejection into positive tests (see "Test integrity in this plan").
- Repair the HEAD-red CFF test.

A planning probe emulated this stage in-process on a CFF2 conversion of CommitMono: `FontProcessor.process` added 580 bridges, the output kept `CFF2` and no `CFF `, `O`, `zero` and `B` lost their islands, `l`'s charstring program was unchanged, and `FontSession.open` listed 467 island glyphs.

Risk walk:

| Area | Contract |
| --- | --- |
| External data (fontTools CFF2 structures) | `cff2-static-roundtrip-writes-bridges` |
| Untrusted input (CFF2 winding) | `cff2-read-normalizes-winding` |
| Reachability (GUI opens CFF2) | `gui-session-opens-cff2` |

### 2. variable-engine: replay engine in `src/stencilizer/variable/`

**Why a stage:**

- Q1. It needs `cff2-static`'s CFF2 read path merged.
- Q3. The replay must be proven against real fixtures, at every validation location, before writers serialize it; a writer bug would otherwise hide a replay bug.
- Q4. Engine plus writers is several algorithmic modules and would pass 500,000 tokens of orchestrator context.

New package `src/stencilizer/variable/`:

- `model.py`: `Support`, `VariableGlyph`.
- `solver.py`: `solve_deltas`.
- `reader.py`: `read_variable_glyph`, `is_variable`, `cff2_vsindex`.
- `flatten.py`: one subdivision schedule shared by all masters (wave 1, so the overlap worker tests against it).
- `overlaps.py`: compatible overlap removal with skia-pathops.
- `replay.py`: the output-to-input map, bridge-line grouping, per-master realignment and wrong-side projection.
- `rounding.py`: `round_variable_glyph`, the integer rounding each target format applies (gvar: default and deltas rounded against each other; CFF2: default rounded, deltas kept as floats). The engine validates the rounded glyph and the writers store it, so what passes validation is what the font contains.
- `validate.py`: validation locations (capped at 64 per glyph) and checks.
- `transform.py`: `transform_variable_glyph` and the picklable worker `process_variable_glyph`.

The contract session builds the fixtures under `tests/fixtures/variable/` (harness). Every `VariationDataError` the engine raises is per glyph, and `transform_variable_glyph` turns it into the no-op outcome, which counts the glyph's default-master islands as unbridged (the islands of the overlap-merged default, else of the flattened default, else of the raw default).

Partial success: the engine keeps the static rule (Goals). A deterministic partial case, built by wrapping the static surgery so it leaves one of two holes unbridged, is part of W2's tests, so the rule is exercised even though no fixture glyph is partial.

Risk walk:

| Area | Contract |
| --- | --- |
| External data | `cff2-variable-reader-uses-varstore-regions`, `solver-reproduces-original-deltas`, `cff2-vsindex-selection` |
| Untrusted input (degenerate regions) | `singular-supports-raise` |
| Serialization fidelity | `rounded-glyph-is-validated` |
| Configuration propagation | `bridge-width-config-reaches-engine` |
| Lifecycle (process-pool IPC) | `variable-glyph-dict-roundtrip` |
| Core behaviour | `realigned-replay-valid-everywhere`, `masters-share-structure`, `overlap-built-counter-bridged`, `outcome-valid-or-untouched`, `validation-locations-cover-corners` |

### 3. variable-writers: gvar and CFF2 blend writers, variable name records

**Why a stage:** Q1. It needs `variable-engine`'s `VariableGlyph` merged, and through it `cff2-static`'s CFF2 handling.

Work:

- `variable/write_gvar.py`: glyf plus gvar, storing the values of `round_variable_glyph(vg)` (engine; idempotent, so a glyph the engine already rounded is unchanged): deltas solved against the original tuple supports and rounded against each other, phantom deltas copied, hmtx lsb moved with xMin, and IUP-optimized at tolerance 0 so the points of one bridge line keep equal deltas. A font with fvar and no gvar (legal; the GUI fixture `variable_font_path` is one) gets glyf and hmtx only and no new gvar table.
- `variable/write_cff2.py`: CFF2 charstrings with blend arguments per the glyph's VarData regions: rounded absolute default, float deltas, `[default, d…, 1]` blend arguments, `preserveTopology=True`. A glyph that never blends (`supports=()`) gets a plain charstring with no blend and no vsindex.
- `io/writer.py`:
  - `FontWriter.update_variable_glyph`, called only for bridged glyphs;
  - move the fvar rejection from `save` into `update_glyph`, so the static path can never write a variable glyph;
  - suffix nameID 25 and the fvar instance PostScript name records once each, and give variable fonts' nameID 4 the suffix after the family name.

Risk walk:

| Area | Contract |
| --- | --- |
| External data (gvar and CFF2 binary structure decoded by fontTools after save) | `gvar-output-reproduces-instances`, `cff2-blend-output-reproduces-instances` |
| Metrics | `gvar-keeps-phantom-deltas` |
| Configuration (fvar without gvar) | `gvar-writer-handles-missing-gvar` |
| Names | `variable-names-suffixed` |
| Wrong path (static writer on a variable font) | `static-writer-refuses-variable` |

### 4. variable-surfaces: processor, CLI and GUI wiring

**Why a stage:** Q1. It needs the writers merged. It owns every pre-existing integration seam: `core/processor.py`, `io/reader.py` fvar rejection, `cli/app.py`, `cli/output.py`, and the GUI session, controller, controls and main window.

Work:

- Variable fonts dispatch to a new `variable/processing.py` from `FontProcessor.classify_glyphs` and `FontProcessor._process_loaded_font`, through function-level imports (`core/__init__.py` imports the processor eagerly, so a module-level import is a cycle). The static pool, result collection and transactional save are generalized with keyword-only parameters and reused, so the spawn-context patch, cancellation and progress reporting cover variable fonts too.
- `process_variable_font` fills the `ProcessingStats` that `FontProcessor.process` created and passed down (`_process_loaded_font` returns `None` and fills `stats` in place; `process` stamps the times around it and returns that same object), so timing, counts and progress need no adoption step.
- Classification never aborts on one glyph's variation data. A `VariationDataError` while reading a glyph skips it as "unsupported variation data" and records the island count of its static default outline in a new `GlyphClassification.unsupported_islands`, which `process_variable_font` adds to `unbridged_count` and the GUI survey reports as unbridged. Failures after reading (flattening, overlap removal, replay, validation) reach the worker and come back as the no-op outcome with the default islands counted.
- `--instance` pins a static instance through `io/instance.py`, removing overlaps and downgrading CFF2 to CFF; STAT-derived naming is attempted and, when no STAT AxisValue names the requested coordinate (Inter wght=650), the instance keeps the variable font's names and the writer adds the suffix. `--list-islands` and `--dry-run` see the same overlap-built counters as processing, and the font info line lists the variable axes.
- The GUI opens and saves variable fonts and previews at an axis location set by one slider per fvar axis, normalized through fvar and avar. Composite glyphs preview at the default location; while one is selected the sliders are disabled and the AXES card shows a note saying so. The preview cache is owned by the session, keyed by every transform input, and bounded.

Risk walk:

| Area | Contract |
| --- | --- |
| Reachability | `cli-writes-variable-stencil`, `axis-sliders-drive-preview`, `gui-saves-variable-font` |
| Configuration propagation | `cli-instance-pins-static`, `cli-shows-variable-axes` |
| Untrusted input | `cli-instance-rejects-bad-spec` |
| Preserve no-op glyphs, serialized validity, stats | `untouched-glyph-keeps-variations`, `variable-output-valid-everywhere`, `unsupported-glyph-counted` |
| GUI behaviour | `session-previews-at-location`, `composite-preview-at-default` |

### Integration Verification

Covers:

- the full suite under pytest-xdist, lint, format and mypy, plus one suite run under Python 3.11, the other interpreter in CI's test matrix;
- parallel `loom-code-reviewer` subagents (correctness, font-format, architecture and size limits);
- end-to-end CLI runs on the three variable fixtures plus a static CFF2 conversion;
- re-running each stage's `reachable` checks.

### Knowledge Distillation

Curates stage memories:

- replay design, realignment rules and validation outcomes into `patterns/variable-replay.md`;
- format support into `architecture.md` and `stack.md`;
- removes the "CFF2 write and variable fonts are unsupported" entry in `concerns.md`;
- updates README "Font Format Support", "Graphical Interface" and "Future Work".

## Sandbox and environment inventory

| Need | Resolution |
| --- | --- |
| Python dependencies per worktree | Provisioned: `uv sync --frozen --no-install-project` in `.` (writes the git-ignored `.venv/`) |
| `uv add skia-pathops` (variable-engine) and `uv run` re-syncing after lock changes | `pypi.org` and `files.pythonhosted.org` allowed; the uv cache is pre-granted |
| Python 3.11 suite run (integration-verify) | CPython 3.11.10 already installed by uv under `~/.local/share/uv/python` (readable, nothing to download); the environment is created at `.venv/py311` inside the git-ignored `.venv/`; wheels come from the pre-granted uv cache or `pypi.org` |
| System fonts read by the fixture build (variable-engine contract session) | Readable (not denied): `/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf`, `~/.local/share/fonts/InterVariable.ttf`, `/usr/share/fonts/opentype/cantarell/Cantarell-VF.otf`. The committed subsets are what tests read |
| Qt GUI tests | Every acceptance command sets `QT_QPA_PLATFORM=offscreen` explicitly. `tests/gui/conftest.py:24` only setdefaults it, and the host desktop session exports `QT_QPA_PLATFORM=wayland;xcb`, which opened real windows on the user's screen during planning; `cff2-static` changes conftest to assign it. No display or socket is needed |
| Contract freeze and stage completion | Sandbox dotfile mounts can make `loom stage contracts freeze <id>` refuse and shift the review fingerprint inside the sandbox (knowledge `mistakes/review-and-completion-gates.md`). When an in-sandbox freeze or `loom stage complete <id>` refuses for that reason, the operator runs it from the worktree root outside the sandbox: four contract stages, so up to eight manual commands |
| Plan path | `loom run` renames this file to `IN_PROGRESS-PLAN-variable-fonts.md`, then `DONE-…`; stage descriptions refer to `doc/plans/*PLAN-variable-fonts.md` |
| Process pools in GUI save tests | Spawn context via the autouse `spawn_process_pool` fixture |
| Credentials, host daemons, git hooks | None (no pre-commit hook fetches) |

Implementation lanes: codex CLI and plugin are installed. `cff2-static`, `variable-writers` and `variable-surfaces` list `["claude", "codex"]`, and `variable-engine` lists `["claude"]`, as confirmed by the user.

---

<!-- loom METADATA -->

```yaml
loom:
  version: 2
  auto_merge: null
  sandbox:
    enabled: true
    auto_allow: true
    allow_unsandboxed_escape: false
    excluded_commands: []
    filesystem:
      deny_read:
      - ~/.ssh/**
      - ~/.aws/**
      - ~/.config/gcloud/**
      - ~/.gnupg/**
      deny_write: []
      allow_write: []
    network:
      allowed_domains:
      - pypi.org
      - files.pythonhosted.org
      additional_domains: []
      allow_local_binding: false
      allow_unix_sockets: []
      allow_all_unix_sockets: false
    linux:
      enable_weaker_nested: false
    command_confinement: confined
  ratchet_files:
  - doc/loom/knowledge/check-baseline.txt
  - tests/regression/golden/commitmono.json.gz
  - tests/regression/golden/commitmono_pipeline.json.gz
  - tests/regression/golden/lato.json.gz
  - tests/regression/golden/roboto.json.gz
  provision:
  - working_dir: .
    command: uv sync --frozen --no-install-project
  stages:
  - id: knowledge-bootstrap
    name: Knowledge audit
    description: |
      The knowledge base already describes this codebase (loom knowledge check is clean at
      4254909). This stage audits the sections the plan touches and records planning facts.
      Model override to sonnet: a short audit needs no opus.
      Use parallel subagents and skills to maximize performance. SINGLE-AGENT here: the
      audit covers four sections, so do not spawn subagents.
      1. Run loom knowledge sync from the repository root.
      2. Audit against the tree, correcting with loom knowledge replace-section where wrong:
         architecture.md "Font format I/O" and "Font format and error boundary"
         (src/stencilizer/io/converter.py, io/reader.py, io/writer.py, gui/session.py),
         concerns.md first entry "Unsupported font formats" (CFF2 write and variable fonts
         unsupported), stack.md "Supported font formats". The pressure test rewrote the first
         three to the tree (FontReader.load rejects fvar/CFF2 at io/reader.py:54-57,
         FontWriter._check_supported_format at io/writer.py:25-29, gui/session.py:63-66);
         confirm the base has those versions. If a section still says "nothing in the core
         rejects them", correct it with replace-section. This plan's distill stage rewrites
         them once support lands. concerns.md "Unchecked classification reuse" (renamed from
         "Swallowed glyph-write failures and ...") must say _save_font raises FontSaveError on
         a failed update_glyph (core/processor.py:349-355); correct it if the base still has
         the old heading.
      3. Record in concerns.md, under a heading "Variable fonts: overlap-built counters"
         (current truth, no history): many variable fonts keep overlapping contours, so
         counters are formed by overlap rather than by a separate contour (Inter A D P R e
         4 &, Monaspace Neon A B D P R a b d e g p q 4 6 8 9 & @ at the default master),
         and GlyphAnalyzer reports no island for them; and island sets were identical at
         every axis extreme for Inter, Ubuntu and Monaspace Neon. Source: the plan's
         Evidence section (doc/plans/*PLAN-variable-fonts.md).
      4. Confirm mistakes.md has "Fixed-parameter replay breaks bridge-cut coincidence
         across masters"; do not duplicate it. It reaches the base only through the plan's
         precondition commit; if it is missing, add it with loom knowledge update mistakes
         "<entry>" from the plan's Evidence section ("Fixed-t replay fails", "Per-master
         realignment works"), with What happened / Why / Prevention / Fix.
      5. Confirm mistakes.md has "GUI tests open real windows when QT_QPA_PLATFORM is set"
         (written by the pressure test; the precondition commit carries it); add it the same
         way if missing: tests/gui/conftest.py:24 uses os.environ.setdefault, and a desktop
         session exporting QT_QPA_PLATFORM=wayland;xcb makes every GUI test show windows.
      Use the loom knowledge CLI, never Write/Edit. NEVER Claude Code auto-memory.
    summary: Checks the knowledge sections this plan will change and records the variable-font facts found while planning.
    dependencies: []
    parallel_group: null
    acceptance:
    - loom knowledge check --strict --baseline doc/loom/knowledge/check-baseline.txt
    - 'rg -qF "Variable fonts: overlap-built counters" doc/loom/knowledge/concerns.md'
    - rg -qF "Fixed-parameter replay breaks bridge-cut coincidence across masters" doc/loom/knowledge/mistakes.md
    setup: []
    files:
    - doc/loom/knowledge/**
    auto_merge: null
    working_dir: .
    stage_type: knowledge
    artifacts:
    - doc/loom/knowledge/concerns.md
    wiring: []
    context_ceiling_tokens: null
    sandbox: {}
    model: sonnet
    ultracode: false
    implementers:
    - claude
  - id: cff2-static
    name: Static CFF2 read and write
    description: |
      Add static CFF2 support. Plan: doc/plans/*PLAN-variable-fonts.md (Evidence section).
      Use parallel subagents and skills to maximize performance.
      Territories below are DISJOINT. Workers NEVER spawn subagents. Spawn every
      worker BY AGENT TYPE, ALL in ONE message, each with the fixed prompt plus
      "Your brief: <path>. Read it in full before anything else."
      W2 is a codex unit: spawn loom-codex-forwarder in the FOREGROUND with
      --model gpt-5.6-terra --effort xhigh and an explicit Bash timeout of 600000 ms;
      tell it not to run git; check git status --short after it returns, then run
      its test file yourself. If codex is unavailable, give W2 to loom-software-engineer.

      | Worker | Role | Tier | Files owned | Shared context | Brief path |
      | ------ | ---- | ---- | ----------- | -------------- | ---------- |
      | W1 | CFF2 I/O, rejection tests, xdist, offscreen conftest | sonnet | src/stencilizer/io/converter.py, src/stencilizer/io/reader.py, src/stencilizer/io/writer.py, src/stencilizer/gui/session.py, pyproject.toml, uv.lock, tests/unit/test_io.py, tests/unit/test_review_io.py, tests/gui/conftest.py, tests/gui/test_session.py, tests/gui/test_main_window.py, tests/gui/test_controller_errors.py | none | doc/plans/briefs/variable-fonts/cff2-static/w1-cff2-io.md |
      | W2 | CFF2 integration tests | codex terra | tests/integration/test_cff2_static.py | src/stencilizer/io/converter.py (read-only) | doc/plans/briefs/variable-fonts/cff2-static/w2-cff2-integration.md |

      Public surface the contracts call (write the contracts from this):
        FontReader(path).load(); FontReader.get_glyph(name) -> Glyph | None (io/reader.py).
        FontProcessor(StencilizerSettings()).process(font_path=Path, output_path=Path)
        (core/processor.py) writes a font; a CFF2 input must produce a CFF2 output.
        FontSession.open(path, processor) (gui/session.py) must succeed for a static CFF2 font.
        A static CFF2 font is made at test time exactly as tests/gui/conftest.py fixture
        cff2_font_path does: TTFont(CommitMono-Cosmix-700-Regular.otf), then
        fontTools.cffLib.CFFToCFF2.convertCFFToCFF2(font), then save to tmp_path.
      A font with an fvar table stays rejected by FontReader.load and FontWriter
      (FontFormatError) in this stage; variable-surfaces lifts that.
      Static TrueType and CFF output must not change: tests/regression pins it.
      HEAD baseline: tests/unit/test_io.py::TestCffGlyphUpdate::test_update_cff_glyph_passes_private_and_global_subrs
      is red at 4254909 (expects no optimize=False); W1 repairs it (a no-op if the base
      already expects optimize=False: an uncommitted fix existed in the main checkout while
      planning). W1 also adds pytest-xdist as a dev dependency (uv add --dev pytest-xdist):
      the full suite takes about 600 s serially and 69 s with --numprocesses=16, and
      every later gate needs it under the 300 s cap.
      WIRING: the converter's new dispatch line must read exactly
      `_update_cff2_glyph(glyph, original_glyph, font)` (the wiring regex matches the call,
      so the def line alone cannot satisfy it).
      INTEGRITY: rewriting the CFF2 rejection tests raises TI-edit events; the plan's
      "Test integrity in this plan" table lists the expected ones. Each rewritten test keeps
      its declaration and gains a positive assertion; never delete a test or assertion.
      Dispute the listed TI-edit events together, once; restore anything behind a TI-assert
      or TI-decl event.
      CODEX: W2 cannot run uv run (no network, read-only uv cache); its brief gives a static
      .venv/bin check. Run its tests yourself after W1 returns.
      CONTRACT SESSION: the frozen contract files must pass the stage gate. Before freezing,
      run uv run ruff format, uv run ruff check and uv run mypy on them. The only mypy errors
      allowed are import errors for this stage's unwritten symbols (_update_cff2_glyph):
      import them inside the test functions with no type: ignore. fontTools imports carry
      # type: ignore[import-untyped] (tests/gui/conftest.py:12-15); quote cast() types,
      prefix unused stub parameters with _. GUI contracts run offscreen.
      MEMORY: record mistakes/decisions/surprises via loom memory immediately;
      NEVER loom knowledge (implementation stage); NEVER Claude Code auto-memory.
    summary: Static CFF2 fonts load with correct winding and save with bridged glyphs; the stale CFF writer test is repaired. Variable fonts stay rejected until the surfaces stage.
    dependencies:
    - knowledge-bootstrap
    parallel_group: null
    acceptance:
    - uv run python -c "import xdist"
    - env QT_QPA_PLATFORM=offscreen uv run pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - uv run ruff check src tests packaging
    - uv run ruff format --check src tests packaging
    - uv run mypy src/stencilizer tests packaging
    - uv lock --check
    setup: []
    files:
    - src/stencilizer/io/**
    - src/stencilizer/gui/session.py
    - pyproject.toml
    - uv.lock
    - tests/unit/test_io.py
    - tests/unit/test_review_io.py
    - tests/unit/test_cff2_contracts.py
    - tests/integration/test_cff2_static.py
    - tests/gui/conftest.py
    - tests/gui/test_session.py
    - tests/gui/test_main_window.py
    - tests/gui/test_controller_errors.py
    - tests/gui/test_cff2_session_contracts.py
    auto_merge: null
    working_dir: .
    stage_type: standard
    artifacts:
    - src/stencilizer/io/converter.py
    - src/stencilizer/io/reader.py
    - src/stencilizer/io/writer.py
    - src/stencilizer/gui/session.py
    - tests/gui/conftest.py
    - tests/integration/test_cff2_static.py
    wiring:
    - source: tests/gui/conftest.py
      pattern: os\.environ\["QT_QPA_PLATFORM"\] = "offscreen"
      description: GUI tests force offscreen instead of setdefault (the host exports wayland;xcb)
    - source: src/stencilizer/io/converter.py
      pattern: _update_cff2_glyph\(glyph, original_glyph, font\)
      description: CFF2 write path dispatched from domain_glyph_to_fonttools (the call, not the def line)
    before_stage:
    - command: rg -c "_update_cff2_glyph" src/stencilizer/io/converter.py
      exit_code: 1
      description: No CFF2 write path at the base commit
    after_stage:
    - command: uv run python -c "from stencilizer.io.converter import _update_cff2_glyph"
      exit_code: 0
      description: CFF2 write path importable
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
    - codex
    skills:
    - loom-python
    contracts:
    - id: cff2-read-normalizes-winding
      file: tests/unit/test_cff2_contracts.py
      test: test_cff2_read_normalizes_winding
      scenario: converts CommitMono to CFF2 with convertCFFToCFF2 into tmp_path, loads it with FontReader, reads glyph 'O' and runs GlyphAnalyzer().analyze on it
      rejects: a reader that reverses contours only when 'CFF ' is present, so CFF2 'O' reads inside-out and the analyzer reports no island
    - id: cff2-static-roundtrip-writes-bridges
      file: tests/unit/test_cff2_contracts.py
      test: test_cff2_static_roundtrip_writes_bridges
      scenario: runs FontProcessor(StencilizerSettings()).process on the converted CFF2 CommitMono into tmp_path, reopens the output with TTFont and FontReader, and inspects glyph 'O'
      rejects: a writer that raises NotImplementedError for CFF2, or saves the CFF2 font with the original 'O' charstring (output still has an island, or no CFF2 table)
    - id: gui-session-opens-cff2
      file: tests/gui/test_cff2_session_contracts.py
      test: test_gui_session_opens_cff2
      scenario: calls FontSession.open on the cff2_font_path fixture with the processor fixture and checks that 'O' is among session.island_glyphs
      rejects: a session that keeps the 'CFF2 outlines are not supported' rejection in unsupported_reason
  - id: variable-engine
    name: Variable replay engine
    description: |
      Build src/stencilizer/variable/ (new package). Plan: doc/plans/*PLAN-variable-fonts.md,
      Evidence section; reference spike code in doc/plans/briefs/variable-fonts/spike/
      (vlib.py mapping, vlib3.py replay3, vlib4.py lines, ovl.py union/umap/ureplay).
      The spike is evidence, not production code: rewrite to the repo's typing and size rules.
      Use parallel subagents and skills to maximize performance.
      Territories below are DISJOINT. Workers NEVER spawn subagents. Spawn every
      worker BY AGENT TYPE, each with the fixed prompt plus
      "Your brief: <path>. Read it in full before anything else."
      WAVE 1 (foundation, compile-order dependency): spawn W1 alone and wait.
      WAVE 2: spawn W2 and W3 in ONE message after W1 returns.
      W2 is core algorithmic work: loom-senior-software-engineer with effort xhigh.

      | Worker | Role | Tier | Files owned | Shared context | Brief path |
      | ------ | ---- | ---- | ----------- | -------------- | ---------- |
      | W1 | Model, solver, reader, flattening, rounding, dependency | sonnet | src/stencilizer/variable/__init__.py, src/stencilizer/variable/model.py, src/stencilizer/variable/solver.py, src/stencilizer/variable/reader.py, src/stencilizer/variable/flatten.py, src/stencilizer/variable/rounding.py, src/stencilizer/exceptions.py, pyproject.toml, uv.lock, tests/unit/test_variable_model.py | tests/fixtures/variable (read-only) | doc/plans/briefs/variable-fonts/variable-engine/w1-foundation.md |
      | W2 | Replay, validate, transform | opus/xhigh | src/stencilizer/variable/replay.py, src/stencilizer/variable/validate.py, src/stencilizer/variable/transform.py, tests/unit/test_variable_replay.py | src/stencilizer/core (read-only) | doc/plans/briefs/variable-fonts/variable-engine/w2-replay.md |
      | W3 | Compatible overlap removal | opus | src/stencilizer/variable/overlaps.py, tests/unit/test_variable_overlaps.py | src/stencilizer/variable/model.py, flatten.py (read-only) | doc/plans/briefs/variable-fonts/variable-engine/w3-overlaps.md |

      Flattening is wave 1 because W3's vertex matching must be tested against the real
      flattener: pathops round-trips coordinates through float32, and the spike's
      "1e-4 after rounding" match only worked at 8 subdivisions (exact in float32); at 7 or 13
      it left every Inter probe glyph unmappable. After wave 2, run
      tests/unit/test_variable_engine_contracts.py yourself before the verifier.
      WINDING: outer contours are clockwise (negative signed area, core/analyzer.py:131); the
      root CLAUDE.md line "TrueType: CCW=outer" is wrong.

      Public surface (pinned; contracts are written from it):
        stencilizer.variable.model:
          @dataclass(frozen=True) Support(axes: tuple[tuple[str, float, float, float], ...])
            with peak() -> dict[str, float] and scalar(location: Mapping[str, float]) -> float
            (delegates to fontTools.varLib.models.supportScalar with OpenType rules; axes
            absent from location count as 0.0; region (-1, 0.5, 1) at {} gives 1.0).
          @dataclass VariableGlyph(default: Glyph, supports: tuple[Support, ...],
            masters: tuple[Glyph, ...], axis_tags: tuple[str, ...], cff2: bool = False)
            where masters[i] is the full outline at supports[i].peak() and cff2 records the
            outline format (set by read_variable_glyph, carried by to_dict/from_dict, used
            by rounding); property name -> str (default.name); methods
            to_dict() -> dict[str, Any], from_dict(data) -> VariableGlyph (classmethod),
            instance(location) -> Glyph (default + sum of scalar * solved delta), and
            deltas() -> list[list[tuple[float, float]]] (one list per support, one (dx, dy)
            per default domain point, from solve_deltas, computed once and cached outside
            eq/repr/to_dict). __post_init__ raises VariationDataError on a master whose
            contour count, point counts or point types differ from the default.
        stencilizer.variable.solver.solve_deltas(supports, default, masters, *,
          glyph_name="<unknown>") -> list[list[tuple[float, float]]]; default is a sequence
          of (x, y), masters one sequence of (x, y) per support; raises
          stencilizer.exceptions.VariationDataError (new, subclass of GlyphError,
          __init__(glyph_name: str, reason: str)) when the support matrix is singular (two
          supports sharing a peak are legal OpenType and singular here).
        stencilizer.variable.reader: is_variable(font: TTFont) -> bool ("fvar" in font);
          read_variable_glyph(font: TTFont, name: str) -> VariableGlyph | None (None for
          empty or composite glyphs); supports come from the glyph's own gvar tuples
          (TrueType; none when the font has no gvar) or from the VarData regions its
          charstring selects (CFF2); cff2_vsindex(font, name) -> int | None returns the
          vsindex fontTools blends with, recorded through charstring.draw(NullPen(), blender)
          so subrs are followed, None when the glyph never blends (then supports=()), and
          raises VariationDataError when the FD Private.vsindex is set and differs (fontTools
          glyph sets ignore it) or when the blender records more than one index for the
          glyph (the CFF2 spec allows one vsindex per charstring; never substitute VarData
          0); outlines come from font.getGlyphSet(location=peak,
          normalized=True) drawn through stencilizer.io.converter.fonttools_glyph_to_domain,
          which already reverses CFF2 winding after cff2-static (do not reverse again).
        stencilizer.variable.flatten.flatten_compatible(vg: VariableGlyph, upm: int) ->
          VariableGlyph: all ON_CURVE, identical structure in every master, one subdivision
          count per segment within core.curve.curve_tolerance(upm) in every master, at most
          64 (beyond that VariationDataError).
        stencilizer.variable.overlaps.remove_overlaps_compatible(vg: VariableGlyph) ->
          VariableGlyph | None.
        stencilizer.variable.rounding.round_variable_glyph(vg: VariableGlyph) ->
          VariableGlyph: the values the target format stores, as a VariableGlyph whose
          masters are rebuilt from them (so deltas() returns the stored deltas).
          gvar (cff2=False): default coordinates otRound-ed; deltas rounded against each
          other: supports visited by (axis count, then largest |peak|),
          RD_k = otRound(vg.instance(peak_k) - (rounded default + sum over visited j of
          scalar_j(peak_k) * RD_j)), falling back to otRound(delta_k) when an unvisited
          support is non-zero at peak_k; masters[k] = rounded default + sum over all j of
          scalar_j(peak_k) * RD_j. CFF2 (cff2=True): default otRound-ed, deltas unchanged
          (CFF2 stores them as 16.16 fixed). Idempotent: rounding a rounded glyph returns
          equal coordinates within 1e-9. Points with equal default coordinates and equal
          deltas get equal rounded values, so bridge lines stay collinear.
        stencilizer.variable.validate.validation_locations(vg: VariableGlyph) ->
          list[dict[str, float]]: normalized coordinates; the default {}, every support peak,
          and a grid built from per-axis value sets {0.0} ∪ {-1.0 if any support peak on
          that axis is negative} ∪ {+1.0 if any is positive} ∪ {each intermediate peak
          coordinate on that axis}: the full product when it has at most 64 locations,
          otherwise each axis's non-zero values alone plus all-minimum and all-maximum.
          Ubuntu's wdth default equals its maximum, so wdth has no +1 there.
      CONTRACT SESSION FIXTURES (harness tests/fixtures/variable/**): write
      tests/fixtures/variable/build_fixtures.py and run it once. It subsets three system
      fonts with fontTools.subset to the characters "ADPRe4&BOabdgopq0689lxÁ" (Á keeps a
      composite glyph, Aacute, in the Ubuntu and Inter subsets for the GUI composite
      contract; Cantarell's Aacute is a plain outline), with Options
      layout_features=[], name_IDs=["*"], name_languages=["*"], notdef_outline=True,
      glyph_names=True:
        /usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf -> Ubuntu-VF-subset.ttf
        ~/.local/share/fonts/InterVariable.ttf -> Inter-VF-subset.ttf
        /usr/share/fonts/opentype/cantarell/Cantarell-VF.otf -> Cantarell-VF-subset.otf
      It also writes tests/fixtures/variable/README.md naming each source path and its
      licence (Ubuntu Font Licence 1.0; SIL OFL 1.1 for Inter and Cantarell). A planning trial
      of the subset without Á produced 14,800 / 16,276 / 7,852 bytes and 23 glyphs each, with
      fvar, gvar (or CFF2), avar, HVAR and STAT kept (MVAR too in Inter and Cantarell); Á
      adds Aacute and its accent glyphs. Contracts name glyphs by character
      and resolve them with font.getBestCmap()[ord(char)] ('8' for eight).
        stencilizer.variable.transform:
          @dataclass(frozen=True) VariableOutcome(glyph: VariableGlyph, bridge_count: int,
            unbridged_count: int)
          transform_variable_glyph(vg: VariableGlyph, bridge: BridgeConfig,
            geometry: GeometryConfig, upm: int) -> VariableOutcome
          process_variable_glyph(vg_dict, config_dict, upm, geometry_dict) -> dict with the
            same keys as core.processor.process_glyph ("glyph" holds VariableGlyph.to_dict()).
      OUTCOME RULE (partial success as in the static pipeline, see the plan's Goals): the
      replayed glyph is passed through round_variable_glyph and the ROUNDED glyph is
      validated and returned, so the writers store exactly what was validated. A glyph whose
      replay cannot be mapped, or whose rounded result fails validation at any validation
      location (more islands than the default surgery left unbridged, i.e. more than
      outcome.unbridged_count, or a bridge line pair inverts order), or for which any step
      raises VariationDataError, returns VariableOutcome(original input glyph, 0, n), where
      n is the island count of the overlap-merged default when overlap removal succeeded,
      else of the flattened default, else (flattening raised) of the raw input default;
      never 0 for a glyph that has islands. A partially bridged glyph that passes returns
      VariableOutcome(rounded result, bridge_count, unbridged_count) with
      unbridged_count > 0. transform_variable_glyph never raises VariationDataError.
      Never modify src/stencilizer/core/**: the static pipeline is pinned by tests/regression.
      Size limits (tests/regression/test_code_structure.py): files <= 400 lines,
      functions <= 50 effective lines, classes <= 300.
      CONTRACT SESSION: the frozen contract file and the harness (build_fixtures.py) must
      pass the stage gate, and nobody can edit them after the freeze. Before freezing, run
      uv run ruff format, uv run ruff check and uv run mypy on them. The only mypy errors
      allowed are import errors for this stage's unwritten stencilizer.variable modules:
      import those inside the test functions with no type: ignore (an ignore becomes an
      unused-ignore error under strict once the module exists). fontTools imports carry
      # type: ignore[import-untyped] (tests/gui/conftest.py:12-15); quote cast() types,
      prefix unused stub parameters with _. Compare outlines through
      stencilizer.io.converter.fonttools_glyph_to_domain point by point, never raw
      RecordingPen values (domain contours differ from glyf: rotation, closing duplicate).
      The contract session reads ~/.local/share/fonts/InterVariable.ttf and the two
      /usr/share/fonts paths in the Sandbox inventory; the committed subsets are what
      tests read.
      MEMORY: record mistakes/decisions/surprises via loom memory immediately;
      NEVER loom knowledge (implementation stage); NEVER Claude Code auto-memory.
    summary: 'Adds the variable-font engine: reads per-glyph variation regions, removes overlaps compatibly, runs bridge surgery on the default master and replays it in every master, then validates every location.'
    dependencies:
    - cff2-static
    parallel_group: null
    acceptance:
    - uv run python -c "import pathops"
    - env QT_QPA_PLATFORM=offscreen uv run pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - uv run ruff check src tests packaging
    - uv run ruff format --check src tests packaging
    - uv run mypy src/stencilizer tests packaging
    - uv lock --check
    setup: []
    files:
    - src/stencilizer/variable/**
    - src/stencilizer/exceptions.py
    - pyproject.toml
    - uv.lock
    - tests/unit/test_variable_*.py
    - tests/unit/test_variable_engine_contracts.py
    - tests/fixtures/variable/**
    auto_merge: null
    working_dir: .
    stage_type: standard
    artifacts:
    - src/stencilizer/variable/__init__.py
    - src/stencilizer/variable/model.py
    - src/stencilizer/variable/solver.py
    - src/stencilizer/variable/reader.py
    - src/stencilizer/variable/flatten.py
    - src/stencilizer/variable/rounding.py
    - src/stencilizer/variable/overlaps.py
    - src/stencilizer/variable/replay.py
    - src/stencilizer/variable/validate.py
    - src/stencilizer/variable/transform.py
    - tests/fixtures/variable/build_fixtures.py
    wiring:
    - source: src/stencilizer/exceptions.py
      pattern: class VariationDataError\(GlyphError\)
      description: per-glyph variation error exists (raised by solver, model, reader, flatten; caught in transform)
    - source: src/stencilizer/variable/transform.py
      pattern: remove_overlaps_compatible\(
      description: transform runs compatible overlap removal before surgery
    - source: src/stencilizer/variable/transform.py
      pattern: round_variable_glyph\(
      description: transform validates the rounded glyph the writers will store
    - source: src/stencilizer/variable/transform.py
      pattern: transform_with_outcome\(
      description: transform reuses the static GlyphTransformer on the default master
    before_stage:
    - command: test -d src/stencilizer/variable
      exit_code: 1
      description: No variable package at the base commit
    after_stage:
    - command: uv run python -c "import stencilizer.variable.transform, stencilizer.variable.overlaps"
      exit_code: 0
      description: Engine modules import
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
    subagent_timeout_secs: 1200
    skills:
    - loom-python
    contracts:
    - id: solver-reproduces-original-deltas
      file: tests/unit/test_variable_engine_contracts.py
      test: test_solver_reproduces_original_deltas
      scenario: 'reads glyph ''o'' from tests/fixtures/variable/Ubuntu-VF-subset.ttf with read_variable_glyph, calls vg.deltas(), and compares with the font''s gvar TupleVariations for ''o'' after calcInferredDeltas (coords, endPts from font[''glyf'']._getCoordinatesAndControls(''o'', font[''hmtx''].metrics)), excluding the 4 phantom points. Domain points are not glyf points (34 against 32 for ''o''): map domain point p of contour k to glyf index start_k + (first_on_k + p) mod L_k, where first_on_k is the index of the contour''s first on-curve glyf point and L_k its glyf point count; skip p == L_k (the closing duplicate the converter appends after a final curve); for a contour with no on-curve point use domain points [1:]. Compare unrounded with abs tolerance 1e-6 (planning measured 2.8e-14)'
      rejects: a solver that takes each delta as master minus default independently, ignoring overlap between the wdth, wght and wdth+wght corner regions
    - id: singular-supports-raise
      file: tests/unit/test_variable_engine_contracts.py
      test: test_singular_supports_raise
      scenario: calls solve_deltas with two identical Support(axes=(('wght', 0.0, 1.0, 1.0),)) entries, one default point and two different master points
      rejects: a solver that divides by a zero pivot or returns NaN/inf deltas instead of raising VariationDataError
    - id: cff2-variable-reader-uses-varstore-regions
      file: tests/unit/test_variable_engine_contracts.py
      test: test_cff2_variable_reader_uses_varstore_regions
      scenario: reads glyph 'o' from tests/fixtures/variable/Cantarell-VF-subset.otf with read_variable_glyph; checks len(vg.supports) == 2, and for each support peak that vg.instance(peak) equals fonttools_glyph_to_domain('o', font.getGlyphSet(location=peak, normalized=True)['o'], font) point by point within 0.5 units, and that GlyphAnalyzer().analyze(vg.default, upm).get_islands() has one entry
      rejects: a reader that only looks at gvar (CFF2 glyph gets zero supports) or skips the CFF2 winding reversal
    - id: masters-share-structure
      file: tests/unit/test_variable_engine_contracts.py
      test: test_masters_share_structure
      scenario: runs transform_variable_glyph on Ubuntu-VF-subset 'o' with BridgeConfig() and GeometryConfig(); asserts bridge_count >= 1 and that the result's default differs from the input default (a no-op engine passes a structure check because the input masters already share structure); compares contour count, per-contour point count and point types of every master with the default
      rejects: an engine that runs GlyphTransformer independently on each master, producing different point counts, or one that returns the input unchanged
    - id: realigned-replay-valid-everywhere
      file: tests/unit/test_variable_engine_contracts.py
      test: test_realigned_replay_valid_everywhere
      scenario: 'runs transform_variable_glyph on Ubuntu-VF-subset ''o'' and ''8''; asserts bridge_count >= 1 and that GlyphAnalyzer().analyze(result.glyph.instance(loc), upm).get_islands() is empty for every loc in validation_locations(result.glyph) and for {''wght'': -0.5, ''wdth'': -0.5}'
      rejects: 'a replay that keeps every cut point at its default edge parameter t (fixed-t), which leaves islands at wght=-1 (spike: 554 of 561 Ubuntu glyphs)'
    - id: overlap-built-counter-bridged
      file: tests/unit/test_variable_engine_contracts.py
      test: test_overlap_built_counter_bridged
      scenario: runs transform_variable_glyph on tests/fixtures/variable/Inter-VF-subset.ttf glyph 'P' (one self-overlapping contour at the default master); asserts bridge_count >= 1 and that GlyphAnalyzer().analyze(result.glyph.instance(loc), upm).get_islands() is empty for every loc in validation_locations(result.glyph)
      rejects: an engine without compatible overlap removal, which sees no island in Inter 'P' and returns bridge_count 0, or one whose overlap union maps vertices at a float32-fragile tolerance and returns None for 'P'
    - id: outcome-valid-or-untouched
      file: tests/unit/test_variable_engine_contracts.py
      test: test_outcome_valid_or_untouched
      scenario: for every glyph of the three fixture fonts that read_variable_glyph returns, runs transform_variable_glyph; asserts either bridge_count == 0 and result.glyph equals the input, or at every location in validation_locations(result.glyph) the island count of result.glyph.instance(loc) is at most result.unbridged_count
      rejects: 'an engine that returns the replayed glyph without validating it (spike: ampersand keeps an island at wght=+1)'
    - id: validation-locations-cover-corners
      file: tests/unit/test_variable_engine_contracts.py
      test: test_validation_locations_cover_corners
      scenario: 'builds a synthetic VariableGlyph with one square contour (four ON_CURVE points, full GlyphMetadata) and supports (((''wdth'', -1.0, -1.0, 0.0),), ((''wght'', -1.0, -1.0, 0.0),), ((''wght'', 0.0, 1.0, 1.0),)), masters shifted per support, axis_tags (''wdth'', ''wght''); asserts validation_locations contains {''wdth'': -1.0, ''wght'': 1.0} and {''wdth'': -1.0, ''wght'': -1.0}, contains {}, and no location has wdth > 0; then builds a 13-axis glyph with ±1 supports on every axis and asserts at most 64 locations (the reduced set is the default, 26 single-axis extremes, all-minimum and all-maximum)'
      rejects: 'a validation_locations that checks only support peaks (the fixtures cannot catch it: every grid corner in the Ubuntu and Inter subsets is already a peak), or one that enumerates the full 3^n product for many axes'
    - id: bridge-width-config-reaches-engine
      file: tests/unit/test_variable_engine_contracts.py
      test: test_bridge_width_config_reaches_engine
      scenario: runs transform_variable_glyph on Ubuntu-VF-subset 'o' with BridgeConfig(width_percent=30) and BridgeConfig(width_percent=110); compares the default-master outlines
      rejects: an engine that builds its own BridgeConfig() and ignores the bridge argument (both outlines identical)
    - id: variable-glyph-dict-roundtrip
      file: tests/unit/test_variable_engine_contracts.py
      test: test_variable_glyph_dict_roundtrip
      scenario: calls process_variable_glyph(vg.to_dict(), BridgeConfig().model_dump(), upm, GeometryConfig().model_dump()) for Ubuntu-VF-subset 'o' inside a ProcessPoolExecutor and rebuilds the result with VariableGlyph.from_dict
      rejects: a to_dict that drops supports or masters, so the rebuilt glyph has no variation
    - id: cff2-vsindex-selection
      file: tests/unit/test_variable_engine_contracts.py
      test: test_cff2_vsindex_selection
      scenario: opens Cantarell-VF-subset in memory; (a) appends to VarStore.otVarStore.VarData a copy of VarData[0] whose VarRegionIndex lists the same regions in reverse order (update VarDataCount), decompiles the 'o' charstring and prepends [1, 'vsindex'] to its program; asserts cff2_vsindex(font, 'o') == 1 and that the supports of read_variable_glyph(font, 'o') follow VarData[1]'s region order (the reverse of the unmodified font's); (b) on a fresh copy with the same extra VarData appended but the 'o' program unmodified, sets FDArray[0].Private.vsindex = 1 and asserts read_variable_glyph(font, 'o') raises VariationDataError; (c) finds a glyph whose cff2_vsindex is None and asserts its read_variable_glyph has supports == ()
      rejects: a reader that scans only the top-level program for vsindex or assumes VarData 0 (case a reads the regions in the wrong order), or one that silently ignores a Private.vsindex default
    - id: rounded-glyph-is-validated
      file: tests/unit/test_variable_engine_contracts.py
      test: test_rounded_glyph_is_validated
      scenario: runs transform_variable_glyph on Ubuntu-VF-subset 'o' and on Cantarell-VF-subset 'o' (bridge_count >= 1 each); asserts round_variable_glyph(result.glyph) equals result.glyph point by point within 1e-9 at every support peak (the returned glyph is already rounded), that every result.glyph.default coordinate is an integer, and for the Ubuntu glyph that every vg.deltas() entry is within 1e-9 of an integer
      rejects: an engine that validates the unrounded replay and leaves rounding to the writers, so the stored glyph was never validated
    harness:
    - tests/fixtures/variable/**
  - id: variable-writers
    name: gvar and CFF2 blend writers
    description: |
      Serialize VariableGlyph (from variable-engine) into fonts. Plan:
      doc/plans/*PLAN-variable-fonts.md. Read src/stencilizer/variable/model.py,
      rounding.py and transform.py first; they are merged. Both writers store
      round_variable_glyph(vg), the values the engine validated.
      Use parallel subagents and skills to maximize performance.
      Territories below are DISJOINT. Workers NEVER spawn subagents. Spawn every
      worker BY AGENT TYPE, ALL in ONE message, each with the fixed prompt plus
      "Your brief: <path>. Read it in full before anything else."
      W3 is a codex unit: spawn loom-codex-forwarder in the FOREGROUND with
      --model gpt-5.6-terra --effort xhigh and an explicit Bash timeout of 600000 ms;
      tell it not to run git; check git status --short after it returns. If codex is
      unavailable, give W3 to loom-software-engineer.

      | Worker | Role | Tier | Files owned | Shared context | Brief path |
      | ------ | ---- | ---- | ----------- | -------------- | ---------- |
      | W1 | gvar writer | sonnet | src/stencilizer/variable/write_gvar.py, tests/unit/test_write_gvar.py | src/stencilizer/variable/model.py (read-only) | doc/plans/briefs/variable-fonts/variable-writers/w1-gvar.md |
      | W2 | CFF2 blend writer | opus | src/stencilizer/variable/write_cff2.py, tests/unit/test_write_cff2.py | src/stencilizer/io/converter.py (read-only) | doc/plans/briefs/variable-fonts/variable-writers/w2-cff2-blend.md |
      | W3 | FontWriter dispatch and names | codex terra | src/stencilizer/io/writer.py, tests/unit/test_variable_writer_names.py, tests/unit/test_review_io.py | src/stencilizer/variable/write_gvar.py (read-only) | doc/plans/briefs/variable-fonts/variable-writers/w3-writer-names.md |

      Public surface (pinned; contracts are written from it):
        stencilizer.variable.write_gvar.write_truetype_variable_glyph(font: TTFont,
          vg: VariableGlyph) -> None: first takes r = round_variable_glyph(vg) (engine
          stage, variable/rounding.py; idempotent, so a glyph from transform_variable_glyph
          is stored exactly as validated, and an untransformed glyph read from a font is
          rounded the same way); replaces glyf[name] (from r.default, every domain
          point a stored point including the converter's closing duplicate, no
          instructions; never TTGlyphPen, which drops points), moves hmtx lsb by the
          change in xMin so phantom pp1 stays put, and replaces gvar.variations[name] (one
          TupleVariation per r.supports entry, axes {tag: (start, peak, end)}, deltas
          otRound(r.deltas()) (integral within 1e-9 already); the 4 phantom-point deltas
          copied from the original tuple with the same full support after
          calcInferredDeltas; then .optimize(..., tolerance=0.0)). When the font has no
          gvar table (fvar without gvar is legal; tests/gui/conftest.py variable_font_path),
          vg.supports must be () (else ValueError), glyf and hmtx are written and no gvar
          table is created. Rounding each delta separately measured 2.0 units of error on
          flattened Ubuntu 'o'; a positive optimize tolerance can give the points of one
          bridge line different deltas and reopen a hairline gap.
        stencilizer.variable.write_cff2.write_cff2_variable_glyph(font: TTFont,
          vg: VariableGlyph) -> None: replaces the CFF2 charstring with one whose blend
          arguments reproduce vg.instance(peak) at every support peak: values from
          round_variable_glyph(vg) (absolute default rounded, deltas kept as floats);
          contours reversed back to CFF winding with list(reversed(points)); each operand
          is the relative difference of consecutive points, for the default and for each
          support's delta list separately; a blended operand is the list
          [default_rel, d_0_rel, ..., d_n-1_rel, 1] (the trailing 1 is the blend count:
          commandsToProgram only flattens the list and appends "blend", so [100, 10, 20]
          would emit "100 10 20 blend" and read 20 as the count; specializeCommands
          asserts the trailing count, specializer.py:464-465, 498-500); encoded with
          specializeCommands(generalizeFirst=False, preserveTopology=True,
          maxstack=maxStackLimit) and commandsToProgram; the program starts with
          [n, "vsindex"] (operand before operator) when reader.cff2_vsindex(font, name) is
          neither None nor 0; private and globalSubrs taken from the existing charstring;
          ValueError when len(vg.supports) differs from the region count of that VarData.
          A glyph with supports == () (cff2_vsindex None: it never blends) gets a plain
          charstring with no blend operands and no vsindex.
        stencilizer.io.writer.FontWriter.update_variable_glyph(vg: VariableGlyph) -> None
          dispatches on "glyf" vs "CFF2"; it is called only for glyphs the engine bridged,
          so every other glyph keeps its glyf/gvar or charstring bytes. FontWriter.save no
          longer rejects fvar; FontWriter.update_glyph raises FontFormatError on a font
          with fvar (the static path would leave gvar stale or drop CFF2 blends).
        stencilizer.io.writer.update_font_names also rewrites nameID 25 (Variations
          PostScript Name Prefix) and every name record referenced by an fvar instance
          postscriptNameID (not None, not 0xFFFF) with the nameID 6 rule (suffix without
          spaces before the first hyphen, appended when there is none), each record once
          even when IDs are shared; for fonts with fvar, a nameID 4 that starts with the
          family name gets the suffix right after it ("Inter Variable" -> "Inter Variable
          Stenciled"); fonts without fvar keep today's rules.
      INTEGRITY: W3 rewrites test_writer_rejects_unsupported_fonts_without_output in
      tests/unit/test_review_io.py (rejection moves from save to update_glyph); never delete
      it. The plan's "Test integrity in this plan" table lists the expected TI-edit event.
      CODEX: W3 cannot run uv run; its brief gives a static .venv/bin check. Run the writer
      contract tests yourself after all three workers return.
      CONTRACT SESSION: the frozen contract file must pass the stage gate. Before freezing,
      run uv run ruff format, uv run ruff check and uv run mypy on it. The only mypy errors
      allowed are import errors for this stage's unwritten modules (write_gvar, write_cff2,
      FontWriter.update_variable_glyph): import them inside the test functions with no
      type: ignore. fontTools imports carry # type: ignore[import-untyped]; quote cast()
      types, prefix unused stub parameters with _. Compare reread outlines through
      stencilizer.io.converter.fonttools_glyph_to_domain point by point, never raw
      RecordingPen values.
      MEMORY: record mistakes/decisions/surprises via loom memory immediately;
      NEVER loom knowledge (implementation stage); NEVER Claude Code auto-memory.
    summary: Writes bridged variable glyphs back into TrueType (glyf + gvar) and CFF2 (blend) fonts, and suffixes the variable-font name records.
    dependencies:
    - variable-engine
    parallel_group: null
    acceptance:
    - env QT_QPA_PLATFORM=offscreen uv run pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - uv run ruff check src tests packaging
    - uv run ruff format --check src tests packaging
    - uv run mypy src/stencilizer tests packaging
    setup: []
    files:
    - src/stencilizer/variable/write_gvar.py
    - src/stencilizer/variable/write_cff2.py
    - src/stencilizer/io/writer.py
    - tests/unit/test_write_gvar.py
    - tests/unit/test_write_cff2.py
    - tests/unit/test_variable_writer_names.py
    - tests/unit/test_review_io.py
    - tests/unit/test_variable_writer_contracts.py
    auto_merge: null
    working_dir: .
    stage_type: standard
    artifacts:
    - src/stencilizer/variable/write_gvar.py
    - src/stencilizer/variable/write_cff2.py
    - src/stencilizer/io/writer.py
    wiring:
    - source: src/stencilizer/variable/write_gvar.py
      pattern: round_variable_glyph\(
      description: gvar writer stores the engine's rounded values
    - source: src/stencilizer/variable/write_cff2.py
      pattern: round_variable_glyph\(
      description: CFF2 writer stores the engine's rounded values
    - source: src/stencilizer/io/writer.py
      pattern: write_truetype_variable_glyph\(
      description: FontWriter dispatches TrueType variable glyphs to the gvar writer
    - source: src/stencilizer/io/writer.py
      pattern: write_cff2_variable_glyph\(
      description: FontWriter dispatches CFF2 variable glyphs to the blend writer
    before_stage:
    - command: test -e src/stencilizer/variable/write_gvar.py
      exit_code: 1
      description: No gvar writer at the base commit
    after_stage:
    - command: uv run python -c "import stencilizer.variable.write_gvar, stencilizer.variable.write_cff2"
      exit_code: 0
      description: Writer modules import
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
    - codex
    subagent_timeout_secs: 1200
    skills:
    - loom-python
    contracts:
    - id: gvar-output-reproduces-instances
      file: tests/unit/test_variable_writer_contracts.py
      test: test_gvar_output_reproduces_instances
      scenario: transforms Ubuntu-VF-subset 'o' with transform_variable_glyph (asserting bridge_count >= 1), writes it with FontWriter(font, tmp_path/'out.ttf').update_variable_glyph then save(), reopens the file, and for every location in validation_locations(result.glyph) converts getGlyphSet(location=loc, normalized=True)['o'] through fonttools_glyph_to_domain('o', glyph, font) and compares it with result.glyph.instance(loc) point by point (same contour and point counts) within 0.01 units (result.glyph is already rounded by the engine, so the stored values are exactly the validated ones), and asserts GlyphAnalyzer finds no island in the reread outline at each location
      rejects: a writer that replaces glyf['o'] but keeps the old gvar tuples, whose point count no longer matches, or one that rounds each delta separately (2-unit error at the Ubuntu corner tuples)
    - id: gvar-keeps-phantom-deltas
      file: tests/unit/test_variable_writer_contracts.py
      test: test_gvar_keeps_phantom_deltas
      scenario: 'after writing the transformed Ubuntu-VF-subset ''o'' and reopening, for each gvar tuple of ''o'' in input and output (matched by axes), deep-copies it, calls calcInferredDeltas(coords, controls[1]) with coords, controls = font[''glyf'']._getCoordinatesAndControls(''o'', font[''hmtx''].metrics), and asserts the last 4 coordinates (phantom deltas, None as (0, 0)) are equal; asserts at least one input phantom delta is non-zero (planning: pp2 moves by -129 under wdth -1), so the check cannot pass on zeros; also asserts the output hmtx entry of ''o'' keeps the input advance width'
      rejects: 'a writer that sets the 4 phantom-point deltas to zero (a width check through the glyph set cannot see this: with HVAR present the width comes from HVAR)'
    - id: gvar-writer-handles-missing-gvar
      file: tests/unit/test_variable_writer_contracts.py
      test: test_gvar_writer_handles_missing_gvar
      scenario: builds Roboto-Regular plus an fvar table and no gvar exactly as tests/gui/conftest.py variable_font_path does (into tmp_path); reads 'O' with read_variable_glyph (supports == ()); runs transform_variable_glyph (bridge_count >= 1); writes it with FontWriter(font, tmp_path/'out.ttf').update_variable_glyph and save(); reopens and asserts 'gvar' not in the output, 'fvar' kept, and GlyphAnalyzer finds no island in fonttools_glyph_to_domain('O', getGlyphSet()['O'], font)
      rejects: a writer that indexes font['gvar'] unconditionally (KeyError on an fvar-only font) or creates an empty gvar table
    - id: cff2-blend-output-reproduces-instances
      file: tests/unit/test_variable_writer_contracts.py
      test: test_cff2_blend_output_reproduces_instances
      scenario: 'transforms Cantarell-VF-subset ''o'' (asserting bridge_count >= 1), writes it with FontWriter.update_variable_glyph and save(), reopens, and at {''wght'': -1.0}, {} and {''wght'': 1.0} converts the reread glyph through fonttools_glyph_to_domain (which reverses CFF2 winding) and compares it with result.glyph.instance(loc) point by point (same point counts) within 1 unit'
      rejects: a writer that stores a static default charstring without blend operators, so every location shows the default outline
    - id: variable-names-suffixed
      file: tests/unit/test_variable_writer_contracts.py
      test: test_variable_names_suffixed
      scenario: saves Inter-VF-subset (nameID 1 and 4 'Inter Variable'; nameID 25 'InterVariable'; instance postscriptNameID 280 'InterVariable-Thin') through FontWriter(font, tmp_path/'out.ttf').save() and asserts nameID 25 == 'InterVariableStenciled', nameID 280 == 'InterVariableStenciled-Thin' and nameID 4 == 'Inter Variable Stenciled'
      rejects: a writer that only rewrites nameIDs 1, 4, 6 and 16, leaving the variations PostScript prefix and instance PostScript names unsuffixed, or splits 'Inter Variable' into 'Inter Stenciled Variable'
    - id: static-writer-refuses-variable
      file: tests/unit/test_variable_writer_contracts.py
      test: test_static_writer_refuses_variable
      scenario: opens Ubuntu-VF-subset with TTFont (FontReader still rejects fvar until variable-surfaces), builds the default Glyph of 'o' with fonttools_glyph_to_domain('o', font.getGlyphSet()['o'], font), and calls FontWriter(font, tmp_path/'out.ttf').update_glyph(glyph); expects FontFormatError and no file at tmp_path/'out.ttf'
      rejects: a writer whose fvar rejection was simply deleted, so the static path writes glyf and leaves gvar stale
  - id: variable-surfaces
    name: Processor, CLI and GUI wiring
    description: |
      Wire the variable engine and writers into the processor, CLI and GUI. Plan:
      doc/plans/*PLAN-variable-fonts.md. Engine and writers are merged; read
      src/stencilizer/variable/transform.py, model.py and io/writer.py first.
      Use parallel subagents and skills to maximize performance.
      Territories below are DISJOINT. Workers NEVER spawn subagents.
      WAVE 1: W1 (processing foundation; W3 calls process_variable_font through
      FontProcessor). WAVE 2: W2 and W3 in ONE message after W1 returns.
      W2 is a codex unit: spawn loom-codex-forwarder in the FOREGROUND with
      --model gpt-5.6-terra --effort xhigh and an explicit Bash timeout of 600000 ms;
      tell it not to run git; check git status --short after it returns. If codex is
      unavailable, give W2 to loom-software-engineer.

      | Worker | Role | Tier | Files owned | Shared context | Brief path |
      | ------ | ---- | ---- | ----------- | -------------- | ---------- |
      | W1 | Variable processing and dispatch | sonnet | src/stencilizer/variable/processing.py, src/stencilizer/core/processor.py, src/stencilizer/io/reader.py, tests/unit/test_variable_processing.py, tests/unit/test_review_io.py, tests/unit/test_processor.py, tests/unit/test_processor_more.py | src/stencilizer/io/writer.py (read-only) | doc/plans/briefs/variable-fonts/variable-surfaces/w1-processing.md |
      | W2 | CLI --instance | codex terra | src/stencilizer/io/instance.py, src/stencilizer/cli/app.py, src/stencilizer/cli/handlers.py, src/stencilizer/cli/output.py, tests/unit/test_instance.py | src/stencilizer/core/processor.py, src/stencilizer/variable/processing.py (read-only) | doc/plans/briefs/variable-fonts/variable-surfaces/w2-cli-instance.md |
      | W3 | GUI variable sessions and axis sliders | sonnet | src/stencilizer/gui/session.py, src/stencilizer/gui/variable_session.py, src/stencilizer/gui/controller.py, src/stencilizer/gui/controls.py, src/stencilizer/gui/axis_controls.py, src/stencilizer/gui/main_window.py, src/stencilizer/gui/theme.py, tests/gui/test_session.py, tests/gui/test_axis_controls.py, tests/gui/test_variable_session.py | tests/gui/conftest.py (read-only) | doc/plans/briefs/variable-fonts/variable-surfaces/w3-gui.md |

      W1's dispatch (is_variable(reader.font)) breaks 8 existing tests whose readers are
      plain Mock() objects with no .font (tests/unit/test_processor.py:116-122, 154-160;
      tests/unit/test_processor_more.py:36-42, 85-91, 124-130, 269-275); W1 owns both files
      and adds mock_reader.font = MagicMock() without changing an assertion.
      IMPORTS: core/__init__.py:34 imports core.processor eagerly, so core/processor.py must
      import stencilizer.variable.processing only inside functions (a module-level import
      is a reproduced ImportError cycle).
      INTEGRITY: W1 rewrites the reader-rejection test in tests/unit/test_review_io.py and W3
      the fvar assertion in tests/gui/test_session.py into positive tests; the plan's "Test
      integrity in this plan" table lists the expected TI-edit events.
      CODEX: W2 cannot run uv run; its brief gives a static .venv/bin check. Run its tests,
      tests/regression/test_code_structure.py and `uv run stencilizer --help` yourself after
      it returns (cli/app.py is 393 of 400 lines, stencilize 46 of 50 effective lines).
      GUI PROBES: every Qt run uses QT_QPA_PLATFORM=offscreen; never show a window on the
      host display.

      Public surface (pinned; contracts are written from it):
        CLI: stencilize INPUT [-o OUTPUT] [--instance SPEC] (Typer app stencilizer.cli.app.app).
          Without --instance, a variable input produces a variable output (fvar and gvar or
          CFF2 kept). --instance "wght=700,wdth=90" uses user-space axis values; axes not
          named are pinned at their fvar default; the result is a static font (no fvar)
          stenciled by the static pipeline. An unknown axis tag, a malformed pair, or a
          value outside the axis range exits with code 1 and a message naming the axis;
          --instance on a non-variable font exits 1. --list-islands and --dry-run on a
          variable font report the same island glyphs processing uses (overlap-built
          counters included).
        stencilizer.io.instance: parse_instance_spec(spec: str, font: TTFont) ->
          dict[str, float] (the only range and tag validation: fontTools' instancer clamps
          out-of-range values and raises a bare KeyError for unknown axes);
          instantiate_static(font_path: Path, spec: str, workdir: Path) -> Path (writes
          the static instance into workdir via instantiateVariableFont(TTFont(font_path),
          limits, static=True, overlap=OverlapMode.REMOVE, downgradeCFF2=True,
          updateFontNames="STAT" in font); without overlap removal the static pipeline
          finds no island in overlap-built counters, measured 0 islands in Inter A D P R e
          4 & at wght=700). Naming is optional and never decides success: when the call
          with updateFontNames=True raises ValueError (fontTools finds no STAT AxisValue
          for the coordinate; Inter wght=650 raises "Cannot find Axis Values"), it reloads
          TTFont(font_path) and repeats the call with updateFontNames=False, keeping the
          variable font's names (FontWriter then adds the Stenciled suffix). A ValueError
          from the second call propagates.
        stencilizer.cli.output.print_font_info(..., axes: str | None = None) prints
          "Variable axes: wght 100–900, opsz 14–32" (fvar order, minValue–maxValue) when
          axes is given; the CLI passes it for fonts with fvar unless --instance is set,
          on the standard run, --dry-run and --list-islands.
        stencilizer.core.processor.GlyphClassification gains
          unsupported_islands: dict[str, int] = field(default_factory=dict) (empty for
          static fonts, so static behaviour and tests are unchanged).
        stencilizer.variable.processing:
          classify_variable_glyphs(processor: FontProcessor, reader: FontReader) ->
            tuple[GlyphClassification, dict[str, VariableGlyph]] (per glyph: read, flatten,
            compatible overlap removal, processor.analyzer.analyze(default, upm); when
            overlap removal returns None the flattened default is analyzed instead, as
            transform_variable_glyph counts it; a VariationDataError from
            read_variable_glyph skips the glyph as "unsupported variation data" and, when
            its static default outline (fonttools_glyph_to_domain on font.getGlyphSet())
            has islands, records their count in unsupported_islands[name]; a
            VariationDataError from flattening after a successful read keeps the glyph in
            glyphs_to_process when the raw default has islands, and the worker returns the
            no-op outcome with those islands counted. Any other exception, i.e. font data
            fontTools cannot decode, propagates as a font-level failure, as in the static
            pipeline);
          variable_island_counts(reader: FontReader) -> list[tuple[str, int]];
          process_variable_font(processor: FontProcessor, reader: FontReader,
            output_path: Path, max_workers: int | None, stats: ProcessingStats,
            progress_callback: ProgressCallback | None,
            classification: GlyphClassification | None,
            directions: Mapping[str, BridgeDirection] | None) -> None, filling the stats
            object FontProcessor.process created in place (process returns that object and
            stamps its times; never rebind or return a new ProcessingStats), adding
            sum(classification.unsupported_islands.values()) to stats.unbridged_count;
            called from FontProcessor._process_loaded_font when
            is_variable(reader.font); it reuses FontProcessor._process_glyphs_parallel
            (worker=process_variable_glyph, rebuild=VariableGlyph.from_dict) and
            FontProcessor._save_font(..., variable=True), whose new keyword-only
            parameters default to today's static behaviour. FontProcessor.classify_glyphs
            returns classify_variable_glyphs(self, reader)[0] for variable fonts.
        GUI: FontSession.open accepts variable fonts; FontSession.axes ->
          tuple[AxisInfo(tag, name, minimum, default, maximum), ...] (empty for static;
          name falls back to the tag), defined Qt-free in stencilizer.gui.variable_session
          with normalize(location_user, axes, avar_segments) -> dict[str, float] (fvar
          normalization then avar: Inter wght 700 -> about 0.54, not 0.6);
          FontSession.preview(name, bridge, geometry, directions=None, location=None)
          where location is user-space {tag: value}; GuiController.set_location(
          location: dict[str, float]); ControlPanel shows an "AXES" card (AxisPanel, a
          QFrame with role "card") with one slider per axis (object names
          "axis-slider-<tag>") only for variable fonts. FontSession.is_composite(name) ->
          bool. AxisPanel.set_location_applies(applies: bool): False disables every axis
          slider and spin box and shows a QLabel with object name "axis-composite-note"
          and text "Composite glyphs preview at the default axis location."; True
          re-enables them and hides the note. MainWindow calls it with
          not session.is_composite(name) whenever the selected glyph changes.
          Unsupported glyphs (classification.unsupported_islands) are listed among the
          session's display glyphs; their preview returns the original outline, no
          stenciled outline, and the error "unsupported variation data"; the survey
          reports them as unbridged.
      Composite glyphs and glyphs whose replay fails stay untouched in the output (their
      glyf/gvar or charstring unchanged), exactly like static no-op glyphs.
      COMPOSITES (product rule, not a docstring note): in variable GUI sessions a
      composite previews at the default location. Its sources come from the variable
      engine at {} and the composition uses the default component offsets, exactly the
      seam DONE-PLAN-gui-bridge-direction built (gui/composites.py compose,
      FontSession._preview_composite), which stays unchanged for static fonts. Composite
      gvar deltas move component offsets per location; following them is out of scope.
      The UI states the limit (set_location_applies above) and the
      composite-preview-at-default contract pins it, including the saved font's composite
      at the default location against the preview.
      PREVIEW COST AND CACHE: an uncached variable preview runs overlap removal, replay,
      rounding and up to 64 validation locations synchronously on the GUI thread (the
      spike measured about 12-14 ms per glyph on the fixtures against the static
      3.6-10.5 ms; measure the worst fixture glyph cold and record it in loom memory; if
      any fixture glyph exceeds 100 ms cold, record it with loom memory note and report
      it, because moving previews off the GUI thread is a separate change). The cache is
      owned by FontSession (a new session on every open, so reopening drops it), keyed by
      (name, bridge.model_dump_json(), geometry.model_dump_json(), direction): every
      input of transform_variable_glyph except upm, which is fixed per session
      (BridgeConfig is not hashable). It is a VariableOutcomeCache(max_entries=64) in
      gui/variable_session.py holding VariableOutcome entries in least-recently-used
      order (collections.OrderedDict); a request whose bridge or
      geometry key differs from the cached entries' clears it. The survey
      (FontSession.unbridged) runs on a pool thread (gui/controller.py _run_survey)
      while previews run on the GUI thread; it stores only (bridge_count) per key in a
      separate dict, never outlines, so a survey over a large font cannot fill memory
      with outlines. Both dicts are guarded by one threading.Lock, held only for lookups
      and inserts, never while transform_variable_glyph runs (two threads may compute the
      same key once each; the second insert wins and both results are equal).
      Slider moves then only evaluate instance() on a cached outcome.
      CONTRACT SESSION: the frozen contract files must pass the stage gate. Before freezing,
      run uv run ruff format, uv run ruff check and uv run mypy on them. The only mypy errors
      allowed are import errors for this stage's unwritten modules (variable.processing,
      io.instance, gui.variable_session, gui.axis_controls): import them inside the test
      functions with no type: ignore. GUI contracts define their own window fixture (it is
      per file, not in conftest), run offscreen, and never call set_start_method.
      MEMORY: record mistakes/decisions/surprises via loom memory immediately;
      NEVER loom knowledge (implementation stage); NEVER Claude Code auto-memory.
    summary: 'Makes variable fonts usable end to end: the CLI stencils them (or pins a static instance with --instance), and the GUI opens, previews at any axis location with sliders, and saves them.'
    dependencies:
    - variable-writers
    parallel_group: null
    acceptance:
    - env QT_QPA_PLATFORM=offscreen uv run pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - uv run ruff check src tests packaging
    - uv run ruff format --check src tests packaging
    - uv run mypy src/stencilizer tests packaging
    - env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help
    - uv run python -c "import sys, stencilizer.cli.app, stencilizer.variable.processing, stencilizer.gui.session, stencilizer.gui.variable_session; assert not any(m.startswith('PySide6') for m in sys.modules)"
    setup: []
    files:
    - src/stencilizer/variable/processing.py
    - src/stencilizer/core/processor.py
    - src/stencilizer/io/reader.py
    - src/stencilizer/io/instance.py
    - src/stencilizer/cli/**
    - src/stencilizer/gui/**
    - tests/unit/test_variable_processing.py
    - tests/unit/test_instance.py
    - tests/unit/test_review_io.py
    - tests/unit/test_processor.py
    - tests/unit/test_processor_more.py
    - tests/unit/test_variable_surface_contracts.py
    - tests/gui/test_session.py
    - tests/gui/test_axis_controls.py
    - tests/gui/test_variable_session.py
    - tests/gui/test_variable_gui_contracts.py
    auto_merge: null
    working_dir: .
    stage_type: standard
    artifacts:
    - src/stencilizer/variable/processing.py
    - src/stencilizer/core/processor.py
    - src/stencilizer/io/instance.py
    - src/stencilizer/cli/app.py
    - src/stencilizer/cli/handlers.py
    - src/stencilizer/cli/output.py
    - src/stencilizer/gui/session.py
    - src/stencilizer/gui/variable_session.py
    - src/stencilizer/gui/controller.py
    - src/stencilizer/gui/controls.py
    - src/stencilizer/gui/axis_controls.py
    - src/stencilizer/gui/main_window.py
    wiring:
    - source: src/stencilizer/core/processor.py
      pattern: unsupported_islands
      description: GlyphClassification carries the island counts of glyphs whose variation data is unsupported
    - source: src/stencilizer/gui/session.py
      pattern: classify_variable_glyphs\(
      description: FontSession.open classifies variable fonts through the variable path
    - source: src/stencilizer/gui/session.py
      pattern: VariableOutcomeCache\(
      description: FontSession owns the bounded variable preview cache
    - source: src/stencilizer/gui/main_window.py
      pattern: set_location_applies\(
      description: selecting a composite disables the axis sliders and shows the default-location note
    - source: src/stencilizer/cli/app.py
      pattern: axes=
      description: the CLI passes the variable axes to print_font_info
    - source: src/stencilizer/core/processor.py
      pattern: process_variable_font\(
      description: FontProcessor._process_loaded_font dispatches variable fonts
    - source: src/stencilizer/core/processor.py
      pattern: classify_variable_glyphs\(
      description: FontProcessor.classify_glyphs dispatches variable fonts
    - source: src/stencilizer/cli/app.py
      pattern: instantiate_static\(
      description: CLI --instance pins a static instance before processing
    - source: src/stencilizer/cli/handlers.py
      pattern: variable_island_counts\(
      description: --list-islands and --dry-run (moved to cli/handlers.py) use the variable island scan
    - source: src/stencilizer/gui/main_window.py
      pattern: controller\.set_location\b
      description: Axis slider changes reach the controller (a direct connect, as main_window.py:123 does for select_glyph, or a call)
    before_stage:
    - command: test -e src/stencilizer/io/instance.py
      exit_code: 1
      description: No --instance support at the base commit
    after_stage:
    - command: env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help
      stdout_contains:
      - --instance
      exit_code: 0
      description: CLI exposes --instance (NO_COLOR and COLUMNS keep Rich from splitting or truncating option names)
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
    - codex
    subagent_timeout_secs: 1200
    skills:
    - loom-python
    contracts:
    - id: cli-writes-variable-stencil
      file: tests/unit/test_variable_surface_contracts.py
      test: test_cli_writes_variable_stencil
      scenario: invokes typer.testing.CliRunner().invoke(app, [str(Ubuntu-VF-subset), '-o', str(tmp_path/'out.ttf'), '-q']); reopens the output; asserts exit code 0, 'fvar' and 'gvar' present, getTableData of fvar, avar, STAT and HVAR equal to the input's, 'o' bridged (its default outline differs from the input's) and no island in 'o' at normalized wght -1.0 and +1.0
      rejects: a CLI that stencils only the default master and writes a static font, keeps fvar with a stale gvar, or rewrites the variation tables the plan keeps untouched
    - id: untouched-glyph-keeps-variations
      file: tests/unit/test_variable_surface_contracts.py
      test: test_untouched_glyph_keeps_variations
      scenario: after the CLI run on Ubuntu-VF-subset, compares glyph 'l' (no counter) glyf coordinates and its gvar TupleVariation coordinates in output and input
      rejects: a processor that rewrites every glyph through the variable writer, re-solving and re-optimizing deltas of glyphs it did not bridge
    - id: cli-instance-pins-static
      file: tests/unit/test_variable_surface_contracts.py
      test: test_cli_instance_pins_static
      scenario: 'invokes the CLI on Inter-VF-subset with [''--instance'', ''wght=700'', ''-o'', out, ''-q'']; asserts the output has no fvar, that ''o'' advance width equals that of instantiateVariableFont(TTFont(input), {''wght'': 700, ''opsz'': 14}, static=True), and that after fontTools.ttLib.removeOverlaps.removeOverlaps(output_font, [''P'']) on a reopened copy of the output, GlyphAnalyzer finds no island in ''P'' (an unbridged self-overlapping ''P'' gains its counter back as an island under overlap removal, so this check cannot pass vacuously); then invokes it again with [''--instance'', ''wght=650'', ''-o'', out2, ''-q''] (in range, but no STAT AxisValue names 650) and asserts exit code 0, an output without fvar, and a nameID 1 ending in ''Stenciled'''
      rejects: an --instance option that is parsed but ignored, leaving a variable output, or one that instances without overlap removal, so overlap-built counters such as 'P' are never bridged, or one that lets STAT naming fail an in-range value (updateFontNames=True raises 'Cannot find Axis Values' for wght=650)
    - id: cli-shows-variable-axes
      file: tests/unit/test_variable_surface_contracts.py
      test: test_cli_shows_variable_axes
      scenario: with env NO_COLOR=1 and COLUMNS=200 set through monkeypatch, invokes CliRunner().invoke(app, [str(Inter-VF-subset), '--dry-run']) and asserts exit code 0, 'Variable axes' in the output, and 'wght 100' and '900' in it; then invokes [str(Inter-VF-subset), '--dry-run', '--instance', 'wght=700'] and asserts 'Variable axes' is absent
      rejects: a print_font_info axes parameter that no caller passes, or a CLI that lists axes for a pinned static instance
    - id: variable-output-valid-everywhere
      file: tests/unit/test_variable_surface_contracts.py
      test: test_variable_output_valid_everywhere
      scenario: 'for each of Ubuntu-VF-subset.ttf, Inter-VF-subset.ttf and Cantarell-VF-subset.otf: classification = FontProcessor(StencilizerSettings()).classify_glyphs(reader) on a FontReader of the input, then stats = FontProcessor(StencilizerSettings()).process(font_path=input, output_path=tmp_path/name, classification=classification); asserts stats.error_count == 0 and stats.bridges_added >= 1; reopens input and output with TTFont; a glyph is modified when its glyf data (glyf[name].compile(glyf)) or CFF2 charstring bytecode (after compile()) differs; asserts at least one modified glyph per font and that every modified name is in {g.name for g in classification.glyphs_to_process}; for every modified glyph, reads read_variable_glyph(output, name) and asserts that for every loc in validation_locations(vg) the island count of vg.instance(loc) is at most the island count at {}; and asserts getTableData equal between input and output for every table of fvar, avar, STAT, HVAR and MVAR that the input has'
      rejects: a writer whose rounding or IUP optimization reopens a bridge gap at some location (the in-memory engine check passes, the saved font fails), a processor that rewrites glyphs it never classified, or one that touches the preserved variation tables
    - id: unsupported-glyph-counted
      file: tests/unit/test_variable_surface_contracts.py
      test: test_unsupported_glyph_counted
      scenario: monkeypatches stencilizer.variable.processing.read_variable_glyph (the name processing.py imports) to raise VariationDataError for Ubuntu-VF-subset 'o' and delegate otherwise; classification = processor.classify_glyphs(reader); asserts classification.skipped_reasons['o'] == 'unsupported variation data' and classification.unsupported_islands['o'] == 1; stats = processor.process(font_path=input, output_path=tmp_path/'out.ttf', classification=classification); asserts stats.bridges_added >= 1 (other glyphs still bridged), stats.unbridged_count >= 1, and that 'o' glyf data and gvar tuples in the output equal the input's
      rejects: a classification that aborts the font on one glyph's VariationDataError, or that skips the glyph without counting its counter as unbridged
    - id: cli-instance-rejects-bad-spec
      file: tests/unit/test_variable_surface_contracts.py
      test: test_cli_instance_rejects_bad_spec
      scenario: invokes the CLI on Inter-VF-subset with --instance values 'wdth=100' (axis absent), 'wght=5000' (out of range) and 'wght' (no value), each into its own tmp output
      rejects: an instancer call that silently drops unknown axes or clamps out-of-range values and exits 0
    - id: session-previews-at-location
      file: tests/gui/test_variable_gui_contracts.py
      test: test_session_previews_at_location
      scenario: 'FontSession.open(Inter-VF-subset, processor); preview(''o'', BridgeConfig(), GeometryConfig(), location={''wght'': 300}) and location={''wght'': 700}; asserts the stenciled outlines differ and GlyphAnalyzer finds no island in either; computes the expected normalized location in the test with fontTools.varLib.models.normalizeLocation({''wght'': 700, ''opsz'': 14}, fvar axes) followed by piecewiseLinearMap through font[''avar''].segments per axis, and asserts the 700 preview equals transform_variable_glyph(read_variable_glyph(font, ''o''), BridgeConfig(), GeometryConfig(), upm).glyph.instance(expected) point by point within 0.5'
      rejects: a session that ignores location and always previews the default master, or one that normalizes through fvar only and skips avar (Inter wght 700 is 0.6 without avar, about 0.54 with it)
    - id: axis-sliders-drive-preview
      file: tests/gui/test_variable_gui_contracts.py
      test: test_axis_sliders_drive_preview
      scenario: opens Ubuntu-VF-subset in MainWindow via load_session-style helpers, finds the QSlider named 'axis-slider-wght', records the stenciled outline of the current preview, moves the slider to its maximum, waits for controller.preview_ready, and asserts the newly emitted stenciled outline differs from the recorded one
      rejects: an AXES card whose sliders are built but never connected to GuiController.set_location (the preview outline does not change)
    - id: gui-saves-variable-font
      file: tests/gui/test_variable_gui_contracts.py
      test: test_gui_saves_variable_font
      scenario: loads Ubuntu-VF-subset in a GuiController, calls save(tmp_path/'out.ttf'), waits for save_finished, and checks the output keeps fvar and gvar and that 'o' has no island at normalized wght +1.0
      rejects: a GUI save path that writes only default-master glyphs through the static writer
    - id: composite-preview-at-default
      file: tests/gui/test_variable_gui_contracts.py
      test: test_composite_preview_at_default
      scenario: 'FontSession.open(Inter-VF-subset, processor); asserts session.is_composite(''Aacute''); previews ''Aacute'' with location=None and with location={''wght'': 700}, asserts both stenciled outlines exist, are equal, and that GlyphAnalyzer finds no island in them; saves through session.save to tmp_path; composes the saved Aacute at the default location from the output with the session''s own CompositeGlyph entry for ''Aacute'' (gui.composites.compose(composite, load_component_outlines(reader, [composite])) inside `with FontReader(output) as reader`; the converter alone reads composites with 0 contours) and asserts outlines_match(saved, preview.stenciled), as tests/gui/test_session.py test_save_writes_stenciled_outlines does for ''O''; then opens the same font in MainWindow, selects ''Aacute'' and asserts the QSlider ''axis-slider-wght'' is disabled and the QLabel ''axis-composite-note'' is visible, selects ''o'' and asserts the slider is enabled and the note hidden'
      rejects: a session whose composite preview runs the static pipeline (no bridge in Inter's overlap-built A) or silently ignores the sliders without telling the user
    reachable:
    - symbol: process_variable_font
      from: stencilize
      description: variable processing is reached from the CLI command (through FontProcessor.process and a function-level import; if loom's graph cannot follow that import, record a memory and dispute this field with dispute-criteria citing the core/__init__.py import cycle; the cli-writes-variable-stencil contract proves the path end to end)
    - symbol: instantiate_static
      from: stencilize
      description: --instance is reached from the CLI command
  - id: integration-verify
    name: Integration Verification
    description: |
      Final verification after all stages. Verify FUNCTIONAL INTEGRATION, not just tests
      passing. NEVER Claude Code auto-memory.
      CONTEXT: read doc/plans/*PLAN-variable-fonts.md, loom memory show --all, and the
      knowledge sections patterns/bridge-algorithm.md "Contour surgery" and
      architecture.md "Font format I/O".
      BUILD & TEST (zero tolerance; fix ALL warnings/errors): the acceptance list below.
      The full suite runs under pytest-xdist (--numprocesses=16, added by cff2-static): serially it takes about 600 s, past the
      300 s acceptance cap; with 16 workers it measured 69 s at 4254909.
      CODE REVIEW: spawn parallel loom-code-reviewer subagents: (1) correctness of
      src/stencilizer/variable/ replay, solver and writers against the OpenType gvar and
      CFF2 specs; (2) CLI/GUI wiring, error handling and the no-op glyph rule;
      (3) architecture, size limits and test coverage. Fix every finding with an engineer
      agent or dispute it; never defer one.
      SUGGESTIONS: weigh every pending reviewer suggestion the signal lists; resolve each one
      implemented with loom memory resolve <id> --outcome implemented --reason <what changed>.
      FUNCTIONAL: run the CLI end to end on tests/fixtures/variable/*.{ttf,otf} into a temp
      dir and on a CFF2 conversion of CommitMono; reopen each output with fontTools and
      confirm fvar kept (variable inputs) and no island in 'o' at every axis extreme. Run
      --list-islands on Inter-VF-subset and confirm P is listed. Open a variable fixture in
      the GUI with QT_QPA_PLATFORM=offscreen set on the command line (never rely on the
      conftest default; the host session exports wayland;xcb) and save. Write every temp
      output under pytest's tmp_path or $TMPDIR, never the worktree.
      Record discoveries to loom memory for knowledge-distill, including any knowledge file
      contradicted by the tree: loom memory note "stale-knowledge: ...".
    summary: Confirms variable and CFF2 fonts go through the real CLI and GUI paths, the full suite passes, and reviewers' findings are fixed.
    dependencies:
    - variable-surfaces
    parallel_group: null
    acceptance:
    - env QT_QPA_PLATFORM=offscreen uv run pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - env UV_PROJECT_ENVIRONMENT=.venv/py311 QT_QPA_PLATFORM=offscreen uv run --python 3.11 --frozen --all-extras pytest --numprocesses=16 --no-cov -q -p no:cacheprovider
    - uv run ruff check src tests packaging
    - uv run ruff format --check src tests packaging
    - uv run mypy src/stencilizer tests packaging
    - uv lock --check
    - env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help
    setup: []
    files: []
    auto_merge: null
    working_dir: .
    stage_type: integration-verify
    artifacts: []
    wiring:
    - source: src/stencilizer/core/processor.py
      pattern: process_variable_font\(
      description: variable fonts dispatched from FontProcessor._process_loaded_font
    wiring_tests:
    - name: CLI help lists --instance
      command: env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help
      success_criteria:
        exit_code: 0
        stdout_contains:
        - --instance
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
  - id: knowledge-distill
    name: Knowledge Distillation
    description: |
      Curate all stage memories into permanent knowledge; update user docs.
      NEVER Claude Code auto-memory.
      SINGLE-AGENT: do NOT spawn subagents; memories are compact summaries; keep code
      spot-reads narrow.
      START with loom memory pending --group (corrections, mistakes, decisions, other);
      read doc/plans/*PLAN-variable-fonts.md and the knowledge sections it touches.
      CORRECTIONS FIRST: apply every stale-knowledge memory in place with
      loom knowledge replace-section <file> "<heading>" "<body>", never with update.
      Known stale sections after this plan: concerns.md "Unsupported font formats" (delete
      it), concerns.md "GUI tests honour an inherited QT_QPA_PLATFORM" (delete it once
      cff2-static's conftest change is merged), architecture.md "Font format I/O" and
      "Font format and error boundary", stack.md "Supported font formats" (add skia-pathops
      as a runtime dependency and pytest-xdist as a dev dependency to stack.md "Runtime and
      tooling"), architecture.md "Processing pipeline" (variable dispatch),
      architecture/gui.md "GUI package layout" (add variable_session.py and
      axis_controls.py), and entry-points.md "CLI" (add --instance; the list-islands and
      dry-run handlers live in cli/handlers.py). concerns.md "Full test suite near the acceptance time cap"
      no longer holds once pytest-xdist is installed: rewrite it to the measured xdist time.
      Then curate: a new tier-2 topic patterns/variable-replay (replay pipeline, bridge-line
      grouping, realignment and projection, validation locations, delta solve against
      original supports, measured failure rates) with a 2-4 line summary and link in
      patterns.md; mistakes, decisions and conventions from memory.
      TIER ROUTING: findings ~40 lines or fewer inline in tier-1; larger via
      loom knowledge update <category>/<slug> plus a tier-1 summary and link.
      README.md: update the "Font Format Support" table (CFF2 static and variable fonts
      supported), the Graphical Interface note that rejects variable and CFF2 fonts, the
      "OTF with CFF2 outlines" bullet, "Future Work", and document --instance under Usage.
      SUGGESTIONS: record every unimplemented reviewer suggestion in concerns or its topic,
      then resolve it promoted, merged or discarded.
      RECEIPTS: every memory taken into knowledge gets loom memory resolve <id>
      --outcome promoted|merged|discarded|deferred right after the write that used it;
      finish with loom memory pending --strict and resolve whatever it lists.
      LAST, if this stage removed structural issues:
      loom knowledge check --write-baseline doc/loom/knowledge/check-baseline.txt
    summary: Records the variable-font replay design and what the plan learned in the knowledge base, and updates the README format table and GUI notes.
    dependencies:
    - integration-verify
    parallel_group: null
    acceptance:
    - loom knowledge check --strict --baseline doc/loom/knowledge/check-baseline.txt
    - loom memory pending --strict
    - rg -qF -- "--instance" README.md
    - '! rg -qF "These are not yet supported" README.md'
    - '! rg -qF "Variable fonts and CFF2 fonts are rejected" README.md'
    - '! rg -q "(CFF2|fvar).*Not supported" README.md'
    setup: []
    files:
    - doc/loom/knowledge/**
    - README.md
    auto_merge: null
    working_dir: .
    stage_type: knowledge-distill
    artifacts:
    - README.md
    wiring: []
    context_ceiling_tokens: null
    sandbox: {}
    ultracode: false
    implementers:
    - claude
```

<!-- END loom METADATA -->
