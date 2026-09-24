# Code Review: Per-glyph bridge direction and a complete glyph grid

**Plan:** PLAN-gui-bridge-direction | **Generated:** 2026-09-24 21:55 UTC

## Summary

## Overview

## Changes by Stage

### Per-glyph bridge direction and complete glyph grid (bridge-direction)

**Status:** completed  
**Purpose:** Add a per-glyph bridge direction (Auto / Vertical / Horizontal) to the GUI preview and
save, list composite glyphs that draw a bridged island glyph in the grid, and report
glyphs where no bridge could be placed.
Use parallel subagents and skills to maximize performance.

CONTRACT: doc/plans/briefs/gui-bridge-direction/bridge-direction/_shared.md pins every
signature, the direction semantics table, the repo rules (mypy strict, ruff, 400/50/300
size limits, frozen tests/regression, Qt-free session and composites, queued
connections) and the measured facts the tests use. Read it before spawning; do not
re-derive it.

FOUNDATION (you, before any spawn; two files, under 20 lines):
1. src/stencilizer/config/settings.py: add the BridgeDirection StrEnum (AUTO "auto",
   VERTICAL "vertical", HORIZONTAL "horizontal") above BridgeConfig and the field
   BridgeConfig.direction: BridgeDirection = Field(default=BridgeDirection.AUTO, ...),
   exactly as _shared.md shows.
2. src/stencilizer/core/surgery_context.py: SurgeryContext gains
   direction: BridgeDirection = BridgeDirection.AUTO directly after use_spanning;
   SurgeryContext.merge sets force_horizontal/force_vertical from it only when the
   caller forced neither (the snippet in _shared.md).
3. Prove it: uv run pytest --no-cov -q -p no:cacheprovider tests/regression
   tests/unit/test_surgery.py tests/unit/test_glyph_transformer.py (Auto output
   unchanged; this also creates the worktree .venv the codex units call through
   .venv/bin/). Read stderr: a blocked PyPI download is a blocker to report.

WAVES (each starts only after the previous wave's files exist and its tests passed;
later workers read earlier workers' files with cat, since loom map shows only the base):
  wave 1: U1, U2, U3, U4, U5
  wave 2: U6
  wave 3: U7
  wave 4: U8, then your edit E below
  wave 5: TG
Territories are DISJOINT. Workers NEVER spawn subagents. Codex rows: spawn every codex
unit of a wave BY AGENT TYPE, ALL in ONE message: loom-codex-forwarder, in the
FOREGROUND, with --model <Tier column> --effort xhigh, --unit-id <worker id>, an
explicit Bash timeout of 600000 ms, and the prompt "Your brief: <Brief path>. Read it
and doc/plans/briefs/gui-bridge-direction/bridge-direction/_shared.md in full before
anything else. Do not run git." Codex units must not run git: after EACH codex run,
check git status --short yourself and confirm only the unit's owned files changed.
Codex units cannot run uv run (no network, read-only uv cache and /tmp in codex's
sandbox): their one check is the brief's static proof through .venv/bin/, run once;
you run the real tests after each wave. Codex cannot record loom memory: record its
reported assumptions yourself. TG row: spawn one loom-software-engineer (sonnet) with
the same prompt minus the git line (the subagent preamble covers git); it runs its
brief's check once and reports.

| Worker | Role | Tier | Files owned | Shared context | Brief path |
| ------ | ---- | ---- | ----------- | -------------- | ---------- |
| F | Foundation (you, not spawned) | orchestrator | src/stencilizer/config/settings.py, src/stencilizer/core/surgery_context.py | none | FOUNDATION section above |
| U1 | Surgery direction mapping and core tests | gpt-5.6-terra | src/stencilizer/core/surgery.py, src/stencilizer/core/surgery_groups.py, tests/integration/test_bridge_direction.py | core/surgery_context.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u1-surgery-direction.md |
| U2 | Processor directions, bridge count, tests | gpt-5.6-terra | src/stencilizer/core/processor.py, tests/integration/test_processor_directions.py | core/analyzer.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u2-processor.md |
| U3 | Composite resolution and tests | gpt-5.6-terra | src/stencilizer/gui/composites.py, tests/gui/test_composites.py | io/reader.py, io/converter.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u3-composites.md |
| U4 | Grid markers, preview label, tests | gpt-6-luna | src/stencilizer/gui/glyph_grid.py, src/stencilizer/gui/glyph_view.py, tests/gui/test_grid_marks.py | gui/session.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u4-grid-and-view.md |
| U5 | Direction picker widget and tests | gpt-6-luna | src/stencilizer/gui/direction_picker.py, tests/gui/test_direction_picker.py | gui/controls.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u5-direction-picker.md |
| U6 | Session composites, directions, survey | gpt-5.6-terra | src/stencilizer/gui/session.py | gui/composites.py, core/processor.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u6-session.md |
| U7 | Controller directions and survey | gpt-5.6-terra | src/stencilizer/gui/controller.py | gui/session.py, gui/tasks.py (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u7-controller.md |
| U8 | Main window wiring | gpt-5.6-terra | src/stencilizer/gui/main_window.py | all gui modules (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/u8-main-window.md |
| E | Grid-count asserts (you, wave 4) | orchestrator | tests/gui/test_main_window.py, tests/gui/test_app.py | none | wave 4 edit below |
| TG | Session, controller, main-window tests | sonnet | tests/gui/test_session_directions.py, tests/gui/test_controller_directions.py, tests/gui/test_main_window_directions.py | all gui modules (read-only) | doc/plans/briefs/gui-bridge-direction/bridge-direction/tg-gui-tests.md |

WAVE 4 EDIT E (you, after U8 lands, before the wave's tests): the grid now lists
composites, so in tests/gui/test_main_window.py
(test_load_font_populates_window_and_selects_first_glyph) and tests/gui/test_app.py
(test_create_window_loads_font) change "assert window.grid.count() == 562" to 1027
(562 island glyphs + 465 composites, measured in _shared.md). Nothing else in those
files changes.

AFTER EACH WAVE (you): uv run ruff format <wave files> && uv run ruff check <wave
files> && uv run mypy <wave files>, then the wave's tests, with tests/gui paths in a
pytest process of their own, separate from tests/regression, tests/unit and
tests/integration (a pool forked after Qt threads start warns and can deadlock):
  wave 0: uv run pytest --no-cov -q -p no:cacheprovider tests/regression
          tests/unit/test_surgery.py tests/unit/test_glyph_transformer.py
  wave 1: uv run pytest --no-cov -q -p no:cacheprovider tests/regression
          tests/unit/test_surgery.py tests/unit/test_glyph_transformer.py
          tests/unit/test_processor.py tests/unit/test_processor_more.py
          tests/integration/test_bridge_direction.py
          tests/integration/test_processor_directions.py
          tests/integration/test_winding_preservation.py tests/integration/test_diagnostic.py
          and uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_composites.py
          tests/gui/test_grid_marks.py tests/gui/test_direction_picker.py
          tests/gui/test_glyph_grid.py tests/gui/test_glyph_view.py
  wave 2: uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_session.py
          tests/gui/test_session_open.py tests/gui/test_session_save.py
          tests/regression/test_code_structure.py
  wave 3: uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_controller.py
          tests/gui/test_controller_errors.py tests/regression/test_code_structure.py
  wave 4: uv run pytest --no-cov -q -p no:cacheprovider tests/gui
          tests/regression/test_code_structure.py
  wave 5: every acceptance command below.
Check wc -l on the new test files (tests/regression/test_code_structure.py covers src/
only).

FAILURES: a unit exiting 124 timed_out is re-split against the partial tree, never
re-forwarded as is: U1-U5 as module then tests; U6 as its step 1 then steps 2-3; U7 as
its step 1 then steps 2-3. A unit whose proof or wave tests still fail after one fix
attempt (a fresh codex unit briefed with the failure output) moves one tier up to a
Claude subagent with the brief, the failed diff and the error output: loom-software-
engineer (sonnet) for luna and terra work, loom-senior-software-engineer (opus) when the
failure is in the direction mapping (U1) or the survey lifecycle (U7). TG failing once
goes to loom-senior-software-engineer. The same worker failing twice gets a loom-advisor
diagnosis before any further attempt. If the codex CLI is unavailable at run time, the
luna and terra rows run on loom-software-engineer.

ERROR HANDLING: the existing hierarchy only (FontLoadError, FontSaveError,
GlyphNotFoundError under StencilizerError); the controller turns failures into its error
signal and the window shows them in a QMessageBox. No new exception types.

DO NOT TOUCH: src/stencilizer/cli/, src/stencilizer/io/, the bridge geometry modules
(merger*.py, bridge_*.py, multi_island*.py, horizontal_*.py, vertical_*.py,
surgery_nested.py, analyzer.py), tests/regression/, tests/unit/test_refactor_contracts.py.
Auto output must stay bit-identical (tests/regression/test_behavior_golden.py).

MEMORY: record mistakes, decisions and surprises with loom memory immediately
(subagents report theirs to you; you record them, codex units cannot). NEVER loom
knowledge in this stage; NEVER Claude Code auto-memory. A knowledge file contradicted by
the tree gets loom memory note "stale-knowledge: <file>#<heading> claims X; the tree
does Y". Record for knowledge-distill: the direction mapping table from _shared.md;
bridges_added now counts islands that left the output; composites reach the grid through
gui/composites.py because the domain reader drops components. Before loom stage
complete, record loom memory note "wiring: stencilizer.gui modules are imported by
dotted path (from stencilizer.gui.<module> import ...), which loom's unwired-file scan
does not match; composites.py is imported by session.py, direction_picker.py by
main_window.py; tests/gui and tests/integration are collected by pytest". Without a
memory mentioning wiring, the completion's unwired-file check fails.


#### Files Changed

- settings.py: BridgeDirection StrEnum + BridgeConfig.direction; surgery_context.py: SurgeryContext.direction applied in merge only when caller forced neither axis. Wave 0 regression + surgery tests green (Auto unchanged).

#### Key Decisions

- Review fixes: bridges_added uses multiset matching (duplicate islands each counted); _component_parts raises ValueError on a component cycle, which FontSession.open wraps as FontLoadError. *(adversarial review findings 1 and 2)*
- Direction mapping (BridgeConfig.direction, per glyph): single island and nested/inverted children -> AUTO uses MergeDispatch.preferred(), explicit D forces D via SurgeryContext.merge (falls back to the other axis). Island group whose arrangement equals D -> spanning always tried (ignores use_spanning_bridges), sequential if it fails; arrangement differs from D -> sequential only; AUTO -> spanning iff use_spanning. _split_child unchanged. A glyph unbuildable on both axes stays unbridged (no extra fallback). *(doc/plans/briefs/gui-bridge-direction/bridge-direction/_shared.md#direction-semantics)*

#### Notes

- found: U8 held session.display_glyphs in a local before set_glyphs, so the plan wiring regex set_glyphs\(\s*session\.display_glyphs failed loom check; orchestrator passed the property inline and selected the first glyph via session.display_names[0]. Prevention: when a plan pins wiring regexes, quote them in the unit brief as literal code shapes.
- wiring: stencilizer.gui modules are imported by dotted path (from stencilizer.gui.<module> import ...), which loom's unwired-file scan does not match; composites.py is imported by session.py, direction_picker.py by main_window.py; tests/gui and tests/integration are collected by pytest
- found: composites reach the GUI grid through stencilizer/gui/composites.py (find_bridged_composites via fontTools glyph set) because the domain reader (fonttools_glyph_to_domain) records only outline segments and drops components, so classify_glyphs files composites as empty glyphs.
- found: process_glyph's bridges_added now counts the analyzer's islands that no longer appear verbatim in the output (0 when no bridge could be placed); FontSession.unbridged and the grid's red marking rely on it.
- found: review finding (composites.py _component_parts treats a glyph mixing contours and addComponent as a leaf) left unfixed on purpose: a TrueType glyf entry is either simple or composite, never both, and CFF has no components, so the case cannot arise from a valid font.
- found: U7 copied directions via a local (directions = dict(self._directions)) so the plan's wiring regex directions=dict\(self\._directions\) did not match; orchestrator switched save to functools.partial(session.save, ..., directions=dict(self._directions)).
- mistake: plan did not foresee that the existing tests/gui/test_controller.py::test_save_uses_current_parameters stubs FontSession.save without a directions kwarg; once the controller passed directions= the stub raised TypeError on the pool and save_finished never fired (30 s timeout). Fix: stub takes **_kwargs. Prevention: when a signature gains a kwarg, rg monkeypatch stubs of that method in existing tests.
- found: codex units U2 assumed 'bbox spans centre' means strictly between bounds; U4/U3 noted loom scratch dir read-only in codex sandbox so they could not record memory (recorded here by orchestrator).
- mistake: U3 codex test helper built a component glyph with TTGlyphPen(None); pen.glyph() raises TypeError when components exist because it checks 'name in glyphSet'. Why: the codex unit cannot run tests. Prevention: pass a glyphSet mapping (dict.fromkeys(names)) to TTGlyphPen when adding components. Fixed by orchestrator.
- gotcha: loom subagents watch with only codex:<unit> workers, started right after spawning the forwarders, printed 'unknown: worker set does not resolve to one Claude parent UUID' and exited at once (tail masked the exit code). Prevention: pipe watch output without tail so the exit code shows; treat as unknown and wait for the forwarders' completion notifications.

### Integration Verification (integration-verify)

**Status:** completed  
**Purpose:** Final verification of the per-glyph direction feature and the complete glyph grid.
Verify FUNCTIONAL INTEGRATION, not only green tests. NEVER Claude Code auto-memory.
CONTEXT: read the plan (doc/plans/), the shared brief
doc/plans/briefs/gui-bridge-direction/bridge-direction/_shared.md, loom memory show
--all, and doc/loom/knowledge/architecture/gui.md plus
doc/loom/knowledge/patterns/bridge-algorithm.md.
BUILD & TEST (zero tolerance, fix every warning and failure): the two pytest commands
below (non-GUI and GUI suites in separate processes: at the base commit a single
coverage run took 264 s, close to the 300 s acceptance limit, and mixing tests/gui with
pool-forking tests printed 42 fork DeprecationWarnings; split, both halves were clean),
uv run ruff check src tests, uv run ruff format --check src tests, uv run mypy.
CODE REVIEW: spawn parallel loom-code-reviewer subagents: (1) concurrency: the survey's
BackgroundTask is referenced until its handler runs, its signals reach bound controller
methods through queued connections, a stale generation is dropped, a font load or a
parameter change invalidates in-flight results, shutdown stops the debounce timer before
waiting on the pool, the save and survey receive copies of the directions; (2) algorithm:
Auto takes exactly the old branches in process_groups and SurgeryContext.merge, explicit
directions follow the table in _shared.md, bridges_added counts islands that left the
output, composite transforms compose child-then-parent and mirrored parts keep TrueType
winding; (3) test coverage of the new code against the briefs' test lists. Fix every
finding with an engineer subagent (the reviewers are read-only).
FUNCTIONAL (prove it is wired in and usable):
- The offscreen launch of the real console script is an acceptance command below.
- Write a short throwaway script under the session scratchpad directory named in your
  system prompt (not the tree, not another temp dir: the Read tool is blocked outside
  the scratchpad). All code sits in def main(); the only top-level code is
  if __name__ == "__main__": multiprocessing.set_start_method("spawn"); main(). It builds
  the window with app.create_window for Roboto, waits for font_loaded and then
  unbridged_changed, selects O and grabs the window as a PNG under Auto, Vertical and
  Horizontal (set through window.direction_picker.combo), selects Aacute with A set to
  Horizontal and grabs again, and saves the font into the scratchpad. Open every PNG:
  the Stencilized pane must show O cut top-and-bottom for Auto/Vertical and
  left-and-right for Horizontal, Aacute cut like its A, the picker disabled for Aacute and
  reading "Follows A", and the grid item "four" in red. Reload the saved font:
  error_count 0, saved O split along the horizontal axis, saved Aacute still a composite
  referencing A. Repeat the load, the unbridged survey and one save for Lato-Black.ttf and
  CommitMono-Cosmix-700-Regular.otf.
If this stage adds files, record the same wiring memory note as bridge-direction before
completing.
Record discoveries with loom memory for knowledge-distill, including whether the fork
warnings concern still holds and any knowledge file contradicted by the tree:
loom memory note "stale-knowledge: ...".


#### Files Changed

No changes recorded.

#### Key Decisions

No decisions recorded.

#### Notes

- mistake: the review-fix engineer's new tests failed the gate on ruff (TC006 unquoted cast type, ARG001 unused 'self' in a monkeypatched method stub) and ruff format; pyright also flagged dict.fromkeys(tuple_of_literals) passed to TTGlyphPen (dict invariance). Why: subagents may run only one scoped pytest check, never lint. Prevention: briefs for test-writing subagents should name these repo lint rules (quote cast() types, prefix unused stub params with _, annotate dict.fromkeys results as dict[str, None]). Fixed by orchestrator.
- found: for the synthetic filled encircled digit (outer square, one hole, two CW bowls inside the hole; tests/unit/test_glyph_transformer.py:161) surgery_nested._process_inverted is never reached at any bridge width or direction: the hole is merged with the outer as a single island and that merge's all_contours/processed_nested bookkeeping marks both bowls processed before process_nested runs. The new tests/integration/test_bridge_direction.py::test_inverted_islands_follow_direction therefore asserts the force flags of that outer merge. Which real glyphs reach _process_inverted is unknown.
- found: concerns.md#Fork warnings when GUI and pool tests share a process no longer reproduces as written: 'uv run pytest --no-cov -q -p no:cacheprovider tests' in ONE process (Python 3.13.1, 313 tests, 224 s) printed 0 'use of fork()' DeprecationWarnings and no warnings summary; a short mix (tests/gui/test_controller.py then tests/integration/test_processor_directions.py) also printed none. Not re-measured with coverage on (the 42 at 111d115 were with coverage). The split gate stays harmless; the concern should say 'not reproduced without coverage at the integration-verify commit'.
- stale-knowledge: architecture/gui.md#GUI package layout lists session.py as 'island list' and omits gui/composites.py (Qt-free composite discovery/compose) and gui/direction_picker.py; #Threading model cites controller.py:65/102/135 for queued connections and busy guards, which are now controller.py:78-86 (_start), 119 and 182, and does not mention the debounced unbridged survey (SURVEY_DELAY_MS=250, own BackgroundTask kept in _survey_task, generation counter, never sets busy). Correction: add both modules to the table, update line refs, add a survey paragraph.
- found: integration review fixed in this stage: (1) _on_survey_failed neither drained _survey_pending nor dropped failures of superseded surveys; (2) composites._transform_contour flipped the direction label on mirrored parts although points.reverse() already restores the original winding, so label contradicted geometry; (3) _component_parts had no depth/part bound, so a font with a doubling component DAG (2^n parts) hung FontSession.open; (4) load_component_outlines skipped a missing base and compose then failed with a bare KeyError. Prevention: briefs for failure handlers should say 'mirror the success path's pending/generation handling'; untrusted recursive structures need an explicit bound.
- found: functional run through app.create_window (offscreen, spawn) on Roboto/Lato/CommitMono matched the plan: Roboto 1027 display glyphs, 14 unbridged; O cut top+bottom under Auto/Vertical and left+right under Horizontal; Aacute follows A with picker disabled reading 'Follows A'; four red with 'no bridge could be placed'; saved O has 4 contours none spanning y=728; saved Aacute still composite [A, acute]; Lato 817/32 unbridged, CommitMono 467/0; all saves error_count 0. Driver: all code in def main(), QEventLoop-based wait_for(signal) helper, set direction via picker.combo.setCurrentIndex(findData(value)).
- concern: CommitMono .notdef (frame + hole + inverted '404' digits) stencils under Auto into overlapping full-width outer rectangles, so preview and saved glyph render as a solid black box. Output is byte-identical to base 9ff48c2 (sha256 prefix c1f9a7e72f56d125, 20 contours); only bridges_added changed 5->3. Pre-existing Auto defect in the inverted-island path; out of this plan's scope (Auto pinned). Repro: process_glyph on CommitMono .notdef with BridgeConfig().

## Open Questions

No open questions.

