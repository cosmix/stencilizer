# Code Review: Proportional bridge width for variable fonts

**Plan:** PLAN-variable-bridge-width | **Generated:** 2026-10-09 23:50 UTC

## Summary

## Overview

## Changes by Stage

### Variable bridge width scaling (width-scaling)

**Status:** completed  
**Purpose:** Add width scaling for variable-font bridges. Plan: doc/plans/IN_PROGRESS-PLAN-variable-bridge-width.md
(Goals, Design decisions, Engine rules, Evidence, Sandbox and environment inventory). Shared
worker context: doc/plans/briefs/variable-bridge-width/width-scaling/_shared.md. Where a brief
and the plan disagree, the plan wins. Ignore any doc/plans/codex-PLAN-* file.
Use parallel subagents and skills to maximize performance.

PUBLIC SURFACE (the contract session writes the contracts from this; implement it exactly):
  stencilizer.config.BridgeWidthScaling(StrEnum): FIXED = "fixed", PROPORTIONAL = "proportional".
  BridgeConfig new fields: width_scaling: BridgeWidthScaling = FIXED;
    scaling_strength: float = 100.0 (ge 0, le 100); min_width_percent: float = 30.0 (ge 10, le 110).
  stencilizer.variable.transform.transform_variable_glyph(vg, bridge, geometry, upm) keeps its
    signature. Per master, each bridge (a DISJOINT pair of bridge lines, pairing rule 1 of the
    plan's Engine rules) gets gap = base in fixed mode and
    gap = max(minimum, base * ratio ** (scaling_strength / 100)) in proportional mode, where
    base is the distance between the pair's two default line coordinates, ratio is the
    master's ink along the pair's centre line through the merged master polygon over the same
    in the default (Engine rules 2-4), and minimum = min(min_width_percent / 100 * 0.1 * upm, base).
    The lower line goes to centre - gap / 2 and the upper to centre + gap / 2, centre being the
    mean of the two lines' fixed-parameter targets. Lines in no pair keep today's mean target.
    align_to_lines keeps today's mean targets, so the default master is unchanged.
    Fallback, per glyph: configured mode, then fixed, then today's per-line mean targets with
    no pairing; only when all fail is the glyph left unchanged. Fixed mode starts at step 2.
  stencilizer.variable.bridge_width.scaled_gap(base: float, ratio: float, bridge: BridgeConfig, upm: int) -> float.
  stencilizer.variable.replay.replay(smap, input_default, output_default, master_input) keeps four
    positional parameters and returns today's mean-target result when smap carries no width
    rule; the pairs and gap rule ride on a new optional SurgeryMap field (default None).
  CLI (stencilizer.cli.app.app): --width-scaling (fixed|proportional, default fixed),
    --scaling-strength FLOAT (0-100, default 100), --min-bridge-width FLOAT (10-110, default 30);
    all three reach BridgeConfig(width_scaling=..., scaling_strength=..., min_width_percent=...).
    With --width-scaling proportional, before anything is pinned or processed:
      a static input (is_variable_font(input_font) is False) prints exactly
      "Width scaling applies only to variable fonts; using fixed width." (unless --quiet) and
      the settings switch to fixed;
      a CFF2 variable input with --instance prints exactly
      "Proportional width scaling with --instance is not supported for CFF2 fonts; pinning first with fixed width."
      (unless --quiet) and the settings switch to fixed.
    The checks read input_font, never the pinned temporary file; a glyf variable input with
    --instance and --dry-run or --list-islands prints neither line.
    Write path for proportional + --instance + a glyf variable input: validate_instance first;
    stencil the variable input into tempfile.TemporaryDirectory(prefix="stencilizer-instance-");
    pinned = pin_stenciled(stenciled, input_font, instance, tmp); then the static second pass:
    classify pinned with fixed settings; when any glyph has islands, FontProcessor processes
    them from pinned into output_path; when none has, publish_pinned(pinned, output_path).
    One success report names output_path, with bridges = the sum of both passes and unbridged
    = the second pass's count (the islands left in the written font). --list-islands and
    --dry-run pin first in every mode; fixed mode keeps today's flow for every command.
    Dry run prints after the bridge-width line "  Width scaling         fixed" or
    "  Width scaling         proportional (strength {s}%, minimum {m}% of a reference stroke)",
    s and m being settings.bridge's floats (e.g. "strength 40.0%, minimum 45.0%").
  stencilizer.cli.pinning (U1): pinned_input(input_font, instance) (today's app._instance_font,
    moved); validate_instance(input_font, instance) -> None;
    pin_stenciled(stenciled, source, instance, workdir) -> Path; publish_pinned(pinned, output_path) -> Path;
    is_variable_font(path) -> bool; is_cff2_font(path) -> bool.
    publish_pinned stages in tempfile.TemporaryDirectory(dir=output_path.parent) and replaces
    the exact output_path with Path.replace (never shutil.move): an existing file is replaced;
    a directory output_path raises FontSaveError and is left untouched; an OSError from the
    replace raises FontSaveError, keeps the previous output byte for byte and leaves no staged
    file. The CLI prints its success report only after publish_pinned or FontProcessor returns.
  GUI (stencilizer.gui.controls.ControlPanel): scaling_box (QWidget holding every width-scaling
    widget, hidden until set_variable(True)), scaling_combo (QComboBox, items "Fixed",
    "Proportional" with item data BridgeWidthScaling.FIXED / PROPORTIONAL), strength_slider +
    strength_spin (0-100, default 100), min_width_slider + min_width_spin (10-110, default 30),
    set_variable(variable: bool): shows or hides scaling_box; set_variable(False) also selects
    Fixed when Proportional is selected (one parameters_changed). bridge_config() returns the
    three new fields, the mode as BridgeWidthScaling(scaling_combo.currentData()).
    MainWindow._on_font_loaded calls self.controls.set_variable(bool(session.axes)) before the
    glyph selection. At 1280x800 with Inter loaded no control is squeezed below its minimum
    height (a QScrollArea around the controls, or rows that fit).

CONTRACT GEOMETRY (tests/unit/test_width_scaling_contracts.py): build glyphs with
tests.unit._variable_cases.glyph_from_outlines (it returns a Glyph) and wrap them in
stencilizer.variable.model.VariableGlyph(default, supports, masters, ("wght",)). Outer square (0,0)-(1000,1000) clockwise,
as _variable_cases.STEM winds; counter square counter-clockwise; UPM 1000; supports
Support((("wght", 0.0, 1.0, 1.0),)) for bold and Support((("wght", -1.0, -1.0, 0.0),)) for thin;
BridgeConfig(direction=BridgeDirection.VERTICAL, width_percent=60) plus the scaling fields under
test; GeometryConfig(). Import BridgeDirection from stencilizer.config.settings: the
stencilizer.config package does not export it (and this plan does not add it).
Measure the gap in out.glyph.instance(location): the distinct point x
values strictly inside the counter's x range at that location (exclusive by 0.5 unit);
gap = max - min of those. Tolerance 1.0 unit (integer rounding).
Ring: default counter 200..800 on both axes, bold master counter 300..700, thin master counter
50..950. Narrow: default counter 200..800, one bold master with the counter at x 465..535 and
y 300..700. Mean-only: the same with the bold counter at x 475..525, y 300..700.
Two-counter (vertical): outer (0,0)-(1600,1000), counters (200,200)-(600,800) and
(1000,200)-(1400,800); the bold master keeps the outer and the second counter and moves the
first counter to (200,300)-(600,700); gaps are measured inside x 200..600 and 1000..1400.
Two-counter (horizontal): direction HORIZONTAL, outer (0,0)-(1000,1600), counters
(200,200)-(800,600) and (200,1000)-(800,1400); the bold master moves the first counter to
(300,200)-(700,600); gaps are the distinct point y values strictly inside y 200..600 and
1000..1400. Wide-master: the ring's default with one bold master whose outer is
(0,0)-(6000,1000) and counter (2980,50)-(3020,950); its gap is measured inside x 2980..3020.
CLI contracts use typer.testing.CliRunner on stencilizer.cli.app.app with -o and --log-file
under tmp_path (pattern: tests/unit/test_variable_surface_contracts.py _cli), on
tests.font_helpers.INTER, CANTARELL and ROBOTO; outlines are compared with
tests.font_helpers.glyph_at(...).to_dict() at a normalized location ({} for default,
{"wght": 1.0} for Black, {"wght": -1.0} for Thin); island counts use tests.font_helpers.island_count.
GUI contracts (tests/gui/test_width_scaling_gui_contracts.py) copy the window fixture and
_load_window helper from tests/gui/test_variable_gui_contracts.py (file-local there) and use
pytest-qt qtbot.

EXECUTION. Territories below are DISJOINT. Workers NEVER spawn subagents. First spawn F0
alone and wait for it. Then spawn W1, W2, W3 BY AGENT TYPE in the background and U1 as a
loom-codex-forwarder in the FOREGROUND (--model gpt-5.6-terra --effort xhigh, Bash timeout
600000 ms), ALL in ONE message, each with the fixed prompt plus
"Your brief: <path>. Read it in full before anything else." Codex units never run git and
never touch .loom/; check git status --short after the forward, then run
uv run pytest tests/unit/test_cli_pinning.py --no-cov -q -p no:cacheprovider yourself.
W1 is opus at effort xhigh (loom-senior-software-engineer): core algorithmic work.
Every worker runs uv run pytest tests/regression/test_code_structure.py --no-cov -q
-p no:cacheprovider before reporting (functions over 50 effective lines fail it; stencilize
has 2 lines to spare and _run_command 5).

| Worker | Role | Tier | Files owned | Shared context | Brief path |
| ------ | ---- | ---- | ----------- | -------------- | ---------- |
| F0 | Settings foundation | haiku | src/stencilizer/config/settings.py, src/stencilizer/config/__init__.py | _shared.md | doc/plans/briefs/variable-bridge-width/width-scaling/f0-settings.md |
| W1 | Replay engine | opus/xhigh | src/stencilizer/variable/bridge_width.py, src/stencilizer/variable/realign.py, src/stencilizer/variable/replay.py, src/stencilizer/variable/transform.py, src/stencilizer/variable/validate.py, tests/unit/test_width_scaling.py, tests/unit/test_variable_replay.py, tests/unit/test_variable_transform.py, tests/unit/test_variable_validate.py | _shared.md | doc/plans/briefs/variable-bridge-width/width-scaling/w1-engine.md |
| W2 | GUI controls | sonnet | src/stencilizer/gui/controls.py, src/stencilizer/gui/main_window.py, tests/gui/test_width_scaling_controls.py, tests/gui/test_controls.py | _shared.md | doc/plans/briefs/variable-bridge-width/width-scaling/w2-gui.md |
| W3 | CLI wiring | sonnet | src/stencilizer/cli/app.py, src/stencilizer/cli/handlers.py, tests/unit/test_cli_width_scaling.py, tests/unit/test_cli_variable.py | _shared.md | doc/plans/briefs/variable-bridge-width/width-scaling/w3-cli.md |
| U1 | cli/pinning.py | codex terra | src/stencilizer/cli/pinning.py, tests/unit/test_cli_pinning.py | _shared.md | doc/plans/briefs/variable-bridge-width/width-scaling/u1-pinning.md |

SANCTIONED TEST CHANGE (the one expected test-integrity event): in
tests/unit/test_variable_transform.py test_replay_failing_in_the_last_master_writes_no_replayed_master,
the stub fails the last master on every attempt
(last = len(results) % len(vg.masters) == len(vg.masters) - 1), and the assertions become
len(results) == 2 * len(vg.masters) and all results except each attempt's last are not None;
_assert_unchanged stays. BridgeConfig() runs two attempts (fixed, then mean). File
loom stage dispute-integrity for that event with this reason; any other test-integrity event
is reverted. tests/unit/test_variable_validate.py stays unchanged because _bridged stays one
attempt with its 5-argument signature and calls transform.replay with 4 positional arguments.

REQUIRED W1 TESTS (new tests only; the acceptance list runs each by node id, so a missing
one fails the gate):
  tests/unit/test_variable_transform.py::test_replay_failing_once_retries_next_step: the
    sanctioned test's glyph and stub, failing the last master on the FIRST attempt only;
    the outcome has bridge_count >= 1, outcome.glyph is not the input, and
    len(results) == 2 * len(vg.masters) (the retry succeeded).
  tests/unit/test_variable_transform.py::test_validation_failure_retries_next_step: wraps
    transform.validate so its first call returns False and later calls delegate to the real
    one; transform_variable_glyph on the same glyph bridges it and validate ran twice.
  tests/unit/test_width_scaling.py::test_fixture_glyphs_keep_baseline_outcome: parametrized
    over the three fixtures and both modes (BridgeConfig() and
    BridgeConfig(width_scaling=PROPORTIONAL)); embeds the plan's per-glyph Baseline table and
    asserts, per island glyph of variable_island_counts, the same bridge_count and
    unbridged_count as the table, and that every glyph bridged in fixed mode is bridged in
    proportional mode.
  tests/unit/test_width_scaling.py::test_spanning_pairs_share_contour_set: Inter eight with
    BridgeConfig(use_spanning_bridges=True, direction=BridgeDirection.HORIZONTAL): pairing
    returns disjoint pairs whose two lines have the same input-contour set, and in fixed
    mode each pair's gap equals its default gap +/- 1 in every master. If the default
    surgery of that glyph yields no spanning bridge, W1 records the measured line count with
    loom memory note and keeps the assertion on whatever pairs it yields.

GATE INTERPRETER: the provisioned .venv (CPython 3.13, CI's newer matrix entry); the 3.11
run is integration-verify's. Gates run pytest with --numprocesses=16 --no-cov, a deliberate
deviation from CI's coverage run (coverage has no threshold and triples the xdist time); the
test set is the full configured tree, never a subset.

GATE: one loom-verifier round after every worker returned, with the acceptance list below.
W1 reports, per fixture (Inter, Ubuntu, Cantarell) and mode, the glyphs bridged and how many
went down each fallback step, against the plan's Baseline counts (20 of 21 per fixture), plus
the Inter o gap at Thin, default and Black and the worst cold preview time that
tests/gui/test_variable_session.py::test_cold_preview_time_per_island_glyph measures. Record
those numbers with loom memory note. If that timing test fails under xdist only, rerun it
serially once and record both results (concerns.md records it as flaky under load).
The static path (core/, io/) and tests/regression goldens must stay unchanged.
MEMORY: record mistakes/decisions/surprises via loom memory immediately; NEVER loom
knowledge (implementation stage); NEVER Claude Code auto-memory.


#### Files Changed

No changes recorded.

#### Key Decisions

- realign.py is the bottom module (contour_spans, axis_value, lerp, mean_target, realign_line, project, _crossing); bridge_width.py imports it at runtime and replay.py types only under TYPE_CHECKING; replay.py imports both. Over keeping the helpers in replay.py and importing bridge_width lazily. *(replay() must reach scaled_gap through plain calls (the plan's reachable check) while bridge_width's pairing needs contour spans: any runtime import of replay.py from realign or bridge_width makes a cycle. _project no longer isinstance-checks Vertex: replay._stops passes line points and EdgePoint slots as stops, which is the same walk.)*

#### Notes

- mistake: first rewrap of transform_variable_glyph's docstring split only the line before the long one, leaving a 120-char line; review round 2 caught it. Why: edited without re-reading the lines after the edit. Prevention: after rewrapping prose, check line lengths with rg '^.{101,}'.
- measured (width-scaling W1, auto direction, island glyphs bridged / fallback step): Inter fixed 20/21 (19 fixed, e mean, ampersand unchanged), proportional 20/21 (18 proportional, a fixed, e mean); Ubuntu fixed 20/21 (20 fixed), proportional 20/21 (15 proportional, 5 fixed: a e o zero eight); Cantarell fixed 20/21 (20 fixed), proportional 20/21 (18 proportional, 2 fixed: a e). Horizontal: Inter 21 (fixed 19+2 mean: e, ampersand), Ubuntu 19 (ampersand, B unchanged), Cantarell 21 (fixed all; proportional 17 + 4 fixed). Baseline per-glyph bridge/unbridged counts equal HEAD in both modes on all 3 fixtures. Inter o gap Thin/default/Black: HEAD 117/123/109, fixed 123/123/123 (opsz 122), proportional 61/123/231 (opsz+Black 241); vertical ratios 0.270 Thin, 1.881 Black. Worst cold preview (test_cold_preview_time_per_island_glyph, Inter, BridgeConfig()): 23.3 ms before, 24.2 ms after.
- stale-knowledge: concerns.md#Variable fonts: known gaps claims replay._crossing searches up to 12 edges; the tree moved it to realign.py (_crossing, _SEARCH_EDGES = 12). Correction: name realign._crossing in the Replay edge cases bullet.
- stale-knowledge: concerns.md#Variable bridge gaps drift across masters claims replay drifts gaps (Inter o 117/123/109); the tree now holds them: fixed mode gives Inter o 123 at Thin, default and Black (122 at the opsz master), proportional 61/123/231 (opsz+Black 241). Only the third fallback step (per-line mean targets, transform._bridged step=None) still drifts. Correction: replace the concern with the whole-glyph fallback and unreported-fallback concern the plan names.
- stale-knowledge: patterns/variable-replay.md#Bridge-line grouping and realignment claims _realign (replay.py:361) sets each line to the mean of its cuts, _crossing/_SEARCH_EDGES live in replay.py, and nothing ties the two lines of one bridge; the tree now has realign.py (mean_target, realign_line, project, _crossing, _SEARCH_EDGES=12, contour_spans, axis_value, lerp), replay() computes every line's mean target up front, and when SurgeryMap.widths (bridge_width.WidthRule) is set, bridge_width.pair_targets moves each disjoint pair's lines to centre -/+ scaled_gap/2 (centre = mean of the two lines' mean targets, ink clipped to both lines' placed cross range). Correction: rewrite the section to that pipeline; transform._first_bridged runs bridge_width.fallback_steps (configured mode, fixed, None=mean) with _bridged(step=...) as one attempt; align_to_lines still uses the widthless map.
- found: Inter eight with BridgeConfig(use_spanning_bridges=True, direction=HORIZONTAL) yields NO spanning bridge: 2 horizontal bridges, 4 lines (y 355.56/478.44 on contours {0,143}, 1035.56/1158.44 on {0,216}), 2 disjoint pairs of 122.88. The spanning bridge (one vertical pair x 571.06/693.94 on {0,143,216}) appears only with direction AUTO; test_spanning_pairs_share_contour_set covers both directions.
- gotcha: loom subagents watch exited 5 ('worker set does not resolve to one Claude parent UUID') when a codex:<unit-id> worker was named before the forwarder had started the wrapper; watching the forwarder's own claude:<agent-id> binds.
- surprise: today's pin-first Inter --instance wght=550 (fixed) leaves glyph a with enclosed_counters 1 (every other glyph 0, ampersand 0). The contract cli-instance-proportional-open-off-master expects 0 for a, which matches the plan's stencil-then-pin evidence (line 138), so a proportional --instance run that still pins first fails that contract too.

### Integration Verification (integration-verify)

**Status:** completed  
**Purpose:** Final verification after all stages. Verify FUNCTIONAL INTEGRATION, not just tests
passing. NEVER Claude Code auto-memory.
CONTEXT: read doc/plans/IN_PROGRESS-PLAN-variable-bridge-width.md (Goals, Design decisions,
Engine rules, Baseline, Sandbox and environment inventory), loom memory show --all, and the
knowledge sections patterns/variable-replay.md "Bridge-line grouping and realignment" and
"Validation".
BUILD & TEST (zero tolerance; fix ALL warnings/errors): the acceptance list below. The full
suite runs under pytest-xdist with 16 workers.
CODE REVIEW: spawn parallel loom-code-reviewer subagents: (1) correctness of
src/stencilizer/variable/bridge_width.py, realign.py, replay.py and transform.py against the
gap formula, the pairing rule (disjoint, same contour set), the ink rule (even-odd from the
first crossing, clipped to the master's extent), the zero-ink rule and the fallback order in
the plan; (2) CLI/GUI wiring, the two warnings, the stencil-first --instance branch (spec
validated first, source names swapped in, static second pass, publish_pinned staging in the
output directory, temp directory cleanup on error and Ctrl-C, one report naming the output
path) and set_variable(False) resetting to Fixed; (3) architecture, size limits and test
coverage. Fix every finding with an engineer agent or dispute it; never defer one.
SUGGESTIONS: weigh every pending reviewer suggestion the signal lists; resolve each one
implemented with loom memory resolve <id> --outcome implemented --reason <what changed>.
FUNCTIONAL (scratch files under the session scratchpad directory; never the worktree):
1. Run the CLI on each of tests/fixtures/variable/Inter-VF-subset.ttf, Ubuntu-VF-subset.ttf
   and Cantarell-VF-subset.otf in fixed and in proportional mode. Reopen each output with
   fontTools: fvar kept, and each run's report shows at least the Baseline bridges (Inter 28,
   Ubuntu 22, Cantarell 22) with no more unbridged islands (2 each). These totals are a
   summary: confirm tests/unit/test_width_scaling.py::test_fixture_glyphs_keep_baseline_outcome
   ran and passed in the suite (it holds the per-glyph guarantee).
2. Run --instance wght=900 --width-scaling proportional on Inter: a static output, no island
   in o or ampersand (tests.font_helpers.island_count), name ID 1 "Inter Variable Text Black
   Stenciled". Run --instance wght=250 --width-scaling proportional on Cantarell: the CFF2
   warning prints and the output matches the fixed --instance wght=250 run glyph for glyph.
3. Render HEAD-vs-new outlines (matplotlib or QPainter, scratchpad PNGs, Read them) for every
   glyph whose non-default masters changed in fixed mode on the three fixtures, at the default
   and at every master peak: bridges keep the default width, nothing folds or closes.
The GUI path is covered by tests/gui/test_width_scaling_controls.py (W2's save test), which
the full suite runs. Record the per-fixture counts with loom memory note.
Record discoveries to loom memory for knowledge-distill, including any knowledge file
contradicted by the tree: loom memory note "stale-knowledge: ...".
If loom stage complete refuses from the sandbox for the fingerprint reason in the plan's
Sandbox and environment inventory, block the stage with that reason for the operator.


#### Files Changed

No changes recorded.

#### Key Decisions

- stencil-first report time is wall clock from before pass 1 to after pass 2 (start_time/end_time on the merged ProcessingStats), not a sum of pass durations *(pass 2 may only publish (empty ProcessingStats, no timestamps); wall clock covers pin and publish too and works with that case)*
- Left four width-scaling suggestions pending for knowledge-distill: 78f07650 (hoist surgery out of the fallback loop; the plan keeps _bridged one attempt and tests call it directly; measured cost at most 15% per glyph), 5664a2ef (zip strict=True in ink is an invariant: the half-open rule gives an even crossing count on closed polygons), cfb04198 (no output file when pass 1 finds no islands matches every other no-islands run), 6790fa9c (the dry-run line reports the configured mode, which the real glyf run uses; CFF2 is switched to fixed before the dry run). Also left: moving Slot/LineMember/BridgeLine into realign.py and moving _stencil/_process_font out of app.py, because tests patch those names by module path. *(integration-verify weighs every pending suggestion; implemented 21bc51ca, 5a06a8b2, 430977c4)*

#### Notes

- mistake: piped loom stage commit through rg -o to shorten output; the relay hook never saw the LOOM_RELAY_V1 line, the ticket (5c25db62, test(variable) commit) stayed unrelayed, its staged file rode along in the next commit, and a soft reset plus two fresh commits split it back out. Why: treated the relay line as noise. Prevention: run every loom write command (stage commit, memory note/decision/resolve) with stdout unfiltered; filter only stderr WARN lines with rg -v. Orphan ticket 5c25db62 was never applied; if it is ever recovered it finds an empty index.
- found/gotcha: on Inter proportional --instance the real pass 1 bridges every island, so pass 2 classifies nothing and only publishes; tests needing a real pass-2 glyph failure must fake pass 1 with a copy of the unstenciled font (tests/unit/test_cli_width_scaling.py)
- mistake: the integration-verify GUI fix agent ran git stash push/apply in the stage worktree to prove its tests fail on the old code while two other engineers were editing cli/ and variable/ in the same worktree. Why: the brief named file ownership but did not forbid git stash, and the subagent preamble's git restrictions did not stop it. Prevention: every brief for parallel engineers in one worktree says: never git stash/checkout/restore; prove a test fails on old code by editing the one line and restoring it by hand. Fix: orchestrator compared each agent's reported changes with git diff before the gate.
- measured (integration-verify functional smoke at 8c93e55): CLI report glyphs/bridges/unbridged, fixed and proportional alike: Inter 21/28/2, Ubuntu 21/22/2, Cantarell 21/22/2 (ampersand unbridged; fvar kept). Fixed vs proportional defaults identical; non-default peaks differ for 17/14/18 glyphs; proportional equals fixed at every peak for Inter a, Ubuntu a e eight o zero, Cantarell a e (fixed fallback). HEAD 3efd97c vs new fixed: defaults identical for the 20 glyphs both bridge; masters changed for 18/19/20 glyphs; new gaps Inter 122-123, Ubuntu 60, Cantarell 60 at every peak (HEAD Inter o 117/123/109, opsz+Black 103); 0 islands and 0 enclosed counters at every peak except ampersand. --instance wght=900 proportional on Inter: static, 30 bridges (pass 2 bridges ampersand), name ID 1 'Inter Variable Text Black Stenciled'. Cantarell --instance wght=250 proportional prints the CFF2 warning and equals the fixed run in all 24 glyphs. test_fixture_glyphs_keep_baseline_outcome ran 6 params, all passed.
- found/gotcha: integration-verify functional smoke (variable-bridge-width): fixed mode gaps constant per glyph at every master (Inter 122-123, Ubuntu/Cantarell 60); in proportional mode glyphs Inter a; Ubuntu a,e,eight,o,zero; Cantarell a,e render identical to fixed at all peaks (fallback to fixed gap). CFF2 --instance outputs carry cid000NN glyph names. The CFF2 proportional-instance warning wraps at 80 cols in non-tty output, so raw substring matches need whitespace flattening.
- stale-knowledge: concerns.md#Source files at the size limit claims variable/replay.py is 397 of 400 lines; the tree has 327 (src/stencilizer/variable/replay.py; helpers moved to variable/realign.py), while cli/app.py is 385/400 with _run_command at 49/50 effective lines (src/stencilizer/cli/app.py:184). Correction: drop replay.py from the near-limit list; name cli/app.py and its _run_command.

## Open Questions

No open questions.

