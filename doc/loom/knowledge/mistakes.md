# Mistakes & Lessons Learned

> Record mistakes made during development and how to avoid them.
> This file is append-only - agents add discoveries, never delete.
>
> Format: Describe what went wrong, why, and how to avoid it next time.

(Add mistakes and lessons as you encounter them)

## Stale root CLAUDE.md claims

**What happened**: The root CLAUDE.md states TrueType is "CCW=outer, CW=inner" (CLAUDE.md:52) and lists a `_update_cff2_glyph()` writer with static CFF2 support (CLAUDE.md:36).
**Why**: Docs written ahead of, or inverted from, the implementation; no CFF2 writer was ever implemented.
**Prevention**: Trust code over CLAUDE.md for winding and format support: TrueType is CW outer / CCW hole (src/stencilizer/core/analyzer.py:122-135); only `glyf` and `CFF ` are writable (src/stencilizer/io/converter.py:89-95).
**Fix**: Correct CLAUDE.md when next editing it (knowledge bootstrap may not touch it).

## Holes filled on nested-contour glyphs

**What happened**: Stencilizing glyphs with nested contours (®, ©, ℗, @, φ, &, ß, Θ) could fill inner holes.
**Why**: Bridge splitting reversed or merged hole contours so they no longer wound CCW.
**Prevention**: After any surgery change, run tests/integration/test_winding_preservation.py (module docstring, lines 1-6) and tests/integration/test_diagnostic.py against Lato-Black.
**Fix**: Regression coverage in those modules.

## Missing bridges in encircled digits and Θ-like glyphs

**What happened**: Filled encircled digits (⑧) lost bridges on "inverted islands" (CW bowls inside a CCW hole); Θ-like glyphs had their structural crossbar treated as an obstruction.
**Why**: Logic assumed two winding levels and treated every spanning bar as blocking.
**Prevention**: Handle three-level nesting and same-winding structural bars; see [patterns/bridge-algorithm](patterns/bridge-algorithm.md).
**Fix**: tests/unit/test_surgery.py:126-266 (structural bars) and :419-470 (inverted islands).

## Codex forwarder spawned outside a loom stage

**What happened**: Four loom-codex-forwarder spawns failed with exit 2 (missing --invocation-id, then 'LOOM_STAGE_ID and LOOM_SESSION_ID are required') in an interactive session with no loom stage.
**Why**: codex-forward.sh builds its companion session id from the stage and session env vars; the guard only injects the invocation id inside a stage.
**Prevention**: Outside a loom stage, route codex work through the codex:codex-rescue plugin agent with --model/--effort/--write; use loom-codex-forwarder only inside a stage.
**Fix**: Re-dispatched the units via codex:codex-rescue.

## Fixed concerns marked "Resolved" instead of deleted

**What happened**: After the 2026-09 refactor, concerns.md entries were rewritten as "Resolved ..." and a history note ("Replaced: ... used to live") went into patterns/bridge-algorithm.md; the user corrected this.
**Why**: The loom template header "append-only - never delete" was on every knowledge file and was taken at face value.
**Prevention**: Only mistakes.md is append-only (conventions.md "Knowledge files hold current state"). Delete fixed concerns; write current facts, not change history.
**Fix**: concerns.md rewritten to extant issues; headers of the other tier-1 files corrected.

## loom knowledge update run from the knowledge directory

**What happened**: Running `loom knowledge update conventions` with cwd doc/loom/knowledge scaffolded a second knowledge base at doc/loom/knowledge/doc/loom/knowledge and wrote the entry there.
**Why**: loom resolves the knowledge root relative to the working directory.
**Prevention**: Run loom knowledge commands from the repository root (`cd <repo> && loom knowledge ...`).
**Fix**: Deleted the nested tree and re-ran from the root.

## Codex worker briefs told to verify with uv run

**What happened**: The GUI plan's worker briefs (doc/plans/briefs/stencilizer-gui/) had every codex unit run `uv run pytest/mypy/ruff` as its proof command; the 2026-09-23 pressure test found none of them could run.
**Why**: The codex companion runs write jobs in codex's workspace-write sandbox: no network, a read-only ~/.cache/uv, and `exclude_slash_tmp = true` in ~/.codex/config.toml, so `uv run` fails and pytest's tmp_path is unwritable. The codex preamble (codex-forward.sh) also forbids verification.
**Prevention**: A codex unit's single check calls the worktree venv directly and stays static: `.venv/bin/mypy <files> && .venv/bin/ruff check <files> && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only <test>`. The orchestrator runs the real tests with `uv run` after each wave. The stage's FOUNDATION step must create .venv first.
**Fix**: Briefs and plan amended in the pressure pass.

## Codex unit proof command misses the function-length limit

**What happened**: A codex-written `ControlPanel.__init__` came out at 59 effective lines and failed `tests/regression/test_code_structure.py::test_function_line_limit`.
**Why**: The static proof command (mypy, ruff, collect-only) does not run that test.
**Prevention**: Include `tests/regression/test_code_structure.py` in every wave's orchestrator pytest run. It covers `src/` only: check `wc -l` on test files by hand (see concerns.md).
**Fix**: Split the constructor.

## Path-based writer aimed at a directory others can write

**What happened**: The first GUI save fix had `FontProcessor` write to an `O_EXCL` temp sibling in the output directory and closed the descriptor. `FontWriter.save` reopens the path with `open(path, 'wb')` after the whole glyph run, so a local user with write access to that directory could swap the file for a symlink to the input.
**Why**: `O_EXCL` protects creation only, not a later path-based reopen.
**Prevention**: Stage in a private `mkdtemp` directory and publish into the destination through the descriptor `O_EXCL` returned, then rename. Found by the adversarial review.
**Fix**: `FontSession.save` now stages privately (architecture/gui.md).

## Existence check ordered before a resolve()-based check

**What happened**: A new `output_path.parent.is_dir()` check placed before the overwrite-input check failed `test_save_refuses_input_path`, which passes `tmp_path/'sub'/'..'/name` with `sub` absent.
**Why**: `Path.is_dir()` stats through the missing `sub`; `Path.resolve()` normalizes `..` without requiring it to exist.
**Prevention**: Put stat-based checks (`is_dir`, `exists`) after a `resolve()`-based check.
**Fix**: `save` checks the input, then the resolve()/samefile overwrite guard, then `parent.is_dir()`.

## Codex units and loom tooling inside the stage sandbox

**What happened**: Codex units could not record memories (loom scratch dir read-only in codex's sandbox), and `loom subagents watch` from the sandboxed Bash tool exited 3 ("process is gone") seconds after a codex forward started. A watch bound only to `codex:<unit>` workers right after the forwarders spawned printed "unknown: worker set does not resolve to one Claude parent UUID" and exited at once; piping it through `tail` hid the exit code.
**Why**: Codex runs in its own workspace-write sandbox; each Bash call gets its own PID namespace, so the watch cannot see codex's pid.
**Prevention**: The orchestrator records codex assumptions itself. Treat those watch exits as unknown, run the watch without a pipe so the exit code shows, and wait for the forwarder's own completion.

## Signature change breaks monkeypatched stubs in existing tests

**What happened**: The controller began passing `directions=` to `FontSession.save`; `tests/gui/test_controller.py::test_save_uses_current_parameters` stubbed `save` without that kwarg, the stub raised `TypeError` on the pool, `save_finished` never fired, and the test hit its 30 s timeout.
**Why**: The plan listed the production callers of `save` but not the test doubles.
**Prevention**: When a method gains a kwarg, `rg` for monkeypatch stubs of it in existing tests and make them accept `**_kwargs` in the same unit.
**Fix**: The stub takes `**_kwargs` (tests/gui/test_controller.py:282).

## Error handlers that skip the success path's bookkeeping, and unbounded recursion on font data

**What happened**: Integration review found four defects in the direction work: `_on_survey_failed` neither drained `_survey_pending` nor dropped failures of superseded surveys; `composites._transform_contour` flipped the direction label on mirrored parts although `points.reverse()` already restores the winding; `_component_parts` had no depth or part bound, so a font whose components form a doubling DAG (2^n parts) hung `FontSession.open`; `load_component_outlines` skipped a missing base and `compose` then failed with a bare `KeyError`.
**Why**: The failure handler was written without the success handler beside it; the recursive walk trusted the font.
**Prevention**: Brief every failure handler as "mirror the success path's pending and generation handling". Give any recursive walk over untrusted font structures (components, nested contours) an explicit depth and size bound plus a cycle check. Check both a flipped transform and its label, not the label alone.
**Fix**: Handlers mirrored, `_component_parts` bounded and raising `ValueError` (wrapped as `FontLoadError`), missing bases reported by name.

## Test-writing units fail the repo's lint gate and hide test-helper traps

**What happened**: Codex and sonnet units may run only one static check, so their tests reached the gate with ruff `TC006` (unquoted `cast` type), `ARG001` (unused `self` in a monkeypatched stub), `ruff format` diffs, and a pyright error for `dict.fromkeys(tuple_of_literals)` passed to `TTGlyphPen`. A codex helper also built a component glyph with `TTGlyphPen(None)`, which raises `TypeError` once components exist because `pen.glyph()` checks `name in glyphSet`. A brief said a bbox "spans centre" and the unit read it as strictly between the bounds.
**Why**: Units cannot run lint or the tests, and the briefs left these rules and definitions implicit.
**Prevention**: Name the rules in test-writing briefs: quote `cast()` types, prefix unused stub params with `_`, annotate `dict.fromkeys` results as `dict[str, None]`, pass a `glyphSet` mapping to `TTGlyphPen` when adding components. Define geometric predicates exactly (`ymin < y < ymax`, strict). The orchestrator runs lint and format after every wave.
**Fix**: The orchestrator corrected each test after the gate.

## Plan wiring regexes and local variables

**What happened**: Two units passed their tests but failed `loom check`: one copied directions into a local (`directions = dict(self._directions)`) where the plan's regex expected `directions=dict(self._directions)`, and one held `session.display_glyphs` in a local before `set_glyphs` where the regex expected it inline.
**Why**: Wiring regexes match literal source shapes and the briefs described the behaviour instead.
**Prevention**: When a plan pins a wiring regex, quote the literal code shape in the unit brief. Run `loom check <stage> --suggest` after each wave. Note also that loom's unwired-file scan does not match dotted imports (`from stencilizer.gui.<module> import ...`).
**Fix**: The orchestrator inlined the expressions (`functools.partial(session.save, ..., directions=dict(self._directions))`, `session.display_names[0]`).

## Review and gate failures in sandboxed stages

Reviewer rounds recorded malformed because the report went through the hand-back tool, and the review fingerprint computed inside the Bash sandbox differing from the hook one because of `/dev/null` dotfile mounts, blocked the finish of both gui-beautify stages. Rules and details: [mistakes/review-and-completion-gates](mistakes/review-and-completion-gates.md).

## README image placement

Place screenshots alongside the instructions they illustrate; avoid stacking large visuals at the top.
See [README layout](mistakes/readme-layout.md) for the correction and placement rule.

## Review probes must supply complete glyph metadata

**What happened**: A review probe failed before exercising the code because GlyphMetadata was constructed with only a name.
**Why**: Assumed defaults for required unicode, advance_width, and left_side_bearing fields.
**Prevention**: Inspect domain constructors before constructing synthetic review inputs.
**Fix**: Supply all four required metadata fields in probes.

## Preserve imports needed by new regression tests

**What happened**: During review fixes, an IO test import of Contour was removed while a new test still needed it; the agent restored it before verification.
**Why**: Import cleanup and test additions were edited together.
**Prevention**: Check imports against all new test references during edits.
**Fix**: Restored the Contour import.

## Analyzer signature changes must include instrumentation callers

**What happened**: Focused integration verification failed because the CLI analysis-count test monkeypatch accepted analyze(self, glyph), while new curve handling passed an additional UPM argument.
**Why**: Analyzer signature changes crossed a test instrumentation boundary during parallel implementation.
**Prevention**: Search wrappers and monkeypatches when extending public method signatures; retain compatibility where a new argument is unnecessary.
**Fix**: Align analyzer callers and regression instrumentation before full verification.

## New regression tests need the same static checks as source

**What happened**: New review code passed focused behavior checks but lint/type checks flagged import order, explicit zip strictness, context-manager style, Path replacement, and FontTools test annotations. A legacy test still expected the final output path at the writer boundary after transactional save introduced a temporary path.
**Why**: Leaf agents leave verification to the coordinator; integration changes also affect mocks and instrumentation.
**Prevention**: Run lint/type checks on new tests and source, and update mocks to assert the new safety guarantees.
**Fix**: Apply scoped static-check fixes and strengthen the custom-output test to validate temporary-save and final-publication behavior.

## Inspect transactional-save patches after replacement

**What happened**: An initial save-path patch retained an obsolete update_glyph call in the finally block; the agent removed it before verification.
**Why**: Multi-hunk patch context retained an old statement.
**Prevention**: Inspect the entire edited function immediately after restructuring cleanup logic.
**Fix**: Removed the obsolete call; finally only cleans up the temporary file.

## Curve flatness must use a finite chord

**What happened**: Initial adaptive flattening measured distance to the infinite chord line, so collinear control points beyond the endpoints could erase curve overshoot.
**Why**: Normal distance alone ignores longitudinal backtracking.
**Prevention**: Use clamped point-to-segment distance, test collinear overshoot, and validate positive finite tolerance with bounded subdivision.
**Fix**: Geometry agent is applying finite-chord subdivision and targeted regressions.

## Review fixes must preserve structural limits

**What happened**: Structure checks found the expanded CLI at 405 lines and two geometry functions above 50 effective lines.
**Why**: Correctness fixes added behavior to units already near their size limits.
**Prevention**: Extract cohesive helpers before adding to nearly full modules and functions; recheck after formatting.
**Fix**: Agents are decomposing the affected units without removing checks.

## Successful analysis is not necessarily a glyph edit

**What happened**: Result collection initially queued zero-bridge glyphs for serialization, which could remove their original hint instructions.
**Why**: Successful analysis was treated as successful outline modification.
**Prevention**: Gate writer updates on confirmed bridge_count greater than zero and retain statistics for analyzed no-ops.
**Fix**: Zero-bridge glyphs remain untouched in the font; regression tests assert they are absent from the writer queue.

## Offscreen GUI screenshots need a large screen

A 2x offscreen grab clamps the window to the tiny default offscreen screen, so the UI comes out cropped and magnified. Pass a 3840x2160 screen through `offscreen:configfile=` and keep `QT_SCALE_FACTOR=2`. See [offscreen-screenshots](mistakes/offscreen-screenshots.md).

## Fixed-parameter replay breaks bridge-cut coincidence across masters

**What happened**: While planning variable-font support, replaying the default master's bridge surgery in other masters by keeping each cut point at the same edge parameter t was recommended and accepted. A spike on Ubuntu[wdth,wght] showed islands at an axis extreme in 554 of 561 bridged glyphs.

**Why**: Surgery output relies on exact coincidence: a cut hole piece touches its outer piece along the bridge line, and the analyzer only accepts touching contours as non-nested. Fixed t keeps each point on its edge but moves the points of one bridge line off a common line, leaving hairline slivers that close the counter.

**Prevention**: Any cross-master replay of surgery must keep every point of one bridge line on one axis-aligned line in each master: recompute the line coordinate per master, intersect it with the polyline near the default edge, and project same-contour vertices that land on the wrong side onto the line. Validate every master with the analyzer before trusting a replay design.

**Fix**: The variable-font plan uses per-master realignment; the spike measured 23 of 561 (Ubuntu) and 27 of 576 (Inter) bridged glyphs still failing at some extreme, which the plan leaves unbridged and counts.

## CI action refs and release version bumps

setup-uv has no floating major tags, so `@v10` fails at job setup; verify every action ref with `gh api`. A version bump must also run `uv lock`, or `uv sync --locked` fails. See [ci-release](mistakes/ci-release.md).

## GUI tests open real windows when QT_QPA_PLATFORM is set

**What happened**: During the 2026-10-09 pressure test of the variable-font plan, a teammate timed `uv run pytest tests/gui/test_session.py tests/gui/test_main_window.py tests/gui/test_controller_errors.py` from a desktop session. The test windows appeared on the user's screen, showing a stenciled glyph preview.
**Why**: `tests/gui/conftest.py:24` sets `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`, and the desktop session exports `QT_QPA_PLATFORM=wayland;xcb`, so the default never applies.
**Prevention**: Run every Qt command with `QT_QPA_PLATFORM=offscreen` set explicitly on the command line; probes and acceptance commands must not rely on the conftest default. Assign the variable instead of setdefault-ing it in conftest.
**Fix**: The variable-font plan forces offscreen in `tests/gui/conftest.py` (stage cff2-static) and prefixes its GUI acceptance commands with `QT_QPA_PLATFORM=offscreen`.
