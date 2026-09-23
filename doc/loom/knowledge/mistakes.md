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

**What happened**: Codex units could not record memories (loom scratch dir read-only in codex's sandbox), and `loom subagents watch` from the sandboxed Bash tool exited 3 ("process is gone") seconds after a codex forward started.
**Why**: Codex runs in its own workspace-write sandbox; each Bash call gets its own PID namespace, so the watch cannot see codex's pid.
**Prevention**: The orchestrator records codex assumptions itself. Treat that watch exit as unknown and wait for the forwarder's own completion.
