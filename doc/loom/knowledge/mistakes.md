# Mistakes & Lessons Learned

> Record mistakes made during development and how to avoid them.
> This file is append-only - agents add discoveries, never delete.
>
> Format: Describe what went wrong, why, and how to avoid it next time.

(Add mistakes and lessons as you encounter them)

## Stale root CLAUDE.md claims

**What happened**: The root CLAUDE.md (as of 6e9f891) states TrueType is "CCW=outer, CW=inner" and lists a `_update_cff2_glyph()` writer with static CFF2 support.
**Why**: Docs written ahead of, or inverted from, the implementation; `cff2.md` is an unimplemented plan.
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
