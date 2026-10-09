# Review And Completion Gates

> Malformed reviews, fingerprint drift, IV process traps, loom tool quirks

## Reviewer rounds recorded malformed

**What happened**: Seven `loom-code-reviewer` rounds across gui-beautify (three) and integration-verify (two of three, plus a repeat after the brief named the rule) were recorded malformed ("no loom-review block") although the reviewer had produced the block.
**Why**: The harness gives the reviewer a `SubagentHandback` tool. The reviewer delivers its report through it, then ends with a plain-text line such as "Review complete and handed back."; the `review-harvest` hook reads only the last plain-text assistant message, so the block inside the hand-back is never seen.
**Prevention**: State in every reviewer brief, and repeat it in each re-review brief, that the LAST plain-text message must itself end with the `loom-review` block with nothing after it, and that no hand-back tool is to be called. Read the hand-back text too: fix its findings even when the round is malformed, then run a further round.

## Review fingerprint differs inside and outside the Bash sandbox

**What happened**: The stage completion command, run from the sandboxed Bash tool, failed only on the review gate after clean rounds (gui-beautify: `sha256:3c1320e7` recorded, `sha256:72c10cf8` computed; integration-verify: `042f5acd` recorded, `9736e4e2` computed). A new round did not close the gap.
**Why**: The sandbox bind-mounts `/dev/null` over denied dotfiles in the worktree root (`.bashrc`, `.gitconfig`, `.gitmodules`, `.idea`, `.mcp.json`, `.profile`, `.ripgreprc`, `.vscode`, `.zshrc`, ...). The review hook runs outside the sandbox and never sees those char devices; the in-sandbox command counts them as untracked changes. The same mounts make `loom stage contracts freeze` refuse to freeze from the sandbox. Committing after the last round also changed the fingerprint once, so a commit is not neutral.
**Prevention**: Expect those files in "changed since"; they are never staged. Commit first, run the final review round after the commit, then run the contract freeze and the stage completion from a shell outside the sandbox (the user runs them with `! <command>` from the worktree root). Report this as a blocker instead of retrying.

## Codex unit reports a lint fix without rerunning the check

**What happened**: A codex unit reported fixing ruff `PT013` in `tests/gui/test_header.py` and left an `I001` import-order error (pytest after PySide6).
**Why**: The unit's one static check ran before its last edit.
**Prevention**: The orchestrator runs `ruff check --fix` and `ruff format` on every codex wave's files before the tests (see mistakes.md "Test-writing units fail the repo's lint gate").

## Concurrent pull recovery

**What happened:** Tracked review edits disappeared while the branch advanced to the remote GUI changes. Restoring the reviewed archive with a three-way merge temporarily left conflict markers in pyproject.toml, causing settings discovery to fail.

**Why:** Recovery overlapped dependency and processing changes added remotely.

**Prevention:** Preserve a patch before concurrent version-control work and resolve manifests before invoking project tools.

**Fix:** Restore the pulled manifests, reapply stable dependency updates through uv, preserve GUI changes, and verify the combined tree before committing.

The recovered early font-format rejection initially used different wording from the pulled GUI. Preserve the existing GUI wording for CFF2 and variable fonts when rejecting in FontReader; GUI integration tests cover this boundary.

## Hardening round on non-blocking suggestions

**What happened**: In integration-verify, after the gate was green and review round 13 was clean, an extra hardening round (an engineer spawn, another full gate and another review) went to non-blocking reviewer suggestions and added about 30 minutes; the user objected to the time.
**Why**: Suggestions were treated as work to finish before completion.
**Prevention**: Once the gate is green and the review round matching the tree has no findings, commit and complete; leave suggestions pending for knowledge-distill unless one names a concrete correctness failure.

## Workers skipped format and type checks

**What happened**: Gate round 1 of variable-engine failed on `ruff format` (overlaps.py) and mypy (replay.py:258, a loop variable reused with two types).
**Why**: Workers read the no-verify rule as covering format and type checks.
**Prevention**: Briefs ask each worker to run `ruff format` and mypy once on its own files as its single narrow check.

## Refactors in integration-verify break earlier stages' wiring patterns

**What happened**: Completion of integration-verify re-runs every earlier stage's wiring grep patterns; moving code (a classification dataclass into its own module) or folding calls into a helper (`print_reader_info`) broke patterns such as `unsupported_islands` in core/processor.py and `axes=` in cli/app.py.
**Prevention**: Before refactoring in integration-verify, list the plan's wiring entries (`rg -n 'pattern:' doc/plans/*PLAN*.md`) and re-check them after every fix round.

## Orchestration tool traps

- Piping `loom stage commit` through `rg` or `head` drops the `LOOM_RELAY_V1` line, so the relay hook never sees the ticket and `loom request status` reports "not relayed yet". Pipe only through `tail -3`, or not at all.
- The completion artifact check flags any `raise NotImplementedError` in an artifact file, including an old unsupported-format branch; raise a typed error instead.
- `loom subagents watch` exits 5 ("worker set does not resolve to one Claude parent UUID") when started right after spawning a mixed Claude and codex pair, and exits 3 when a codex job record still says running while its process is gone even though the job ended `completed`, exit 0. Check the job record before treating exit 3 as a codex failure; fall back to the agent completion notifications on exit 5.
- Untracked dotfiles at the worktree root (`.bashrc`, `.zshrc`, `.gitconfig`, `.idea`, `.vscode`, `.mcp.json`, ...) are sandbox mount points; stage named files only (see "Review fingerprint differs inside and outside the Bash sandbox").
- The Read tool is blocked for files under `$TMPDIR`; copy the file into the scratchpad directory or read it with `rg -n '' <file>`.
- A codex unit that times out at 540 s leaves its fix unfinished (fontTools `instantiateVariableFont(downgradeCFF2=True)` raises "Input font does not contain a CFF2 table" on glyf fonts; pass `downgradeCFF2="CFF2" in font`); the orchestrator finishes it or re-splits the unit.
