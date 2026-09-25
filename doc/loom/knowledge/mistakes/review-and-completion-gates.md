# Review And Completion Gates

> Malformed reviewer rounds, sandbox fingerprint drift, codex lint leftovers

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
