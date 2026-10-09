# Ci Release

> setup-uv lacks major tags; version bumps need uv lock

## setup-uv has no floating major tags

**What happened**: The CI workflows referenced `astral-sh/setup-uv@v10`; the ref does not exist, so every job using it would fail at job setup. actionlint did not flag it.

**Why**: Since v8.0.0 setup-uv publishes only immutable full-version tags (v10.2.0). The implementer read the latest release (v10.2.0) and assumed a v10 major tag existed, as it does for actions/checkout.

**Prevention**: Confirm every action ref resolves with `gh api repos/<owner>/<repo>/git/ref/tags/<ref>` before committing a workflow; pin setup-uv to a full version or SHA.

**Fix**: Pin `astral-sh/setup-uv` to a full version or commit SHA.

## Version bump without uv lock breaks locked CI

**What happened**: The README release steps said to bump the version in pyproject.toml and `__init__.py` only. uv.lock records the project's own version, so `uv sync --locked` would fail in every CI and release job after a bump.

**Why**: The lockfile including the root package version was overlooked.

**Prevention**: Any version bump runs `uv lock` and commits uv.lock with it.

**Fix**: Add `uv lock` to the release steps.
