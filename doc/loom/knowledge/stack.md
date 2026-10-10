# Stack & Dependencies

> Project technology stack, frameworks, and key dependencies.
> Keep current: correct or delete entries the code no longer supports.

(Add stack information as you discover it)

## Runtime and tooling

Python >=3.11; hatchling builds the src/stencilizer package. Use uv for packages. Runtime dependencies are fonttools>=4.66.0, pydantic>=2.13.5, rich>=15.0.0, skia-pathops>=0.9.2 (overlap removal and the enclosed-counter count in `variable/`), structlog>=26.1.0 and typer>=0.27.2. Development tooling includes ruff>=0.16.9, mypy, pytest, hypothesis, pytest-cov and pytest-xdist>=3.8.0 (`-n auto`; concerns.md "Full test suite near the acceptance time cap"). Run uv run pytest, uv run mypy src, and uv run ruff check src tests.

## Supported font formats

Supported: TrueType (`glyf`), static OpenType/CFF (`CFF `), static CFF2, variable TrueType (`fvar` + `glyf`, gvar rewritten) and variable CFF2 (`fvar` + `CFF2`, charstrings rewritten with `blend`). Not supported: `fvar` with `CFF ` outlines (loads, fails at save), and glyphs whose variation data the engine cannot use (left unchanged, counted as skipped with their islands counted as unbridged). `--instance` pins a variable font to a static instance first. How each route reads and writes: architecture.md "Font format I/O".

## GUI dependencies

`pyside6-essentials>=6.11.2` is in the optional `gui` extra and the dev group; `pytest-qt>=4.5.0` and `qt_api = "pyside6"` are dev-only (pyproject.toml:23-38, 115). Install with `uv pip install -e ".[gui]"`. GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set in tests/gui/conftest.py); the offscreen plugin prints `This plugin does not support propagateSizeHints()` on show, which is not a defect. Importing `stencilizer.cli.app` must not load PySide6.

## Dependency release policy

Use the latest stable releases for dependency updates; exclude prereleases. User confirmed this policy during the September 2026 dependency update and review.

## Build and release

PyInstaller (dependency group `build`, synced with `--no-dev`) freezes the CLI (onefile, PySide6 excluded) and the GUI (onedir on Linux, `Stencilizer.app` on macOS) through `packaging/build.sh`; `--smoke` runs the frozen binaries under the inherited `QT_QPA_PLATFORM` (CI uses `xvfb-run` on Linux). The entry scripts in `packaging/` call `multiprocessing.freeze_support()` first, because frozen spawn-mode ProcessPoolExecutor workers otherwise re-run main. CI (`.github/workflows/ci.yml`) calls the reusable `.github/workflows/checks.yml` and `.github/workflows/build.yml` (ubuntu-26.04 x86_64, macos-15 arm64); actions are pinned to commit SHAs and Dependabot updates them. The Linux GUI bundle copies system libraries from the build image, so its glibc floor equals the runner's (2.43 on 26.04); the CLI needs 2.35. The git tag is the only version source: hatch-vcs derives it and its build hook writes the git-ignored `src/stencilizer/_version.py`, which `__init__.py` imports; untagged builds get `X.Y.(Z+1).devN+g<sha>` (plus `.dYYYYMMDD` on a dirty tree). `packaging/build.sh` reads the version from the installed package, `.github/workflows/build.yml` checks out with `fetch-depth: 0` so tags are visible, and `[tool.uv] cache-keys` rebuilds the editable install after a commit or tag. `.github/workflows/release.yml` runs on `v*` tags: it rejects a tag whose commit is not on main or that is not `vMAJOR.MINOR.PATCH` with an optional `a`/`b`/`rc` number, runs `uv lock --check`, then publishes a GitHub release with the archives and SHA256SUMS.
