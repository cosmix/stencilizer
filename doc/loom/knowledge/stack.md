# Stack & Dependencies

> Project technology stack, frameworks, and key dependencies.
> Keep current: correct or delete entries the code no longer supports.

(Add stack information as you discover it)

## Runtime and tooling

Python >=3.11; hatchling builds the src/stencilizer package. Use uv for packages. Runtime dependencies include fonttools>=4.66.0, pydantic>=2.13.5, rich>=15.0.0, structlog>=26.1.0 and typer>=0.27.2. Development tooling includes ruff>=0.16.9, mypy, pytest, hypothesis and pytest-cov. Run uv run pytest, uv run mypy src, and uv run ruff check src tests.

## Supported font formats

TrueType (glyf) and static OpenType/CFF (CFF ) fonts are supported. FontReader rejects variable fonts (fvar) and CFF2 fonts with FontFormatError before processing; FontWriter enforces the same boundary.

## GUI dependencies

`pyside6-essentials>=6.11.2` is in the optional `gui` extra and the dev group; `pytest-qt>=4.5.0` and `qt_api = "pyside6"` are dev-only (pyproject.toml:23-38, 115). Install with `uv pip install -e ".[gui]"`. GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set in tests/gui/conftest.py); the offscreen plugin prints `This plugin does not support propagateSizeHints()` on show, which is not a defect. Importing `stencilizer.cli.app` must not load PySide6.

## Dependency release policy

Use the latest stable releases for dependency updates; exclude prereleases. User confirmed this policy during the September 2026 dependency update and review.

## Build and release

PyInstaller (dependency group `build`, synced with `--no-dev`) freezes the CLI (onefile, PySide6 excluded) and the GUI (onedir on Linux, `Stencilizer.app` on macOS) through `packaging/build.sh`; `--smoke` runs the frozen binaries under the inherited `QT_QPA_PLATFORM` (CI uses `xvfb-run` on Linux). The entry scripts in `packaging/` call `multiprocessing.freeze_support()` first, because frozen spawn-mode ProcessPoolExecutor workers otherwise re-run main. CI (`.github/workflows/ci.yml`) calls the reusable `.github/workflows/checks.yml` and `.github/workflows/build.yml` (ubuntu-26.04 x86_64, macos-15 arm64); actions are pinned to commit SHAs and Dependabot updates them. The Linux GUI bundle copies system libraries from the build image, so its glibc floor equals the runner's (2.43 on 26.04); the CLI needs 2.35. The version has one source, `__version__` in `src/stencilizer/__init__.py` (hatch dynamic version; uv.lock does not record it). `.github/workflows/release.yml` runs on `v*` tags: it rejects a tag whose commit is not on main or that differs from `__version__`, runs `uv lock --check`, then publishes a GitHub release with the archives and SHA256SUMS.
