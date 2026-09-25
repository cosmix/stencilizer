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
