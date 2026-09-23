# Stack & Dependencies

> Project technology stack, frameworks, and key dependencies.
> Keep current: correct or delete entries the code no longer supports.

(Add stack information as you discover it)

## Runtime and tooling

- Python >=3.11 (pyproject.toml:9); hatchling build, package `src/stencilizer`; uv as package manager (`uv.lock` at root).
- Runtime: fonttools>=4.65.0 (TTFont, RecordingPen, TTGlyphPen, T2CharStringPen), pydantic>=2.13.5, rich>=15.0.0, structlog>=26.1.0, typer>=0.27.2 (pyproject.toml:10-16). typer 0.27 no longer depends on click.
- Dev group: hypothesis>=6.168.1, mypy>=2.3.1, pytest>=9.1.1, pytest-cov>=7.1.0, ruff>=0.16.8 (pyproject.toml:25-32).
- Commands: `uv run pytest`, `uv run mypy`, `uv run ruff check src tests`, `uv run ruff format src tests`.
- No CI configuration in the repository.

## Supported font formats

TrueType (`glyf`) and OpenType/CFF (`CFF `) are read and written (src/stencilizer/io/converter.py:89-95). CFF2 is detected (src/stencilizer/io/reader.py:63) but writing raises `NotImplementedError`. Variable fonts (`fvar`) are unsupported. The root CLAUDE.md lists a `_update_cff2_glyph()` writer and claims static CFF2 support; neither exists. The CFF2 and variable-font plan documents that earlier notes cited are no longer in the tree.

## GUI dependencies

`pyside6-essentials>=6.11.2` is in the optional `gui` extra and the dev group; `pytest-qt>=4.5.0` and `qt_api = "pyside6"` are dev-only (pyproject.toml:23-38, 115). Install with `uv pip install -e ".[gui]"`. GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set in tests/gui/conftest.py); the offscreen plugin prints `This plugin does not support propagateSizeHints()` on show, which is not a defect. Importing `stencilizer.cli.app` must not load PySide6.
