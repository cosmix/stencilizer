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

TrueType (`glyf`) and OpenType/CFF (`CFF `) are read and written (src/stencilizer/io/converter.py:89-95). CFF2 is detected (src/stencilizer/io/reader.py:63) but writing raises `NotImplementedError`. Variable fonts (`fvar`) are unsupported. The root CLAUDE.md lists a `_update_cff2_glyph()` writer and claims static CFF2 support; neither exists at 6e9f891. `cff2.md` and `variable-fonts.md` at the repo root are unimplemented plans with unchecked phase checklists (variable-fonts.md:1360-1425).
