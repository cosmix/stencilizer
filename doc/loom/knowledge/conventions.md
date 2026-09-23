# Coding Conventions

> Discovered coding conventions in the codebase.
> Keep current: correct or delete entries the code no longer supports.

(Add conventions as you discover them)

## Code style

- Python >=3.11 with `X | None` hints; mypy `strict = true` over `src/stencilizer` and `tests`, pydantic mypy plugin, tests exempt from `disallow_untyped_defs` (pyproject.toml `[tool.mypy]`, `[[tool.mypy.overrides]] module = "tests.*"`).
- ruff: line length 100, py311, rules E/F/I/N/W/UP/B/C4/PT/RUF/SIM/TCH/ARG/PTH, double quotes, `known-first-party = ["stencilizer"]` (pyproject.toml `[tool.ruff]`).
- Domain models are dataclasses with paired `to_dict`/`from_dict` so they cross process boundaries (src/stencilizer/domain/glyph.py:139-164, src/stencilizer/domain/contour.py:221-248).
- Config flow: CLI flags → `StencilizerSettings` (src/stencilizer/cli/app.py:179-190) → `FontProcessor(settings)` → `BridgeConfig.model_dump()` per worker task, rebuilt with `BridgeConfig(**config_dict)` (src/stencilizer/core/processor.py:56, 393).

## Tests

- Unit tests construct glyphs from domain objects with TrueType winding (CW outer / CCW hole); reader and writer tests mock fonttools.
- `tests/unit/conftest.py` shares processor fixtures across split processor test modules; `tests/integration/conftest.py` shares Roboto and CommitMono fixtures across split stencilization test modules. Other integration modules keep their local fixtures. Missing fonts or glyphs use inline `pytest.skip(...)`.
- `tests/regression/` and `tests/unit/test_refactor_contracts.py` are frozen. `tests/test_domain_models.py` sits outside `unit/`.
- Pytest adds coverage by default through pyproject.toml; use `--no-cov` for a focused run.

## Knowledge files hold current state

Only mistakes.md (and topics under mistakes/) is append-only. Every other knowledge file lists current facts: delete a concern once it is fixed and correct or remove stale claims, rather than marking them "Resolved".
