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

## Qt and GUI code

- Qt event overrides need `# noqa: N802` (`closeEvent`, `paintEvent`), plus `ARG002` when the event argument is unused (gui/main_window.py:110, gui/glyph_view.py:38). The earlier `main_window.py:87` reference was stale: `closeEvent` sits at line 110.
- Draw glyphs with `QtPen(None, path=path)` and a PySide6 `QPainterPath`: without `path=`, `fontTools.pens.qtPen` imports PyQt5 (gui/outline.py:50). The import needs `# type: ignore[import-untyped]`.
- Connect worker signals to controller slots with `Qt.ConnectionType.QueuedConnection` so handlers run on the GUI thread; never touch widgets from a `QRunnable`.
- `FontSession` and other logic stay Qt-free; only controller, tasks and widgets import PySide6.
- The domain `Point` field is `point_type`, not `type` (src/stencilizer/domain/contour.py:58).

## GUI tests

- Under `tests/gui/`: pytest-qt `qtbot`, no `skip`/`xfail`/`importorskip`; build unsupported fonts (CFF2, variable) in `tmp_path` fixtures; never build a `FontProcessor` without a `tmp_path` `log_file`.
- Error-message asserts must match the reason text, and fixture filenames must not contain the words a test searches for (`FontLoadError` embeds the path).
- To check a thread, compare `QThread.currentThread() == controller.thread()` inside the slot and store the bool; stored PySide6 `QThread` wrappers compare unequal or raise `libshiboken: Internal C++ object already deleted` under suite load.
- Mutation spot-checks: re-apply one `str.replace` per mutant, run the named node ids, restore the bytes. Run with `PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__` with `fd -H -I` afterwards; a same-length mutant restored within one mtime second keeps its stale `.pyc`.
- In a loom stage sandbox, set `COVERAGE_FILE=$TMPDIR/cov.data` for `uv run pytest`: an existing `.coverage` is bind-mounted and coverage.py fails with EBUSY. `UV_LINK_MODE=copy` silences the hardlink warning on a fresh worktree venv.
