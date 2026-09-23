# W0: GUI test scaffold (wave 0, codex gpt-6-luna)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first. The orchestrator has already
added the dependencies and the `pyproject.toml` entries; do not touch `pyproject.toml` or
`uv.lock`.

## Files owned

- `tests/gui/__init__.py` (empty, like `tests/unit/__init__.py`)
- `tests/gui/conftest.py`

Read-only anchors: `tests/integration/conftest.py` (fixture style), `FontProcessor.__init__` in
`src/stencilizer/core/processor.py` (it opens a log file; without `log_file` the name is
`stencilizer_<timestamp>.log` in the working directory).

## Steps

1. Create the empty `tests/gui/__init__.py`.
2. Create `tests/gui/conftest.py` with exactly this content:

```python
"""Shared fixtures for GUI tests."""

import functools
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont, newTable  # type: ignore[import-untyped]
from fontTools.ttLib.tables._f_v_a_r import Axis  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.domain import Glyph

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def spawn_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save workers start with spawn, as app.main does, without the process-global switch."""
    monkeypatch.setattr(
        "stencilizer.core.processor.ProcessPoolExecutor",
        functools.partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    )


@pytest.fixture
def processor(tmp_path: Path) -> FontProcessor:
    """FontProcessor logging into tmp_path, never the working directory."""
    settings = StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "gui.log"))
    return FontProcessor(settings)


@pytest.fixture
def roboto_path() -> Path:
    """Roboto Regular (TrueType, UPM 2048)."""
    return FIXTURES_DIR / "Roboto-Regular.ttf"


@pytest.fixture
def commit_mono_path() -> Path:
    """CommitMono 700 (OpenType/CFF)."""
    return FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf"


@pytest.fixture
def cff2_font_path(tmp_path: Path, commit_mono_path: Path) -> Path:
    """CommitMono converted to CFF2 outlines (unsupported by the core)."""
    font = TTFont(commit_mono_path)
    convertCFFToCFF2(font)
    path = tmp_path / "CommitMono-CFF2.otf"
    font.save(path)
    return path


@pytest.fixture
def variable_font_path(tmp_path: Path, roboto_path: Path) -> Path:
    """Roboto with a one-axis fvar table: a variable font (unsupported by the core)."""
    font = TTFont(roboto_path)
    axis = Axis()
    axis.axisTag = "wght"
    axis.minValue = 100.0
    axis.defaultValue = 400.0
    axis.maxValue = 900.0
    axis.axisNameID = 256
    fvar = newTable("fvar")
    fvar.axes = [axis]
    fvar.instances = []
    font["fvar"] = fvar
    path = tmp_path / "Roboto-Variable.ttf"
    font.save(path)
    return path


def _outlines_match(saved: Glyph, expected: Glyph) -> bool:
    """Same contour/point structure, every coordinate within 1 font unit (TrueType rounds)."""
    saved_shape = [len(contour.points) for contour in saved.contours]
    expected_shape = [len(contour.points) for contour in expected.contours]
    if saved_shape != expected_shape:
        return False
    return all(
        abs(a.x - b.x) <= 1.0 and abs(a.y - b.y) <= 1.0
        for saved_contour, expected_contour in zip(saved.contours, expected.contours, strict=True)
        for a, b in zip(saved_contour.points, expected_contour.points, strict=True)
    )


@pytest.fixture
def outlines_match() -> Callable[[Glyph, Glyph], bool]:
    """Compare a glyph read back from a saved font with the preview's stenciled glyph."""
    return _outlines_match
```

3. Lint both files. The environment variable must be set at import time of the conftest (before
   pytest-qt creates the QApplication); keep it at module level. The autouse fixture keeps
   CPython 3.13's default `fork` from forking a threaded test process (the controller saves
   from a pool thread).
   `cff2_font_path` and `variable_font_path` build unsupported fonts in `tmp_path` (measured:
   the CFF2 copy has a `CFF2` table and no `CFF `; the variable copy loads with `fvar`).
   `outlines_match` checks a saved glyph against the preview (measured: saved `O`/`B` of Roboto
   differ from the preview by at most 0.06 units, CommitMono by 0.0, same point counts).

## Proof command

```bash
.venv/bin/ruff check tests/gui && .venv/bin/mypy tests/gui/conftest.py && .venv/bin/python -c "import tests.gui.conftest"
```
