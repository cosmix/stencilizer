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

import os
from pathlib import Path

import pytest

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


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
```

3. Format and lint both files. The environment variable must be set at import time of the
   conftest (before pytest-qt creates the QApplication); keep it at module level.

## Proof command

```bash
uv run ruff format tests/gui && uv run ruff check tests/gui && uv run mypy tests/gui/conftest.py && uv run python -c "import tests.gui.conftest"
```
