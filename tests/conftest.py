"""Suite-wide fixtures."""

import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from stencilizer.utils import logging as logging_utils

# Captured at import, before any test replaces the module attribute.
_REAL_DEFAULT_LOG_PATH = logging_utils.default_log_path


@pytest.fixture(autouse=True)
def default_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Send default-named log files to a temporary directory instead of the working directory.

    Every call returns a fresh name, so a run that resolves the default twice leaves two files.
    """
    log_dir = tmp_path_factory.mktemp("default-logs")
    names = itertools.count(1)

    def in_log_dir() -> Path:
        return log_dir / f"stencilizer_{next(names):04d}.log"

    # configure_logging resolves the default in utils.logging; the CLI resolves it once per run.
    for target in (
        "stencilizer.utils.logging.default_log_path",
        "stencilizer.cli.app.default_log_path",
    ):
        monkeypatch.setattr(target, in_log_dir)
    return log_dir


@pytest.fixture
def real_default_log_path() -> Callable[[], Path]:
    """The unpatched ``default_log_path``."""
    return _REAL_DEFAULT_LOG_PATH
