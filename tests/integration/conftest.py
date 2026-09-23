"""Shared real-font test fixtures."""

from collections.abc import Generator
from pathlib import Path

import pytest

from stencilizer.io import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
ROBOTO_PATH = FIXTURES_DIR / "Roboto-Regular.ttf"
COMMIT_MONO_OTF_PATH = FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf"


@pytest.fixture
def roboto_reader() -> Generator[FontReader, None, None]:
    """Load Roboto font for testing."""
    if not ROBOTO_PATH.exists():
        pytest.skip("Roboto font fixture not available")
    reader = FontReader(ROBOTO_PATH)
    reader.load()
    yield reader
    reader.close()


@pytest.fixture
def commit_mono_otf_reader() -> Generator[FontReader, None, None]:
    """Load CommitMono OTF font for testing."""
    if not COMMIT_MONO_OTF_PATH.exists():
        pytest.skip("CommitMono OTF font fixture not available")
    reader = FontReader(COMMIT_MONO_OTF_PATH)
    reader.load()
    yield reader
    reader.close()
