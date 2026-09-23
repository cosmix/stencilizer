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
    path = tmp_path / "converted.otf"
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
    path = tmp_path / "modified.ttf"
    font.save(path)
    return path


def _outlines_match(saved: Glyph, expected: Glyph) -> bool:
    """Same contour/point structure and types, coordinates within 1 font unit (TrueType rounds)."""
    saved_shape = [len(contour.points) for contour in saved.contours]
    expected_shape = [len(contour.points) for contour in expected.contours]
    if saved_shape != expected_shape:
        return False
    return all(
        a.point_type == b.point_type and abs(a.x - b.x) <= 1.0 and abs(a.y - b.y) <= 1.0
        for saved_contour, expected_contour in zip(saved.contours, expected.contours, strict=True)
        for a, b in zip(saved_contour.points, expected_contour.points, strict=True)
    )


@pytest.fixture
def outlines_match() -> Callable[[Glyph, Glyph], bool]:
    """Compare a glyph read back from a saved font with the preview's stenciled glyph."""
    return _outlines_match
