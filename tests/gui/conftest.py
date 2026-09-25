"""Shared fixtures for GUI tests."""

import functools
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2  # type: ignore[import-untyped]
from fontTools.misc.textTools import Tag  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.tables._f_v_a_r import Axis, table__f_v_a_r  # type: ignore[import-untyped]
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig, LoggingConfig, ProcessingConfig, StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.domain import Glyph
from stencilizer.gui.controller import GuiController
from stencilizer.gui.session import FontSession

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LOAD_TIMEOUT = 30_000
SAVE_TIMEOUT = 120_000


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
def staging_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Stage saves under tmp_path and fail any test that leaves a staging directory behind."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    yield root
    assert list(root.glob("stencilizer-gui-*")) == []


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
    axis.axisTag = Tag("wght")
    axis.minValue = 100.0
    axis.defaultValue = 400.0
    axis.maxValue = 900.0
    axis.axisNameID = 256
    fvar = table__f_v_a_r()
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


@pytest.fixture
def controller(tmp_path: Path) -> Iterator[GuiController]:
    """Create a controller whose private thread pool is cleaned up after each test."""
    result = GuiController(tmp_path / "gui.log")
    yield result
    result.shutdown()


def load_session(controller: GuiController, qtbot: QtBot, path: Path) -> FontSession:
    """Load a font and return the session delivered by the controller."""
    with qtbot.waitSignal(controller.font_loaded, timeout=LOAD_TIMEOUT) as blocker:
        controller.open_font(path)
    session = blocker.args[0]
    assert isinstance(session, FontSession)
    return session


def build_settings(
    processor: FontProcessor, bridge: BridgeConfig | None = None
) -> StencilizerSettings:
    """Build serial-save settings while retaining the fixture logger."""
    return StencilizerSettings(
        bridge=bridge or BridgeConfig(),
        processing=ProcessingConfig(max_workers=1),
        logging=processor.config.logging,
    )
