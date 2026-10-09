"""Contracts for variable-font GUI sessions and axis sliders (stage variable-surfaces)."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.varLib.models import (  # type: ignore[import-untyped]
    normalizeLocation,
    piecewiseLinearMap,
)
from PySide6.QtWidgets import QLabel, QMessageBox, QSlider
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.domain import Glyph
from stencilizer.gui.composites import compose, load_component_outlines
from stencilizer.gui.controller import GuiController
from stencilizer.gui.main_window import MainWindow
from stencilizer.gui.session import FontSession
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats
from tests.font_helpers import INTER, UBUNTU, glyph_at
from tests.font_helpers import island_count as _islands
from tests.font_helpers import units_per_em as _upm
from tests.gui.conftest import LOAD_TIMEOUT, SAVE_TIMEOUT, build_settings, load_session

pytestmark = pytest.mark.usefixtures("staging_root")


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    """A main window whose controller is shut down after each test.

    Controller errors reach the test through the error signal; the modal warning
    box is replaced so a failed load cannot block the offscreen event loop.
    """
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: None)
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _load_window(window: MainWindow, qtbot: QtBot, path: Path) -> None:
    loaded: list[object] = []
    errors: list[str] = []
    window.controller.font_loaded.connect(loaded.append)
    window.controller.error.connect(errors.append)
    window.load_font(path)
    qtbot.waitUntil(lambda: bool(loaded or errors), timeout=LOAD_TIMEOUT)
    assert not errors, errors
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)


def _expected_location(font: TTFont, user: dict[str, float]) -> dict[str, float]:
    """fvar normalization followed by the avar segment map, per axis."""
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in font["fvar"].axes}
    normalized = normalizeLocation(user, axes)
    segments = font["avar"].segments
    return {
        tag: float(piecewiseLinearMap(value, segments[tag])) for tag, value in normalized.items()
    }


def _assert_points_close(actual: Glyph, expected: Glyph, tolerance: float) -> None:
    assert [len(c.points) for c in actual.contours] == [len(c.points) for c in expected.contours]
    for actual_contour, expected_contour in zip(actual.contours, expected.contours, strict=True):
        for a, b in zip(actual_contour.points, expected_contour.points, strict=True):
            assert abs(a.x - b.x) <= tolerance
            assert abs(a.y - b.y) <= tolerance


def test_session_previews_at_location(processor: FontProcessor) -> None:
    from stencilizer.variable.reader import read_variable_glyph
    from stencilizer.variable.transform import transform_variable_glyph

    session: Any = FontSession.open(INTER, processor)
    light = session.preview("o", BridgeConfig(), GeometryConfig(), location={"wght": 300})
    bold = session.preview("o", BridgeConfig(), GeometryConfig(), location={"wght": 700})
    assert light.stenciled is not None
    assert bold.stenciled is not None
    assert light.stenciled.to_dict() != bold.stenciled.to_dict()
    font = TTFont(INTER)
    upm = _upm(font)
    assert _islands(light.stenciled, upm) == 0
    assert _islands(bold.stenciled, upm) == 0
    expected_location = _expected_location(font, {"wght": 700, "opsz": 14})
    assert abs(expected_location["wght"] - 0.54) < 0.01
    vg = read_variable_glyph(font, "o")
    assert vg is not None
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    _assert_points_close(bold.stenciled, outcome.glyph.instance(expected_location), 0.5)


def test_axis_sliders_drive_preview(window: MainWindow, qtbot: QtBot) -> None:
    _load_window(window, qtbot, UBUNTU)
    assert window.grid.select_glyph("o")
    slider = window.findChild(QSlider, "axis-slider-wght")
    assert slider is not None
    current = window.comparison.after_canvas.glyph
    assert current is not None
    recorded = current.to_dict()
    assert slider.value() != slider.maximum()
    with qtbot.waitSignal(window.controller.preview_ready, timeout=LOAD_TIMEOUT) as blocker:
        slider.setValue(slider.maximum())
    result = blocker.args[0]
    assert result.stenciled is not None
    assert result.stenciled.to_dict() != recorded


def test_gui_saves_variable_font(controller: GuiController, qtbot: QtBot, tmp_path: Path) -> None:
    load_session(controller, qtbot, UBUNTU)
    out = tmp_path / "out.ttf"
    with qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        controller.save(out)
    assert isinstance(blocker.args[0], ProcessingStats)
    saved = TTFont(out)
    assert "fvar" in saved
    assert "gvar" in saved
    bold_o = glyph_at(saved, "o", {"wght": 1.0})
    assert _islands(bold_o, _upm(saved)) == 0


def _assert_composite_session(
    processor: FontProcessor, tmp_path: Path, outlines_match: Callable[[Glyph, Glyph], bool]
) -> None:
    session: Any = FontSession.open(INTER, processor)
    assert session.is_composite("Aacute")
    default = session.preview("Aacute", BridgeConfig(), GeometryConfig(), location=None)
    moved = session.preview("Aacute", BridgeConfig(), GeometryConfig(), location={"wght": 700})
    assert default.stenciled is not None
    assert moved.stenciled is not None
    assert default.stenciled.to_dict() == moved.stenciled.to_dict()
    assert _islands(default.stenciled, session.units_per_em) == 0
    out = tmp_path / "out.ttf"
    session.save(out, build_settings(processor))
    composite = next(c for c in session.composites if c.name == "Aacute")
    with FontReader(out) as reader:
        saved = compose(composite, load_component_outlines(reader, [composite]))
    assert outlines_match(saved, default.stenciled)


def _assert_composite_window(window: MainWindow, qtbot: QtBot) -> None:
    _load_window(window, qtbot, INTER)
    slider = window.findChild(QSlider, "axis-slider-wght")
    assert slider is not None
    assert window.grid.select_glyph("Aacute")
    note = window.findChild(QLabel, "axis-composite-note")
    assert note is not None
    assert not slider.isEnabled()
    assert note.isVisibleTo(window)
    assert note.text() == "Composite glyphs preview at the default axis location."
    assert window.grid.select_glyph("o")
    assert slider.isEnabled()
    assert not note.isVisibleTo(window)


def test_composite_preview_at_default(
    processor: FontProcessor,
    tmp_path: Path,
    outlines_match: Callable[[Glyph, Glyph], bool],
    window: MainWindow,
    qtbot: QtBot,
) -> None:
    _assert_composite_session(processor, tmp_path, outlines_match)
    _assert_composite_window(window, qtbot)
