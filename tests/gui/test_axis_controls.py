"""Axis sliders and location normalization."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from PySide6.QtWidgets import QLabel, QSlider
from pytestqt.qtbot import QtBot

from stencilizer.gui.axis_controls import AxisPanel
from stencilizer.gui.variable_session import AxisInfo, normalize, read_avar, read_axes

FIXTURES = Path(__file__).parent.parent / "fixtures" / "variable"


@pytest.fixture
def ubuntu_axes() -> tuple[AxisInfo, ...]:
    """The fvar axes of the Ubuntu fixture (wdth, wght)."""
    return read_axes(TTFont(FIXTURES / "Ubuntu-VF-subset.ttf"))


@pytest.fixture
def panel(qtbot: QtBot, ubuntu_axes: tuple[AxisInfo, ...]) -> AxisPanel:
    """A shown panel built from the Ubuntu axes."""
    widget = AxisPanel()
    qtbot.addWidget(widget)
    widget.set_axes(ubuntu_axes)
    widget.show()
    return widget


def test_panel_builds_one_slider_per_axis(
    panel: AxisPanel, ubuntu_axes: tuple[AxisInfo, ...]
) -> None:
    assert [axis.tag for axis in ubuntu_axes] == ["wdth", "wght"]
    for axis in ubuntu_axes:
        slider = panel.findChild(QSlider, f"axis-slider-{axis.tag}")
        assert slider is not None
    assert panel.location() == {axis.tag: axis.default for axis in ubuntu_axes}


def test_slider_emits_user_value(panel: AxisPanel, ubuntu_axes: tuple[AxisInfo, ...]) -> None:
    wght = next(a for a in ubuntu_axes if a.tag == "wght")
    slider = panel.findChild(QSlider, "axis-slider-wght")
    assert slider is not None
    emitted: list[dict[str, float]] = []
    panel.location_changed.connect(emitted.append)
    slider.setValue(slider.maximum())
    assert emitted[-1]["wght"] == wght.maximum
    # wdth spans under 50 units, so its slider steps in tenths.
    wdth = panel.findChild(QSlider, "axis-slider-wdth")
    assert wdth is not None
    wdth.setValue(wdth.minimum() + 5)
    assert emitted[-1]["wdth"] == pytest.approx(
        next(a for a in ubuntu_axes if a.tag == "wdth").minimum + 0.5
    )


def test_empty_axes_hide_the_panel(panel: AxisPanel) -> None:
    assert panel.isVisible()
    panel.set_axes(())
    assert not panel.isVisible()
    assert panel.location() == {}


def test_normalize_applies_avar() -> None:
    font = TTFont(FIXTURES / "Inter-VF-subset.ttf")
    axes, avar = read_axes(font), read_avar(font)
    mapped = normalize({"wght": 700}, axes, avar)
    assert abs(mapped["wght"] - 0.54) < 0.01
    assert normalize({"wght": 700}, axes, {})["wght"] == pytest.approx(0.6)
    assert normalize({}, axes, avar) == {axis.tag: 0.0 for axis in axes}


def test_composite_note_toggles_inputs(panel: AxisPanel) -> None:
    note = panel.findChild(QLabel, "axis-composite-note")
    slider = panel.findChild(QSlider, "axis-slider-wght")
    assert note is not None
    assert slider is not None
    assert not note.isVisibleTo(panel)
    panel.set_location_applies(False)
    assert not slider.isEnabled()
    assert note.isVisibleTo(panel)
    assert note.text() == "Composite glyphs preview at the default axis location."
    panel.set_location_applies(True)
    assert slider.isEnabled()
    assert not note.isVisibleTo(panel)
