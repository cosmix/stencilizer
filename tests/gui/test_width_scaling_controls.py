"""Tests for the width-scaling controls in the sidebar and main window."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea, QSlider, QSplitter, QWidget
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig, BridgeWidthScaling
from stencilizer.gui.controller import GuiController
from stencilizer.gui.controls import ControlPanel
from stencilizer.gui.main_window import MainWindow
from stencilizer.utils import ProcessingStats
from tests.font_helpers import INTER, glyph_at
from tests.gui.conftest import LOAD_TIMEOUT, SAVE_TIMEOUT

pytestmark = pytest.mark.usefixtures("staging_root")


@pytest.fixture
def panel(qtbot: QtBot) -> ControlPanel:
    """A control panel shown for visibility checks."""
    widget = ControlPanel()
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture
def window(tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    """A main window whose controller is shut down after each test."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: None)
    controller = GuiController(tmp_path / "gui.log")
    main_window = MainWindow(controller)
    qtbot.addWidget(main_window)
    yield main_window
    controller.shutdown()


def _load_inter(window: MainWindow, qtbot: QtBot) -> None:
    errors: list[str] = []
    window.controller.error.connect(errors.append)
    with qtbot.waitSignal(window.controller.font_loaded, timeout=LOAD_TIMEOUT):
        window.load_font(INTER)
    assert not errors, errors
    qtbot.waitUntil(lambda: window.comparison.after_canvas.glyph is not None, timeout=LOAD_TIMEOUT)


def _select(panel: ControlPanel, mode: BridgeWidthScaling) -> None:
    panel.scaling_combo.setCurrentIndex(panel.scaling_combo.findData(mode))


def test_defaults_equal_settings(panel: ControlPanel) -> None:
    assert panel.bridge_config() == BridgeConfig()
    assert [panel.scaling_combo.itemText(i) for i in range(2)] == ["Fixed", "Proportional"]
    assert not panel.strength_spin.isEnabled()
    assert not panel.min_width_spin.isEnabled()


def test_proportional_enables_rows_and_emits_once(panel: ControlPanel) -> None:
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))
    _select(panel, BridgeWidthScaling.PROPORTIONAL)
    assert len(changes) == 1
    for widget in (panel.strength_slider, panel.strength_spin):
        assert widget.isEnabled()
    for widget in (panel.min_width_slider, panel.min_width_spin):
        assert widget.isEnabled()


def test_strength_and_minimum_emit_once_and_reach_config(panel: ControlPanel) -> None:
    _select(panel, BridgeWidthScaling.PROPORTIONAL)
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))
    panel.strength_slider.setValue(40)
    assert len(changes) == 1
    assert panel.strength_spin.value() == 40
    panel.min_width_spin.setValue(55)
    assert len(changes) == 2
    assert panel.min_width_slider.value() == 55
    config = panel.bridge_config()
    assert config.scaling_strength == 40.0
    assert config.min_width_percent == 55.0


def test_fixed_disables_rows(panel: ControlPanel) -> None:
    _select(panel, BridgeWidthScaling.PROPORTIONAL)
    _select(panel, BridgeWidthScaling.FIXED)
    assert not panel.strength_spin.isEnabled()
    assert not panel.min_width_slider.isEnabled()


def test_set_variable_false_resets_to_fixed_once(panel: ControlPanel) -> None:
    panel.set_variable(True)
    _select(panel, BridgeWidthScaling.PROPORTIONAL)
    panel.strength_spin.setValue(70)
    changes: list[None] = []
    panel.parameters_changed.connect(lambda: changes.append(None))
    panel.set_variable(False)
    assert len(changes) == 1
    config = panel.bridge_config()
    assert config.width_scaling is BridgeWidthScaling.FIXED
    assert config.scaling_strength == 70.0


def test_visibility_follows_set_variable(panel: ControlPanel) -> None:
    assert not panel.strength_slider.isVisibleTo(panel)
    assert not panel.min_width_slider.isVisibleTo(panel)
    panel.set_variable(True)
    assert panel.strength_slider.isVisibleTo(panel)
    assert panel.min_width_slider.isVisibleTo(panel)
    panel.set_variable(False)
    assert not panel.strength_slider.isVisibleTo(panel)
    assert not panel.min_width_slider.isVisibleTo(panel)


def test_selection_reaches_controller(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[BridgeConfig] = []
    monkeypatch.setattr(
        window.controller, "set_parameters", lambda bridge, _workers: recorded.append(bridge)
    )
    _select(window.controls, BridgeWidthScaling.PROPORTIONAL)
    assert recorded
    assert recorded[-1].width_scaling is BridgeWidthScaling.PROPORTIONAL
    assert recorded[-1].scaling_strength == 100.0
    assert recorded[-1].min_width_percent == 30.0


def test_controls_keep_minimum_height(window: MainWindow, qtbot: QtBot) -> None:
    window.resize(1280, 800)
    with qtbot.waitExposed(window):
        window.show()
    _load_inter(window, qtbot)
    assert window.controls.scaling_box.isVisibleTo(window)
    controls = window.controls
    qtbot.waitUntil(lambda: controls.height() >= controls.minimumSizeHint().height(), timeout=2000)


def _save_o(window: MainWindow, mode: BridgeWidthScaling, out: Path, qtbot: QtBot) -> TTFont:
    _select(window.controls, mode)
    with qtbot.waitSignal(window.controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        window.controller.save(out)
    assert isinstance(blocker.args[0], ProcessingStats)
    return TTFont(out)


def test_save_default_master_same_other_masters_differ(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    _load_inter(window, qtbot)
    fixed = _save_o(window, BridgeWidthScaling.FIXED, tmp_path / "fixed.ttf", qtbot)
    proportional = _save_o(window, BridgeWidthScaling.PROPORTIONAL, tmp_path / "prop.ttf", qtbot)
    assert glyph_at(fixed, "o", {}).to_dict() == glyph_at(proportional, "o", {}).to_dict()
    bold = {"wght": 1.0}
    assert glyph_at(fixed, "o", bold).to_dict() != glyph_at(proportional, "o", bold).to_dict()


def _shown_inter(window: MainWindow, qtbot: QtBot) -> ControlPanel:
    """Show the window at 1280x800 with Inter loaded and Proportional selected."""
    window.resize(1280, 800)
    with qtbot.waitExposed(window):
        window.show()
    _load_inter(window, qtbot)
    controls = window.controls
    _select(controls, BridgeWidthScaling.PROPORTIONAL)
    return controls


def _wheel(widget: QWidget, delta: int) -> QWheelEvent:
    """Send a wheel event of ``delta`` eighths of a degree to the middle of ``widget``."""
    position = QPointF(widget.rect().center())
    event = QWheelEvent(
        position,
        widget.mapToGlobal(position),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)
    return event


def _sidebar(window: MainWindow) -> QScrollArea:
    """The scroll area that wraps the control panel."""
    areas = window.findChildren(QScrollArea)
    return next(area for area in areas if area.widget() is window.controls)


def test_wheel_over_unfocused_widgets_changes_nothing(window: MainWindow, qtbot: QtBot) -> None:
    controls = _shown_inter(window, qtbot)
    strength, combo = controls.strength_slider, controls.scaling_combo
    strength.setValue(50)
    for widget in (strength, combo):
        widget.clearFocus()
        assert not widget.hasFocus()
    changes: list[None] = []
    controls.parameters_changed.connect(lambda: changes.append(None))
    for widget in (strength, combo):
        assert not _wheel(widget, -120).isAccepted()
    assert strength.value() == 50
    assert combo.currentIndex() == combo.findData(BridgeWidthScaling.PROPORTIONAL)
    assert changes == []


def test_wheel_over_unfocused_slider_scrolls_the_sidebar(window: MainWindow, qtbot: QtBot) -> None:
    controls = _shown_inter(window, qtbot)
    strength = controls.strength_slider
    strength.setValue(50)
    strength.clearFocus()
    bar = _sidebar(window).verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0, timeout=2000)
    over_slider = QPointF(strength.mapTo(window, strength.rect().center()))
    QTest.wheelEvent(window.windowHandle(), over_slider, QPoint(0, -120))
    assert bar.value() > 0
    assert strength.value() == 50


def test_wheel_over_focused_slider_changes_it(window: MainWindow, qtbot: QtBot) -> None:
    controls = _shown_inter(window, qtbot)
    strength = controls.strength_slider
    strength.setValue(50)
    window.activateWindow()
    strength.setFocus()
    qtbot.waitUntil(strength.hasFocus, timeout=2000)
    with qtbot.waitSignal(controls.parameters_changed, timeout=2000):
        _wheel(strength, 120)
    assert strength.value() > 50


def test_value_widgets_added_later_ignore_the_wheel_too(panel: ControlPanel) -> None:
    late = QWidget(panel)
    slider = QSlider(Qt.Orientation.Horizontal, late)
    late.show()
    slider.setValue(5)
    _wheel(slider, 120)
    assert slider.value() == 5
    assert slider.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_sidebar_never_clips_or_outgrows_the_controls(window: MainWindow, qtbot: QtBot) -> None:
    controls = _shown_inter(window, qtbot)
    scroll = _sidebar(window)
    splitter = scroll.parentWidget()
    assert isinstance(splitter, QSplitter)
    bar = scroll.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.isVisibleTo(scroll), timeout=2000)
    extent = bar.sizeHint().width()
    assert scroll.maximumWidth() <= controls.maximumWidth() + extent
    splitter.setSizes([0, 520, 460])
    qtbot.waitUntil(lambda: scroll.width() == scroll.minimumWidth(), timeout=2000)
    assert scroll.viewport().width() >= controls.minimumWidth()
    assert controls.width() <= scroll.viewport().width()
    splitter.setSizes([2000, 10, 10])
    qtbot.waitUntil(lambda: scroll.width() == scroll.maximumWidth(), timeout=2000)
    assert scroll.width() <= controls.maximumWidth() + extent
    assert controls.width() == scroll.viewport().width()


def test_sidebar_refits_when_the_scroll_bar_style_changes(window: MainWindow) -> None:
    scroll = _sidebar(window)
    controls = window.controls
    assert scroll.verticalScrollBar().sizeHint().width() != 23
    scroll.setStyleSheet("QScrollBar:vertical { width: 23px; }")
    assert scroll.verticalScrollBar().sizeHint().width() == 23
    assert scroll.minimumWidth() == controls.minimumWidth() + 23
    assert scroll.maximumWidth() == controls.maximumWidth() + 23
