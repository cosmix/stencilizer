"""Tests for per-glyph bridge direction on the GUI controller."""

from pathlib import Path

from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.gui.controller import GuiController
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats
from tests.gui.conftest import SAVE_TIMEOUT, load_session


def _spans(bbox: tuple[float, float, float, float], value: float) -> bool:
    """Return whether an (min_x, min_y, max_x, max_y) bbox spans a coordinate on the y axis."""
    return bbox[1] < value < bbox[3]


def test_set_direction_refreshes_preview(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Setting a glyph's direction refreshes its preview and is readable back."""
    load_session(controller, qtbot, roboto_path)
    with qtbot.waitSignal(controller.preview_ready) as auto_blocker:
        controller.select_glyph("O")
    previous = auto_blocker.args[0]

    with (
        qtbot.waitSignal(controller.preview_ready) as preview_blocker,
        qtbot.waitSignal(controller.direction_changed) as direction_blocker,
    ):
        controller.set_direction("O", BridgeDirection.HORIZONTAL)

    updated = preview_blocker.args[0]
    assert previous.stenciled is not None
    assert updated.stenciled is not None
    assert updated.stenciled.to_dict() != previous.stenciled.to_dict()
    assert direction_blocker.args == ["O", "horizontal"]
    assert controller.direction_for("O") is BridgeDirection.HORIZONTAL

    with qtbot.waitSignal(controller.preview_ready):
        controller.set_direction("O", BridgeDirection.AUTO)
    assert controller.direction_for("O") is BridgeDirection.AUTO


def test_set_direction_rejects_composite(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """A direction requested on a composite is refused and leaves its sources untouched."""
    load_session(controller, qtbot, roboto_path)

    with qtbot.waitSignal(controller.error) as blocker:
        controller.set_direction("Aacute", BridgeDirection.VERTICAL)

    assert "follows A" in blocker.args[0]
    assert controller.direction_for("A") is BridgeDirection.AUTO
    assert controller.direction_for("Aacute") is BridgeDirection.AUTO


def test_font_load_resets_directions(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Loading a font, even the same one, clears any previously chosen directions."""
    load_session(controller, qtbot, roboto_path)
    controller.set_direction("O", BridgeDirection.HORIZONTAL)
    assert controller.direction_for("O") is BridgeDirection.HORIZONTAL

    load_session(controller, qtbot, roboto_path)

    assert controller.direction_for("O") is BridgeDirection.AUTO


def test_survey_reports_unbridged_glyphs(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """Loading a font schedules a survey whose result reports glyphs with no bridge placed."""
    busy_values: list[bool] = []
    controller.busy_changed.connect(busy_values.append)

    with qtbot.waitSignal(controller.unbridged_changed, timeout=10_000) as blocker:
        load_session(controller, qtbot, roboto_path)

    names = blocker.args[0]
    assert "four" in names
    assert "AEacute" in names
    assert "O" not in names
    assert busy_values == [True, False]


def test_stale_survey_result_is_dropped(
    controller: GuiController, qtbot: QtBot, roboto_path: Path
) -> None:
    """A survey result from an outdated generation is dropped; the current one is emitted."""
    with qtbot.waitSignal(controller.unbridged_changed, timeout=10_000):
        load_session(controller, qtbot, roboto_path)

    stale_generation = controller._survey_generation - 1
    with qtbot.assertNotEmitted(controller.unbridged_changed):
        controller._on_survey_finished((stale_generation, frozenset({"O"})))

    with qtbot.waitSignal(controller.unbridged_changed) as blocker:
        controller._on_survey_finished((controller._survey_generation, frozenset({"O"})))
    assert blocker.args == [frozenset({"O"})]


def test_save_uses_directions(
    controller: GuiController, qtbot: QtBot, roboto_path: Path, tmp_path: Path
) -> None:
    """A save applies the controller's chosen directions to the written outlines."""
    load_session(controller, qtbot, roboto_path)
    controller.set_parameters(BridgeConfig(), 1)
    controller.set_direction("O", BridgeDirection.HORIZONTAL)
    output_path = tmp_path / "out.ttf"

    with qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT) as blocker:
        controller.save(output_path)

    stats = blocker.args[0]
    assert isinstance(stats, ProcessingStats)
    assert stats.error_count == 0
    with FontReader(output_path) as reader:
        saved_o = reader.get_glyph("O")
    assert saved_o is not None
    assert not any(_spans(c.bounding_box(), 728) for c in saved_o.contours)
