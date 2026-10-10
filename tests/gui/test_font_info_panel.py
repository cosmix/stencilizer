"""Tests for the sidebar font information panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea
from pytestqt.qtbot import QtBot

from stencilizer.gui.font_info import FontInfo, InfoSection
from stencilizer.gui.font_info_panel import PLACEHOLDER_TITLE, FontInfoPanel

INFO = FontInfo(
    "Sample Regular",
    (
        InfoSection("Font", (("Family", "Sample"), ("Style", "Regular"))),
        InfoSection("Legal & credits", (("Copyright", "(c) " + "long text " * 40),)),
    ),
)


def _panel(qtbot: QtBot) -> FontInfoPanel:
    """Create a panel registered for cleanup."""
    panel = FontInfoPanel()
    qtbot.addWidget(panel)
    return panel


def test_placeholder_until_info(qtbot: QtBot) -> None:
    """A new panel asks for a font."""
    panel = _panel(qtbot)
    texts = [label.text() for label in panel.findChildren(QLabel)]

    assert PLACEHOLDER_TITLE in texts


def test_set_info_populates_plain_text_rows(qtbot: QtBot) -> None:
    """Each row becomes a selectable plain-text value label."""
    panel = _panel(qtbot)

    panel.set_info(INFO)

    assert [label.text() for label in panel.value_labels][:2] == ["Sample", "Regular"]
    for label in panel.value_labels:
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert label.wordWrap()
        assert label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert PLACEHOLDER_TITLE not in [label.text() for label in panel.findChildren(QLabel)]


def test_second_set_info_replaces_rows(qtbot: QtBot) -> None:
    """Setting info again does not accumulate rows."""
    panel = _panel(qtbot)
    panel.set_info(INFO)
    first = len(panel.value_labels)

    panel.set_info(INFO)

    assert len(panel.value_labels) == first == 3


def test_clear_restores_placeholder(qtbot: QtBot) -> None:
    """Clearing drops the rows and shows the placeholder again."""
    panel = _panel(qtbot)
    panel.set_info(INFO)

    panel.clear()

    assert panel.value_labels == []
    assert PLACEHOLDER_TITLE in [label.text() for label in panel.findChildren(QLabel)]


def test_no_horizontal_scrollbar(qtbot: QtBot) -> None:
    """Long values wrap; the horizontal scrollbar is disabled."""
    panel = _panel(qtbot)
    panel.set_info(INFO)
    area = panel.findChild(QScrollArea)
    assert area is not None
    assert area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
