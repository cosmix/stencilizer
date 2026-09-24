"""Display and select glyph thumbnails in a font's island glyphs."""

from collections.abc import Collection

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from stencilizer.config.settings import BridgeDirection
from stencilizer.domain import Glyph
from stencilizer.gui.outline import glyph_frame, render_glyph_image

THUMBNAIL_SIZE = 64
UNBRIDGED_ROLE = Qt.ItemDataRole.UserRole + 1
BASE_TOOLTIP_ROLE = Qt.ItemDataRole.UserRole + 2


class GlyphGrid(QListWidget):
    """Thumbnail grid of a font's island glyphs."""

    glyph_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a single-selection grid with fixed-size glyph thumbnails."""
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.currentItemChanged.connect(self._on_current_item_changed)

    def _on_current_item_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        """Emit the name stored on the newly current item."""
        if current is not None:
            self.glyph_selected.emit(current.data(Qt.ItemDataRole.UserRole))

    def set_glyphs(self, glyphs: list[Glyph], ascender: int, descender: int) -> None:
        """Replace the grid with thumbnails for glyphs in the supplied order."""
        self.clear()
        foreground = self.palette().text().color()
        background = self.palette().base().color()
        for glyph in glyphs:
            item = QListWidgetItem(glyph.name)
            item.setData(Qt.ItemDataRole.UserRole, glyph.name)
            unicode_value = glyph.metadata.unicode
            tooltip = glyph.name if unicode_value is None else f"{glyph.name} U+{unicode_value:04X}"
            item.setToolTip(tooltip)
            item.setData(BASE_TOOLTIP_ROLE, tooltip)
            item.setData(UNBRIDGED_ROLE, False)
            frame = glyph_frame(glyph, ascender, descender)
            image = render_glyph_image(glyph, frame, THUMBNAIL_SIZE, foreground, background)
            item.setIcon(QIcon(QPixmap.fromImage(image)))
            self.addItem(item)

    def set_direction_marker(self, name: str, direction: BridgeDirection) -> None:
        """Show a direction marker on the named glyph without changing its identity."""
        markers = {
            BridgeDirection.AUTO: "",
            BridgeDirection.VERTICAL: " ↕",
            BridgeDirection.HORIZONTAL: " ↔",
        }
        for index in range(self.count()):
            item = self.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                item.setText(f"{name}{markers[direction]}")
                return

    def set_unbridged(self, names: Collection[str]) -> None:
        """Mark glyphs without bridges and restore all others to their base styling."""
        unbridged = set(names)
        for index in range(self.count()):
            item = self.item(index)
            if item is None:
                continue
            is_unbridged = item.data(Qt.ItemDataRole.UserRole) in unbridged
            item.setData(UNBRIDGED_ROLE, is_unbridged)
            base_tooltip = item.data(BASE_TOOLTIP_ROLE)
            if is_unbridged:
                item.setToolTip(f"{base_tooltip}\nNo bridge could be placed")
                item.setData(Qt.ItemDataRole.ForegroundRole, QColor("#c62828"))
            else:
                item.setToolTip(base_tooltip)
                item.setData(Qt.ItemDataRole.ForegroundRole, None)

    def select_glyph(self, name: str) -> bool:
        """Make the matching glyph current, returning whether it exists."""
        for index in range(self.count()):
            item = self.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self.setCurrentItem(item)
                return True
        return False
