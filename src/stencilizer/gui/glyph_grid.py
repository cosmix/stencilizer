"""Display and select glyph thumbnails for a font's island glyphs and their composites."""

from collections.abc import Collection

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
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
_UNBRIDGED_ON_LIGHT = "#c62828"
_UNBRIDGED_ON_DARK = "#ff8a80"
_BADGE_SIZE = 16.0
_BADGE_INSET = 3.0


def _badged(thumbnail: QPixmap, fill: QColor, mark: QColor) -> QPixmap:
    """Return a copy of ``thumbnail`` carrying a no-bridge badge in its top-right corner.

    The badge is a ``fill`` disc with an exclamation mark and a thin ring in ``mark``, the
    thumbnail background, so it reads without colour and stands apart from glyph strokes under
    it. It stays inside the thumbnail's two-pixel inset.
    """
    badged = thumbnail.copy()
    disc = QRectF(
        badged.width() - _BADGE_INSET - _BADGE_SIZE, _BADGE_INSET, _BADGE_SIZE, _BADGE_SIZE
    )
    centre = disc.center()
    painter = QPainter(badged)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(mark, 1.5))
    painter.setBrush(fill)
    painter.drawEllipse(disc)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(mark)
    painter.drawRoundedRect(QRectF(centre.x() - 1.25, disc.top() + 3.0, 2.5, 6.5), 1.0, 1.0)
    painter.drawEllipse(QPointF(centre.x(), disc.bottom() - 3.25), 1.4, 1.4)
    painter.end()
    return badged


class GlyphGrid(QListWidget):
    """Thumbnail grid of a font's island glyphs and the composites that draw them."""

    glyph_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a single-selection grid with fixed-size glyph thumbnails."""
        super().__init__(parent)
        self._glyphs: list[Glyph] = []
        self._thumbnails: list[QPixmap] = []
        self._ascender = 0
        self._descender = 0
        self._icon_colors: tuple[QColor, QColor] | None = None
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setObjectName("glyphGrid")
        self.setGridSize(QSize(THUMBNAIL_SIZE + 28, THUMBNAIL_SIZE + 30))
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setFrameShape(QFrame.Shape.NoFrame)
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
        self._glyphs = list(glyphs)
        self._ascender = ascender
        self._descender = descender
        for glyph in self._glyphs:
            item = QListWidgetItem(glyph.name)
            item.setData(Qt.ItemDataRole.UserRole, glyph.name)
            unicode_value = glyph.metadata.unicode
            tooltip = glyph.name if unicode_value is None else f"{glyph.name} U+{unicode_value:04X}"
            item.setToolTip(tooltip)
            item.setData(BASE_TOOLTIP_ROLE, tooltip)
            item.setData(UNBRIDGED_ROLE, False)
            self.addItem(item)
        self._render_icons()

    def _render_icons(self) -> None:
        """Render every thumbnail against the current palette colours."""
        foreground = self.palette().text().color()
        background = self.palette().base().color()
        self._icon_colors = (foreground, background)
        self._thumbnails = [
            QPixmap.fromImage(
                render_glyph_image(
                    glyph,
                    glyph_frame(glyph, self._ascender, self._descender),
                    THUMBNAIL_SIZE,
                    foreground,
                    background,
                )
            )
            for glyph in self._glyphs
        ]
        for index in range(self.count()):
            self._update_icon(index)

    def _update_icon(self, index: int) -> None:
        """Show an item's thumbnail, badged while the item is marked unbridged."""
        item = self.item(index)
        if item is None or index >= len(self._thumbnails):
            return
        thumbnail = self._thumbnails[index]
        if item.data(UNBRIDGED_ROLE):
            thumbnail = _badged(thumbnail, self._unbridged_color(), self.palette().base().color())
        item.setIcon(QIcon(thumbnail))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Refresh palette-dependent thumbnails and unbridged marks."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            colors = (self.palette().text().color(), self.palette().base().color())
            if colors != self._icon_colors:
                self._render_icons()
            self._recolor_unbridged()

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
            changed = item.data(UNBRIDGED_ROLE) != is_unbridged
            item.setData(UNBRIDGED_ROLE, is_unbridged)
            if changed:
                self._update_icon(index)
            base_tooltip = item.data(BASE_TOOLTIP_ROLE)
            if is_unbridged:
                item.setToolTip(f"{base_tooltip}\nNo bridge could be placed")
                item.setData(Qt.ItemDataRole.ForegroundRole, self._unbridged_color())
            else:
                item.setToolTip(base_tooltip)
                item.setData(Qt.ItemDataRole.ForegroundRole, None)

    def _unbridged_color(self) -> QColor:
        """Return a red that contrasts with the current grid background."""
        base = self.palette().base().color()
        color = _UNBRIDGED_ON_LIGHT if base.lightness() >= 128 else _UNBRIDGED_ON_DARK
        return QColor(color)

    def _recolor_unbridged(self) -> None:
        """Update marked glyphs after the palette changes."""
        for index in range(self.count()):
            item = self.item(index)
            if item is not None and item.data(UNBRIDGED_ROLE):
                item.setData(Qt.ItemDataRole.ForegroundRole, self._unbridged_color())

    def select_glyph(self, name: str) -> bool:
        """Make the matching glyph current, returning whether it exists."""
        for index in range(self.count()):
            item = self.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self.setCurrentItem(item)
                return True
        return False
