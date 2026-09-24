"""Widgets for painting original and stencilized glyph previews."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from stencilizer.domain import Glyph
from stencilizer.gui.outline import font_to_widget_transform, glyph_frame, glyph_path
from stencilizer.gui.session import PreviewResult


class GlyphCanvas(QWidget):
    """Paint one glyph outline scaled into the widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty glyph canvas."""
        super().__init__(parent)
        self._glyph: Glyph | None = None
        self._frame: QRectF | None = None
        self.setMinimumSize(160, 160)

    @property
    def glyph(self) -> Glyph | None:
        """Return the glyph currently displayed by the canvas."""
        return self._glyph

    @property
    def frame(self) -> QRectF | None:
        """Return the font-unit frame used to scale the displayed glyph."""
        return self._frame

    def set_glyph(self, glyph: Glyph | None, frame: QRectF | None) -> None:
        """Set the glyph and common comparison frame, then repaint."""
        self._glyph = glyph
        self._frame = frame
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        """Paint the glyph and its font baseline into the widget."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if self._glyph is not None and self._frame is not None:
            target = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setTransform(font_to_widget_transform(self._frame, target))
            pen = QPen(self.palette().mid().color())
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QPointF(self._frame.left(), 0.0), QPointF(self._frame.right(), 0.0))
            painter.fillPath(glyph_path(self._glyph), self.palette().text())
        painter.end()


class ComparisonView(QWidget):
    """Original and stencilized glyph side by side."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the paired glyph canvases and their shared detail label."""
        super().__init__(parent)
        layout = QGridLayout(self)
        before_title = QLabel("Original", self)
        after_title = QLabel("Stencilized", self)
        before_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.before_canvas = GlyphCanvas(self)
        self.after_canvas = GlyphCanvas(self)
        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(before_title, 0, 0)
        layout.addWidget(after_title, 0, 1)
        layout.addWidget(self.before_canvas, 1, 0)
        layout.addWidget(self.after_canvas, 1, 1)
        layout.addWidget(self.info_label, 2, 0, 1, 2)
        layout.setRowStretch(1, 1)

    def show_preview(self, result: PreviewResult, ascender: int, descender: int) -> None:
        """Display one preview result at a shared scale in both canvases."""
        frame = glyph_frame(result.original, ascender, descender)
        if result.stenciled is not None:
            frame = frame.united(glyph_frame(result.stenciled, ascender, descender))
        self.before_canvas.set_glyph(result.original, frame)
        self.after_canvas.set_glyph(result.stenciled, frame)
        self.info_label.setText(self._preview_text(result))

    def clear(self) -> None:
        """Remove both glyph previews and their descriptive text."""
        self.before_canvas.set_glyph(None, None)
        self.after_canvas.set_glyph(None, None)
        self.info_label.clear()

    def _preview_text(self, result: PreviewResult) -> str:
        """Format the user-facing outcome summary for a glyph preview."""
        name = result.original.metadata.name
        if result.stenciled is None:
            return f"{name}: transform failed: {result.error}"
        unicode = result.original.metadata.unicode
        code = f" (U+{unicode:04X})" if unicode is not None else ""
        if result.bridges_added == 0:
            return f"{name}{code} - no bridge could be placed, {result.duration_ms:.1f} ms"
        return (
            f"{name}{code} - {result.bridges_added} island(s) bridged, {result.duration_ms:.1f} ms"
        )
