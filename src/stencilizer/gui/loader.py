"""Loading view shown while a font opens: the wordmark is cut, sprayed and placed letter by letter.

The animation is driven by wall time, never by frame count, so dropped frames (the opening
worker holds the GIL for most of each second) skip ahead instead of slowing the loop down.
"""

from collections.abc import Callable

from PySide6.QtCore import QElapsedTimer, QEvent, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QFont,
    QGuiApplication,
    QHideEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from stencilizer.gui.loader_paint import Scene

LOADER_DELAY_MS = 500
FRAME_INTERVAL_MS = 16
HINT_TEXT = "Reading outlines and finding islands"
_REDUCED_MOTION_PROBES = ("prefersReducedMotion", "reducedMotion")


def prefers_reduced_motion() -> bool:
    """Return the platform's reduced-motion preference when Qt exposes one, else ``False``."""
    app = QGuiApplication.instance()
    if not isinstance(app, QGuiApplication):
        return False
    hints = app.styleHints()
    for name in _REDUCED_MOTION_PROBES:
        probe = getattr(hints, name, None)
        if callable(probe):
            try:
                return bool(probe())
            except (TypeError, RuntimeError):
                return False
    return False


class _ElapsedClock:
    """Monotonic seconds since construction, from a ``QElapsedTimer``."""

    def __init__(self) -> None:
        self._timer = QElapsedTimer()
        self._timer.start()

    def __call__(self) -> float:
        return self._timer.nsecsElapsed() / 1e9


class LoadingView(QWidget):
    """Animated stand-in for the glyph grid while a font is being read.

    ``start`` names the file and runs the loop; ``stop`` halts it. The frame timer only runs
    while the view is both started and visible. ``clock`` returns seconds and defaults to a
    monotonic timer; tests pass their own to render deterministic frames.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        clock: Callable[[], float] | None = None,
        reduced_motion: bool | None = None,
    ) -> None:
        """Build the labels and the scene; nothing animates until ``start``."""
        super().__init__(parent)
        self.setObjectName("loadingView")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._clock = clock or _ElapsedClock()
        self._origin = 0.0
        self._started = False
        self._status_text = ""
        reduced = prefers_reduced_motion() if reduced_motion is None else reduced_motion
        self._scene = Scene(self.palette(), reduced)
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)
        self._build_labels()

    def _build_labels(self) -> None:
        self.status_label = QLabel(self)
        self.status_label.setObjectName("loaderStatus")
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        font = QFont(self.status_label.font())
        font.setPointSizeF(font.pointSizeF() + 1.0)
        font.setWeight(QFont.Weight.DemiBold)
        self.status_label.setFont(font)
        self.hint_label = QLabel(HINT_TEXT, self)
        self.hint_label.setObjectName("loaderHint")
        self.hint_label.setProperty("role", "hint")
        self.hint_label.setTextFormat(Qt.TextFormat.PlainText)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.hint_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(24, 24, 24, 26)
        self._column.setSpacing(6)
        self._column.addStretch(1)
        self._column.addWidget(self.status_label)
        self._column.addWidget(self.hint_label)

    @property
    def reduced_motion(self) -> bool:
        """Whether the calmer variant (no sparks, no drops, no lifts) is in use."""
        return self._scene.reduced_motion

    @property
    def is_running(self) -> bool:
        """``True`` while the frame timer is ticking."""
        return self._timer.isActive()

    @property
    def particle_count(self) -> int:
        """Live sparks plus paint droplets, bounded by the scene's caps."""
        return len(self._scene.sparks.items) + len(self._scene.mist.items)

    def start(self, file_name: str) -> None:
        """Show ``Opening <file_name>…`` and run the animation from its first frame."""
        self._status_text = f"Opening {file_name}…"
        self._refresh_status_text()
        self._origin = self._clock()
        self._started = True
        self._scene.reset()
        if self.isVisible():
            self._timer.start()
        self.update()

    def stop(self) -> None:
        """Halt the animation; the view keeps its last frame until hidden."""
        self._started = False
        self._timer.stop()

    def elapsed(self) -> float:
        """Seconds since ``start``."""
        return self._clock() - self._origin

    def sizeHint(self) -> QSize:  # noqa: N802
        """Prefer the central area's usual size."""
        return QSize(520, 700)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Stay usable down to a narrow pane."""
        return QSize(300, 320)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Resume the frame timer when a started view becomes visible."""
        super().showEvent(event)
        if self._started:
            self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Never tick while hidden."""
        super().hideEvent(event)
        self._timer.stop()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Re-elide the status text for the new width."""
        super().resizeEvent(event)
        self._refresh_status_text()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Re-derive the scene colours when the theme's palette changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._scene.set_palette(self.palette())
            self.update()

    def _refresh_status_text(self) -> None:
        margins = self._column.contentsMargins()
        available = max(self.width() - margins.left() - margins.right(), 40)
        metrics = self.status_label.fontMetrics()
        self.status_label.setText(
            metrics.elidedText(self._status_text, Qt.TextElideMode.ElideMiddle, available)
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        """Paint the frame for the current clock reading; labels paint themselves on top."""
        painter = QPainter(self)
        bottom = float(self.status_label.geometry().top()) - 10.0
        self._scene.paint(painter, QRectF(self.rect()), bottom, self.elapsed())
        painter.end()
