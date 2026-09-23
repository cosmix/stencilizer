# Shared contract: stencilizer GUI (stage gui-app)

Every worker reads this file in full before its own brief. The signatures below are binding: a
worker implements its own module exactly as written and imports the others exactly as written.
Where a body is described in prose, the prose is the specification.

## Rules for every worker

- Never run `git`. Write only the files your brief lists under "Files owned".
- Modules written by earlier waves exist in the worktree but NOT in the source graph: `loom map`
  answers from the base commit and will not show them. Read them with `cat`.
- Python >= 3.11, `X | None` unions. mypy runs in strict mode (`uv run mypy <files>`); ruff
  (`uv run ruff format <files>`, then `uv run ruff check --fix <files>`; zero findings left).
- `tests/regression/test_code_structure.py` enforces on everything under `src/`: file <= 400
  lines, function <= 50 lines (docstring excluded), class <= 300 lines, and no `x._font` access
  outside `io/reader.py` (use the public `FontReader.font`).
- Qt override methods (`paintEvent`, `closeEvent`) need `# noqa: N802`; add `ARG002` when the
  event parameter is unused. Ruff reports both otherwise (checked with this repo's ruff config).
- fontTools ships no type hints. Import with `# type: ignore[import-untyped]`, as
  `src/stencilizer/io/converter.py` does. Values read from a `TTFont` are `Any`: wrap them in
  `int()`/`float()` before returning (mypy `warn_return_any`).
- `fontTools.pens.qtPen.QtPen(glyphSet, path=None)` imports PyQt5 when `path` is None. Always
  construct it as `QtPen(None, path=QPainterPath())`.
- No `@Slot` decorators (they erase signatures under mypy strict).
- Signals emitted from a `QThreadPool` thread are connected ONLY to bound methods of a QObject
  that lives in the GUI thread, with `Qt.ConnectionType.QueuedConnection`. Never connect them to
  a lambda or free function: that runs the callback in the pool thread.
- Every module, public class and public function gets a docstring (the style of `src/`).
- Errors use the existing hierarchy in `src/stencilizer/exceptions.py`: `StencilizerError`,
  `FontLoadError(path: str, reason: str)` (message `Failed to load font '<path>': <reason>`),
  `FontSaveError(path: str, reason: str)` (message `Failed to save font '<path>': <reason>`),
  `GlyphNotFoundError(glyph_name: str)`. Add no new exception types.

## Tests

- pytest + pytest-qt (`qtbot`, `qapp` fixtures) under `tests/gui/`. `qt_api = "pyside6"` is set
  in `pyproject.toml`.
- `tests/gui/conftest.py` already exists. It sets `QT_QPA_PLATFORM=offscreen` and provides the
  fixtures `processor` (a `FontProcessor` logging into `tmp_path`), `roboto_path` and
  `commit_mono_path`. Never construct a `FontProcessor` in a test without a `log_file`: it would
  write `stencilizer_<timestamp>.log` into the working directory.
- Tests are exempt from `disallow_untyped_defs`, but annotate fixtures and helpers anyway.
- Run your proof command once, at the end. It must pass before you report.

## Measured facts tests may rely on (measured at commit 389557c)

- `tests/fixtures/Roboto-Regular.ttf`: format `TrueType`, UPM 2048, `hhea` ascent 2146 and
  descent -555, 3387 glyphs, 562 island glyphs. `O`, `B`, `eight` and `A` are island glyphs.
- Preview of `O` with default `BridgeConfig()`/`GeometryConfig()`: `bridges_added == 1`,
  2 contours before and 4 after. Widths 30.0 and 110.0 give different `O` outputs.
  `use_spanning_bridges=False` changes the output of `B` and `eight` versus `True`.
- `tests/fixtures/CommitMono-Cosmix-700-Regular.otf`: format `OpenType` (CFF), 1932 glyphs,
  467 island glyphs.
- One glyph transforms in at most 8.3 ms on all three fixtures (all island glyphs of a font
  together: at most 0.17 s), so previews run synchronously on the GUI thread. Classifying a font
  takes 0.37-0.65 s, so loading runs on a worker thread.
- `process_glyph` returns the island count under the key `bridges_added`; label it in the UI as
  "island(s) bridged".

## Module contract

All modules live in `src/stencilizer/gui/`. `src/stencilizer/gui/__init__.py` holds only the
docstring `"""Desktop GUI for stencilizer (PySide6)."""` and imports nothing (the CLI must keep
working without the `gui` extra).

### session.py (Qt-free)

```python
from dataclasses import dataclass
from pathlib import Path

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification, ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.utils import ProcessingStats


@dataclass(frozen=True)
class PreviewResult:
    """Outcome of stencilizing one glyph for preview."""

    glyph_name: str
    original: Glyph
    stenciled: Glyph | None  # None when the transform failed
    bridges_added: int
    error: str | None
    duration_ms: float


@dataclass
class FontSession:
    """A loaded, classified font ready for preview and saving."""

    path: Path
    font_format: str  # FontReader.format: "TrueType" or "OpenType"
    units_per_em: int
    glyph_count: int
    ascender: int  # hhea ascent
    descender: int  # hhea descent (negative)
    classification: GlyphClassification
    processor: FontProcessor

    @classmethod
    def open(cls, path: Path, processor: FontProcessor) -> "FontSession": ...

    @property
    def island_glyphs(self) -> list[Glyph]: ...

    def glyph(self, name: str) -> Glyph | None: ...

    def preview(
        self, name: str, bridge: BridgeConfig, geometry: GeometryConfig
    ) -> PreviewResult: ...

    def save(
        self,
        output_path: Path,
        settings: StencilizerSettings,
        progress: ProgressCallback | None = None,
    ) -> ProcessingStats: ...
```

### outline.py

```python
from typing import Any

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainterPath, QTransform

from stencilizer.domain import Glyph


def draw_glyph(glyph: Glyph, pen: Any) -> None: ...
def glyph_path(glyph: Glyph) -> QPainterPath: ...
def glyph_frame(glyph: Glyph, ascender: int, descender: int) -> QRectF: ...
def font_to_widget_transform(frame: QRectF, target: QRectF) -> QTransform: ...
def render_glyph_image(
    glyph: Glyph, frame: QRectF, size: int, foreground: QColor, background: QColor
) -> QImage: ...
```

### tasks.py

```python
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

ProgressFn = Callable[[int, int], None]


class TaskSignals(QObject):
    """Signals a BackgroundTask emits from its pool thread."""

    finished = Signal(object)  # the work function's return value
    failed = Signal(str)  # user-facing error message
    progress = Signal(int, int)  # completed, total


class BackgroundTask(QRunnable):
    """Run one work function on a QThreadPool thread."""

    def __init__(self, work: Callable[[ProgressFn], object]) -> None: ...
    # attribute: self.signals: TaskSignals
    def run(self) -> None: ...
```

### controls.py

```python
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QLabel, QProgressBar, QPushButton, QSlider, QSpinBox, QWidget,
)

from stencilizer.config import BridgeConfig


class ControlPanel(QWidget):
    """Font selection, parameters, and the save action."""

    open_requested = Signal()
    save_requested = Signal()
    parameters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None: ...
    # public widget attributes (tests use them):
    #   open_button: QPushButton, font_info_label: QLabel, width_slider: QSlider,
    #   width_spin: QSpinBox, spanning_check: QCheckBox, workers_spin: QSpinBox,
    #   save_button: QPushButton, progress_bar: QProgressBar
    def bridge_config(self) -> BridgeConfig: ...
    def max_workers(self) -> int | None: ...
    def set_font_info(self, text: str) -> None: ...
    def set_font_loaded(self, loaded: bool) -> None: ...
    def set_busy(self, busy: bool) -> None: ...
    def set_progress(self, completed: int, total: int) -> None: ...
    def reset_progress(self) -> None: ...
```

### glyph_view.py

```python
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import QLabel, QWidget

from stencilizer.domain import Glyph
from stencilizer.gui.session import PreviewResult


class GlyphCanvas(QWidget):
    """Paint one glyph outline scaled into the widget."""

    def __init__(self, parent: QWidget | None = None) -> None: ...
    @property
    def glyph(self) -> Glyph | None: ...
    @property
    def frame(self) -> QRectF | None: ...
    def set_glyph(self, glyph: Glyph | None, frame: QRectF | None) -> None: ...
    def paintEvent(self, event: QPaintEvent) -> None: ...  # noqa: N802, ARG002


class ComparisonView(QWidget):
    """Original and stencilized glyph side by side."""

    def __init__(self, parent: QWidget | None = None) -> None: ...
    # public attributes: before_canvas: GlyphCanvas, after_canvas: GlyphCanvas, info_label: QLabel
    def show_preview(self, result: PreviewResult, ascender: int, descender: int) -> None: ...
    def clear(self) -> None: ...
```

### glyph_grid.py

```python
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget, QWidget

from stencilizer.domain import Glyph

THUMBNAIL_SIZE = 64


class GlyphGrid(QListWidget):
    """Thumbnail grid of a font's island glyphs."""

    glyph_selected = Signal(str)  # glyph name

    def __init__(self, parent: QWidget | None = None) -> None: ...
    def set_glyphs(self, glyphs: list[Glyph], ascender: int, descender: int) -> None: ...
    def select_glyph(self, name: str) -> bool: ...
```

### controller.py

```python
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from stencilizer.config import BridgeConfig
from stencilizer.gui.session import FontSession


class GuiController(QObject):
    """Own the font session, the worker pool, and the current parameters."""

    font_loaded = Signal(object)  # FontSession
    preview_ready = Signal(object)  # PreviewResult
    save_progress = Signal(int, int)  # completed, total
    save_finished = Signal(object)  # ProcessingStats
    error = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, log_file: Path, parent: QObject | None = None) -> None: ...
    @property
    def session(self) -> FontSession | None: ...
    @property
    def is_busy(self) -> bool: ...
    def open_font(self, path: Path) -> None: ...
    def set_parameters(self, bridge: BridgeConfig, max_workers: int | None) -> None: ...
    def select_glyph(self, name: str) -> None: ...
    def save(self, output_path: Path) -> None: ...
    def shutdown(self) -> None: ...
```

### main_window.py

```python
from pathlib import Path

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from stencilizer.gui.controller import GuiController


class MainWindow(QMainWindow):
    """Top-level window: controls | glyph grid | before/after comparison."""

    def __init__(self, controller: GuiController, parent: QWidget | None = None) -> None: ...
    # public attributes: controller: GuiController, controls: ControlPanel,
    #   grid: GlyphGrid, comparison: ComparisonView
    def load_font(self, path: Path) -> None: ...
    def save_font(self, path: Path) -> None: ...
    def open_font_dialog(self) -> None: ...
    def save_font_dialog(self) -> None: ...
    def closeEvent(self, event: QCloseEvent) -> None: ...  # noqa: N802
```

### app.py

```python
import argparse
from pathlib import Path

from stencilizer.gui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser: ...
def default_log_file() -> Path: ...
def create_window(font: Path | None, log_file: Path) -> MainWindow: ...
def main(argv: list[str] | None = None) -> int: ...
```

Console script (already in `pyproject.toml`): `stencilizer-gui = "stencilizer.gui.app:main"`.
