# Shared contract: stencilizer GUI (stage gui-app)

Every worker reads this file in full before its own brief. The signatures below are binding: a
worker implements its own module exactly as written and imports the others exactly as written.
Where a body is described in prose, the prose is the specification.

## Rules for every worker

- Never run `git`. Write only the files your brief lists under "Files owned".
- Modules written by earlier waves exist in the worktree but NOT in the source graph: `loom map`
  answers from the base commit and will not show them. Read them with `cat`.
- Python >= 3.11, `X | None` unions. mypy runs in strict mode and ruff with the repo config.
  Inside codex's sandbox `uv run` fails (no network, read-only uv cache and `/tmp`): call the
  tools through `.venv/bin/` (`.venv/bin/ruff check --fix <files>`, `.venv/bin/mypy <files>`).
  The orchestrator formats (`ruff format`) and runs the tests after each wave.
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
- No `@Slot` decorators (not needed: plain methods work as PySide6 slots).
- Signals emitted from a `QThreadPool` thread are connected ONLY to bound methods of a QObject
  that lives in the GUI thread, with `Qt.ConnectionType.QueuedConnection`. Never connect them to
  a lambda or free function: that runs the callback in the pool thread.
- The import block shown per module below lists only the names its public signatures use.
  Import whatever else the implementation needs (`Qt`, `QPointF`, `QSignalBlocker`, layouts,
  `QListWidgetItem`, `QIcon`, `QPixmap`, `QSize`, ...) and let `ruff check --fix` sort the block
  (ruff reports I001 otherwise).
- Every module, public class and public function gets a docstring (the style of `src/`).
- Errors use the existing hierarchy in `src/stencilizer/exceptions.py`: `StencilizerError`,
  `FontLoadError(path: str, reason: str)` (message `Failed to load font '<path>': <reason>`),
  `FontSaveError(path: str, reason: str)` (message `Failed to save font '<path>': <reason>`),
  `GlyphNotFoundError(glyph_name: str)`. Add no new exception types.

## Tests

- pytest + pytest-qt (`qtbot`, `qapp` fixtures) under `tests/gui/`. `qt_api = "pyside6"` is set
  in `pyproject.toml`.
- `tests/gui/conftest.py` already exists. It sets `QT_QPA_PLATFORM=offscreen` and provides the
  fixtures `processor` (a `FontProcessor` logging into `tmp_path`), `roboto_path`,
  `commit_mono_path`, `cff2_font_path` and `variable_font_path` (unsupported fonts built in
  `tmp_path`), and `outlines_match(saved, expected) -> bool` (same contour/point structure,
  coordinates within 1 unit: compares a glyph read back from a saved font with a preview). Never construct a `FontProcessor` in a test without a `log_file`: it would
  write `stencilizer_<timestamp>.log` into the working directory. It also has an autouse fixture
  patching `stencilizer.core.processor.ProcessPoolExecutor` to a spawn context, so every save in
  a GUI test starts its workers the way `app.main` does (never forked from a threaded process).
- Tests are exempt from `disallow_untyped_defs`, but annotate fixtures and helpers anyway.
- Run your brief's proof command once, at the end, and report its output whether it passes or
  fails. Never loop on it and never run the tests themselves (codex cannot: `tmp_path` needs a
  writable temp dir); the orchestrator runs them after the wave.
- No `skip`/`xfail`/`importorskip` under `tests/gui/` (an acceptance command rejects them).
- These test names are fixed; the stage acceptance runs them by node id:
  `test_session.py::test_save_refuses_input_path`, `test_session.py::test_save_uses_given_settings`,
  `test_session.py::test_save_writes_stenciled_outlines`,
  `test_session.py::test_save_refuses_changed_source`,
  `test_session.py::test_open_rejects_unsupported_fonts`,
  `test_outline.py::test_winding_fill_shows_broken_hole`,
  `test_glyph_view.py::test_show_preview_shares_one_frame`,
  `test_controller.py::test_open_font_busy_guard`,
  `test_controller.py::test_single_processor_per_controller`,
  `test_controller.py::test_signals_delivered_on_gui_thread`,
  `test_controller.py::test_shutdown_waits_for_active_save`,
  `test_main_window.py::test_close_while_busy_is_refused`,
  `test_main_window.py::test_saved_font_matches_preview`,
  `test_main_window.py::test_unsupported_font_is_rejected`,
  `test_app.py::test_main_sets_spawn_and_shows_window`,
  `test_app.py::test_main_subprocess_saves_with_spawn`.
- Every test that saves a supported fixture asserts `stats.error_count == 0` and the exact
  processed count (Roboto 562, CommitMono 467), and inspects the saved outlines after reopening
  the output; `processed_count + error_count == total` alone passes when every glyph fails.

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
- `FontProcessor.process` reads `self.config.bridge`/`geometry` at call time
  (`config_dict = self.config.bridge.model_dump()` in `FontProcessor._process_glyphs_parallel`,
  `src/stencilizer/core/processor.py`), so swapping `processor.config` per save
  takes effect: a saved Roboto `O` differs between widths 30 and 110 (4 contours each).
- Saving with `ProcessingConfig(max_workers=1)` and default settings: Roboto 562 processed,
  0 errors, about 0.6 s; CommitMono 467 processed, 0 errors, about 0.7 s. The saved `O` has
  4 contours in both, with the preview's point counts; coordinates differ from the preview by at
  most 0.06 units (Roboto, TrueType rounding) and 0.0 (CommitMono). Saved `B` matches too.
- `fontTools.cffLib.CFFToCFF2.convertCFFToCFF2` turns CommitMono into a font with a `CFF2`
  table and no `CFF `; `FontReader` loads it and calls it `"OpenType"`, and without a GUI check
  it classifies 26 island glyphs whose writes then fail with `NotImplementedError`.
- `.notdef` is the first island glyph of all three fixtures, so the glyph auto-selected after a
  load is `.notdef`; its preview succeeds.

## Module contract

All modules live in `src/stencilizer/gui/`. `src/stencilizer/gui/__init__.py` holds only the
docstring `"""Desktop GUI for stencilizer (PySide6)."""` and imports nothing (the CLI must keep
working without the `gui` extra).

### session.py (Qt-free)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor, process_glyph
from stencilizer.core.processor import GlyphClassification, ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.exceptions import (
    FontLoadError,
    FontSaveError,
    GlyphNotFoundError,
    StencilizerError,
)
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats


def source_digest(path: Path) -> str:
    """SHA-256 hex digest of the file's bytes (the pinned source revision)."""
    ...


def unsupported_reason(font: Any) -> str | None:
    """Why the core cannot stencilize this TTFont (fvar, CFF2, no glyf/CFF), or None."""
    ...


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
    source_sha256: str  # source_digest(path) at open; save refuses a changed source

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

from PySide6.QtCore import QPointF, QRectF
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
