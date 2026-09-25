# Shared contract: gui-beautify

Every worker reads this file before its own brief. It pins the public surface, the styling hooks
the theme targets, and the repository rules. Do not re-derive any of it.

## Target layout

```text
MainWindow (QMainWindow, 1280x800 initial, 960x600 minimum)
  central QWidget, QVBoxLayout, margins 0, spacing 0
    HeaderBar                        (top bar: title | font name + details | Open | Save)
    QSplitter (horizontal, stretch 1)
      ControlPanel  #sidebar         (BRIDGES card, PROCESSING card, stretch)
      QStackedWidget grid_stack      (page 0: QLabel #emptyState, page 1: GlyphGrid #glyphGrid)
      QWidget #previewPane           (ComparisonView stretch 1, QFrame[role=card] holding DirectionPicker)
  QStatusBar                         (messages left, QProgressBar #saveProgress as permanent widget)
```

## Styling hooks (the theme's selectors)

Layout workers set these; `theme.py` targets exactly these. Set a dynamic property with
`widget.setProperty("role", "<value>")` in the constructor, before the widget is shown.

| Widget | Hook | Set by |
| --- | --- | --- |
| `HeaderBar` | objectName `headerBar` | U1 |
| `HeaderBar.title_label` | objectName `appTitle` | U1 |
| `HeaderBar.font_name_label` | objectName `fontName` | U1 |
| `HeaderBar.font_details_label` | objectName `fontDetails` | U1 |
| `HeaderBar.open_button` | property `role` = `secondary` | U1 |
| `HeaderBar.save_button` | property `role` = `primary` | U1 |
| `ControlPanel` | objectName `sidebar`, `WA_StyledBackground` | U2 |
| section heading labels (sidebar, preview cards) | property `role` = `sectionTitle` | U2, U4 |
| card frames (sidebar, preview cards, picker card) | `QFrame`, property `role` = `card` | U2, U4, U5 |
| `ControlPanel.workers_value_label` | property `role` = `value` | U2 |
| explanatory labels | property `role` = `hint` | U2 |
| `ComparisonView.info_label` | property `role` = `status` | U4 |
| `GlyphGrid` | objectName `glyphGrid` | U3 |
| empty-state label | objectName `emptyState` | U5 |
| right pane `QWidget` | objectName `previewPane`, `WA_StyledBackground` | U5 |
| `MainWindow.progress_bar` | objectName `saveProgress` | U5 |

## Public surface after the stage

`src/stencilizer/gui/header.py` (new, U1):

```python
class HeaderBar(QFrame):
    open_requested = Signal()
    save_requested = Signal()
    title_label: QLabel          # "Stencilizer"
    font_name_label: QLabel      # "No font loaded" until set_font_info
    font_details_label: QLabel   # "Open a TrueType or OpenType font (.ttf, .otf)" until set_font_info
    open_button: QPushButton     # "Open Font…"
    save_button: QPushButton     # "Stencilize && Save…" (renders "Stencilize & Save…"), disabled at start
    def __init__(self, parent: QWidget | None = None) -> None: ...
    def set_font_info(self, name: str, details: str) -> None: ...
    def set_font_loaded(self, loaded: bool) -> None: ...
    def set_busy(self, busy: bool) -> None: ...
```

`src/stencilizer/gui/controls.py` (U2): `ControlPanel` keeps `parameters_changed`, `width_slider`,
`width_spin`, `spanning_check`, `bridge_config()`, `max_workers()`. It gains `workers_slider`
(`QSlider`, horizontal, range `0..(os.cpu_count() or 1)`, value 0) and `workers_value_label`
(`QLabel`, "Auto" at 0, else the number). It loses `workers_spin`, `open_requested`,
`save_requested`, `open_button`, `save_button`, `font_info_label`, `progress_bar`,
`set_font_info`, `set_font_loaded`, `set_busy`, `set_progress`, `reset_progress`.

`src/stencilizer/gui/main_window.py` (U5): `MainWindow` gains `header: HeaderBar`,
`grid_stack: QStackedWidget`, `empty_state: QLabel`, `progress_bar: QProgressBar`,
`set_progress(completed: int, total: int) -> None`, `reset_progress() -> None`. It keeps
`controls`, `grid`, `comparison`, `direction_picker`, `controller` and every existing method.

`src/stencilizer/gui/glyph_grid.py` (U3): `GlyphGrid` API unchanged; it re-renders thumbnails and
re-colours unbridged marks when its palette changes; the unbridged colour is `#c62828` on a light
base (`palette().base().color().lightness() >= 128`) and `#ff8a80` on a dark one.

`src/stencilizer/gui/theme.py` (new, D1):

```python
@dataclass(frozen=True)
class ThemeColors:
    window: str      # hex "#rrggbb" for every field
    surface: str
    base: str
    border: str
    text: str
    muted_text: str
    accent: str
    accent_text: str

LIGHT: ThemeColors
DARK: ThemeColors

def colors_for(scheme: Qt.ColorScheme) -> ThemeColors: ...          # DARK for Dark, LIGHT otherwise
def palette_for(colors: ThemeColors) -> QPalette: ...
def stylesheet_for(colors: ThemeColors) -> str: ...
def apply_theme(app: QApplication, scheme: Qt.ColorScheme | None = None) -> None: ...
```

`src/stencilizer/gui/app.py` (D1): imports `from stencilizer.gui.theme import apply_theme` and calls
`apply_theme(application)` directly after `application = QApplication(sys.argv[:1])`, before
`create_window`.

## Repository rules

- Python 3.11+, `mypy --strict` (pyproject.toml `[tool.mypy]`), ruff with the repo config, `ruff format`.
- Size limits: file 400 lines, function 50 effective lines, class 300 lines.
  `tests/regression/test_code_structure.py` enforces them on `src/` only; keep test files under 400
  lines by hand (`tests/gui/test_main_window.py` is already 364).
- Qt event overrides need `# noqa: N802` (`changeEvent`, `paintEvent`).
- `FontSession`, `composites.py` and every module outside `src/stencilizer/gui/` stay Qt-free and
  untouched. Importing `stencilizer.cli.app` must not load PySide6.
- Labels that show font-controlled text (file names, glyph names) use `Qt.TextFormat.PlainText`.
- Tests: pytest-qt `qtbot`, no `skip`/`xfail`/`importorskip`; quote `cast()` types (ruff TC006);
  prefix unused stub parameters with `_`; a `FontProcessor` or `GuiController` always gets a
  `tmp_path` log file.
- Do not touch `tests/regression/`, `tests/gui/test_beautify_contracts.py` (frozen contracts), or
  any file outside your "Files owned" row.
- No attribution to any AI system in code, comments or docstrings.

## Codex units

Codex units never run `git`, `uv` or `loom`. Your one check is static and calls the worktree venv
directly: `.venv/bin/mypy <your files> && .venv/bin/ruff check <your files> && .venv/bin/ruff format --check <your files>`.
The orchestrator runs the real tests after each wave. `loom map` answers from the base commit and
cannot see files changed earlier in this stage: read those with `cat`.
