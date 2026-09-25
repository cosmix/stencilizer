# U5: MainWindow layout (codex gpt-5.6-terra, wave 2)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first.

**Owns:** `src/stencilizer/gui/main_window.py`.
**Reads (changed in wave 1, so read them with `cat`, not `loom map`):**
`src/stencilizer/gui/header.py` (new `HeaderBar`), `src/stencilizer/gui/controls.py` (the file
actions are gone; `workers_slider` replaced `workers_spin`).
**Start from:** `MainWindow.__init__`, `_on_font_loaded`, `_on_save_finished`, `_on_error` in
`src/stencilizer/gui/main_window.py`.

## Steps

1. Split construction. `__init__` keeps `super().__init__(parent)`, `self.controller = controller`,
   `self._output_path`, `self._current_glyph = None`, `self.setWindowTitle("Stencilizer")`, and adds
   `self.resize(1280, 800)` and `self.setMinimumSize(960, 600)`; it then calls
   `self._build_panes()`, `self._build_status_bar()`, `self._connect_signals()`,
   `self._connect_direction_signals()` (unchanged) and `self._update_parameters()`.
   - `_build_panes(self) -> None`: `self.header = HeaderBar()`; `self.controls = ControlPanel()`;
     `self.grid = GlyphGrid()`; `self.empty_state = QLabel("Open a font to see the glyphs that need bridges")`
     with objectName `emptyState`, `setAlignment(Qt.AlignmentFlag.AlignCenter)`, `setWordWrap(True)`;
     `self.grid_stack = QStackedWidget()` with `addWidget(self.empty_state)` then
     `addWidget(self.grid)` (the empty state shows first). Right pane: `right_pane = QWidget()`,
     objectName `previewPane`, `setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)`,
     `right_layout = QVBoxLayout(right_pane)` margins `(16, 16, 16, 16)` spacing 12;
     `self.comparison = ComparisonView(right_pane)`; `picker_card = QFrame(right_pane)` with
     `setProperty("role", "card")` and a `QVBoxLayout` (margins `(4, 4, 4, 4)`) holding
     `self.direction_picker = DirectionPicker(picker_card)`; `right_layout.addWidget(self.comparison, 1)`;
     `right_layout.addWidget(picker_card)`. `splitter = QSplitter(Qt.Orientation.Horizontal)`: add
     `self.controls`, `self.grid_stack`, `right_pane`; `setChildrenCollapsible(False)`,
     `setHandleWidth(1)`, `setStretchFactor(0, 0)`, `setStretchFactor(1, 1)`, `setStretchFactor(2, 0)`,
     `setSizes([260, 560, 460])`. `central = QWidget()`, `QVBoxLayout(central)` margins 0 spacing 0:
     `addWidget(self.header)`, `addWidget(splitter, 1)`; `self.setCentralWidget(central)`.
   - `_build_status_bar(self) -> None`: `self.progress_bar = QProgressBar()`, objectName
     `saveProgress`, `setMaximumWidth(220)`, `setTextVisible(True)`, `hide()`;
     `self.statusBar().addPermanentWidget(self.progress_bar)`.
   - `_connect_signals(self) -> None` holds the connections from today's `__init__` with these
     changes, written in exactly these shapes (the plan's wiring checks match them):
     `self.header.open_requested.connect(self.open_font_dialog)`,
     `self.header.save_requested.connect(self.save_font_dialog)`,
     `self.controls.workers_slider.valueChanged.connect(self._update_parameters)`,
     `self.controller.save_progress.connect(self.set_progress)`,
     `self.controller.busy_changed.connect(self.header.set_busy)`. The other connections
     (`parameters_changed`, `glyph_selected` to `controller.select_glyph`, `font_loaded`,
     `preview_ready`, `save_finished`, `error`) stay as they are.
2. Progress in the status bar. New public methods with docstrings:
   `set_progress(self, completed: int, total: int) -> None` (`progress_bar.setRange(0, total)`,
   `setValue(completed)`, `show()`) and `reset_progress(self) -> None` (`progress_bar.reset()`,
   `hide()`). `_on_save_finished` and `_on_error` call `self.reset_progress()` in place of
   `self.controls.reset_progress()`.
3. Font loaded. In `_on_font_loaded` replace the `controls.set_font_info` and
   `controls.set_font_loaded` calls with
   `self.header.set_font_info(session.path.name, f"{session.font_format} · {session.units_per_em} UPM · {session.glyph_count} glyphs · {len(session.island_glyphs)} with islands · {len(session.composites)} composites")`
   and `self.header.set_font_loaded(True)`, and add `self.grid_stack.setCurrentWidget(self.grid)`.
   Keep `self.grid.set_glyphs(session.display_glyphs, session.ascender, session.descender)` and
   `self.grid.select_glyph(session.display_names[0])` exactly in that inline shape (existing wiring
   checks match them). Imports: add `HeaderBar` (`from stencilizer.gui.header import HeaderBar`),
   `QFrame`, `QLabel`, `QProgressBar`, `QStackedWidget`.

## Done

`.venv/bin/mypy src/stencilizer/gui/main_window.py && .venv/bin/ruff check src/stencilizer/gui/main_window.py && .venv/bin/ruff format --check src/stencilizer/gui/main_window.py`
exits 0. Every function is at most 50 effective lines (`_build_panes` is the one at risk: move the
right pane into its own `_build_preview_pane(self) -> QWidget` if it grows past 40). The file stays
under 400 lines.
