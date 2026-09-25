# U2: ControlPanel sidebar, workers slider, and its tests (codex gpt-5.6-terra, wave 1)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first.

**Owns:** `src/stencilizer/gui/controls.py`, `tests/gui/test_controls.py`.
**Start from:** `ControlPanel` in `src/stencilizer/gui/controls.py` (155 lines at base). Keep
`_BRIDGE_WIDTH_TOOLTIP`, `bridge_config`, `_sync_width_spin`, `_emit_parameters_changed` and the
width slider, width spin and spanning check exactly as they are.

## Steps

1. Move the file actions out and turn the worker count into a slider.
   - Remove what moves to `HeaderBar` (U1) and `MainWindow` (U5): the signals `open_requested` and
     `save_requested`; the widgets `open_button`, `font_info_label`, `save_button`, `progress_bar`;
     the attributes `_font_loaded`, `_busy`; the methods `_emit_open_requested`,
     `_emit_save_requested`, `set_font_info`, `set_font_loaded`, `set_busy`, `set_progress`,
     `reset_progress`; the imports only they used (`QProgressBar`, `QPushButton`, `QSpinBox` stays
     for `width_spin`). Module docstring: `"""Sidebar with the bridge and processing parameters."""`;
     class docstring: `"""Bridge and processing parameters for the preview and the save."""`.
   - Replace `workers_spin`: `self.workers_slider = QSlider(Qt.Orientation.Horizontal, self)`,
     `setRange(0, os.cpu_count() or 1)`, `setValue(0)`, `setPageStep(1)`,
     `setToolTip(_WORKERS_TOOLTIP)` with the module constant
     `_WORKERS_TOOLTIP = "Worker processes used when saving; Auto lets the processor decide"`;
     `self.workers_value_label = QLabel("Auto", self)`, `setProperty("role", "value")`,
     `setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)`. In
     `_connect_signals` add `self.workers_slider.valueChanged.connect(self._sync_workers_label)`. The
     slider must NOT emit `parameters_changed`: MainWindow connects `valueChanged` itself. New
     `_sync_workers_label(self, value: int) -> None`: text `"Auto"` when `value == 0`, else
     `str(value)`. `max_workers()` reads `self.workers_slider.value()` (0 still gives `None`).
2. Rebuild the layout as a styled sidebar. In `__init__`, after `super().__init__(parent)`:
   `self.setObjectName("sidebar")`, `self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)`,
   `self.setMinimumWidth(240)`, `self.setMaximumWidth(360)`. `_build_layout`:
   `layout = QVBoxLayout(self)`, `setContentsMargins(16, 16, 16, 16)`, `setSpacing(10)`; add
   `_section_title("BRIDGES", self)`, `self._bridges_card()`, `layout.addSpacing(6)`,
   `_section_title("PROCESSING", self)`, `self._processing_card()`, `layout.addStretch()`.
   - Module function `_section_title(text: str, parent: QWidget) -> QLabel`: a label with
     `setProperty("role", "sectionTitle")`.
   - `_bridges_card(self) -> QFrame`: `card = QFrame(self)`, `card.setProperty("role", "card")`,
     `QVBoxLayout(card)` margins `(12, 12, 12, 12)` spacing 8; first row a `QHBoxLayout` with
     `QLabel("Width")` (`setBuddy(self.width_spin)`), `addStretch()`, `self.width_spin`; then
     `self.width_slider`; then `self.spanning_check`.
   - `_processing_card(self) -> QFrame`: the same card shape; first row `QLabel("Workers")`
     (`setBuddy(self.workers_slider)`), `addStretch()`, `self.workers_value_label`; then
     `self.workers_slider`; then `QLabel("Parallel processes used when saving the font")` with
     `setProperty("role", "hint")` and `setWordWrap(True)`.
   Add `QFrame` to the `PySide6.QtWidgets` import. Every function stays at most 50 effective lines
   (a codex-written `ControlPanel.__init__` once reached 59 and failed the structure test).
3. `tests/gui/test_controls.py`:
   - `test_control_panel_defaults`: replace the two asserts on `save_button` and `progress_bar`
     (lines 15-16) with `assert panel.workers_slider.value() == 0`,
     `assert panel.workers_value_label.text() == "Auto"`, `assert panel.max_workers() is None`;
     update its docstring to `"""The panel starts with default parameters and automatic workers."""`.
   - Delete `test_set_font_info_updates_label`, `test_busy_state_restores_save_availability`,
     `test_open_and_save_buttons_emit_requests` and `test_progress_can_be_shown_and_reset`: U1 ports
     the header ones to `tests/gui/test_header.py`, T1 ports the progress one to
     `tests/gui/test_main_window_layout.py`.
   - `test_worker_limit_does_not_emit_parameter_change`: `panel.workers_spin` becomes
     `panel.workers_slider`; nothing else changes.
   - Add `test_workers_label_follows_slider`: `top = panel.workers_slider.maximum()`;
     `assert top == (os.cpu_count() or 1)`; `setValue(top)` gives `workers_value_label.text() == str(top)`
     and `max_workers() == top`; `setValue(0)` gives `"Auto"` and `None`.
   Remove imports the deletions orphan (`Qt`, `QtBot` if unused); add `import os`.

## Done

Your one check:
`.venv/bin/mypy src/stencilizer/gui/controls.py tests/gui/test_controls.py && .venv/bin/ruff check src/stencilizer/gui/controls.py tests/gui/test_controls.py && .venv/bin/ruff format --check src/stencilizer/gui/controls.py tests/gui/test_controls.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_controls.py`
exits 0. `main_window.py` still references the removed members until U5 rewrites it in wave 2;
do not edit it. mypy on `main_window.py` fails until then: run mypy on your two files only.
