# W2: width-scaling controls in the GUI (stage width-scaling)

Tier: sonnet (`loom-software-engineer`). Never run git. Read `_shared.md` in this directory first.

You own:

- `src/stencilizer/gui/controls.py`
- `src/stencilizer/gui/main_window.py`
- `tests/gui/test_width_scaling_controls.py` (new)
- `tests/gui/test_controls.py`

Read-only:

- `src/stencilizer/gui/axis_controls.py`, `gui/theme.py`, `gui/controller.py`, `gui/variable_session.py`, `gui/session.py`
- the frozen `tests/gui/test_width_scaling_gui_contracts.py`
- `tests/gui/conftest.py`, `tests/gui/test_variable_gui_contracts.py` (the window fixture and `_load_window` pattern)

## Facts (verified at HEAD)

- `ControlPanel` (`controls.py`, 169 lines) builds the BRIDGES card in `_bridges_card` (L94-111): a Width row (spin box), `width_slider` (30-110, default 60) and `spanning_check`. `_sync_width_spin` (L136-142) makes the spin/slider pair emit `parameters_changed` once per change; mirror that pairing. `bridge_config()` (L159-164) builds the `BridgeConfig`.
- `tests/gui/test_controls.py:14` asserts `panel.bridge_config() == BridgeConfig()`, so the new widgets' defaults must equal the settings defaults (fixed, 100, 30).
- The axes card shows only for variable fonts: `AxisPanel.set_axes` (axis_controls.py:89-111) toggles visibility and emits `axes_changed(bool)`. `MainWindow._on_font_loaded` (main_window.py:200-215) calls `self.controls.axes_panel.set_axes(session.axes)`; `session.axes` is empty for static fonts.
- `MainWindow` wires `controls.parameters_changed` to `_update_parameters` (L121, L181-183), which hands `controls.bridge_config()` to the controller. The variable preview cache keys on `bridge.model_dump_json()` (`variable_session.py:185`), so the new fields refresh the preview with no change outside your files.

## Public surface (pinned; the frozen GUI contracts use these names)

On `ControlPanel`:

- `scaling_box: QWidget`: a container for every width-scaling widget, hidden until `set_variable(True)`.
- `scaling_combo: QComboBox`: items "Fixed" and "Proportional", in that order, with item data `BridgeWidthScaling.FIXED` and `BridgeWidthScaling.PROPORTIONAL`.
- `strength_slider: QSlider` and `strength_spin: QSpinBox`: range 0-100, default 100, suffix " %".
- `min_width_slider: QSlider` and `min_width_spin: QSpinBox`: range 10-110, default 30, suffix " %".
- `set_variable(variable: bool) -> None`: shows or hides `scaling_box`.

The strength and minimum rows are enabled only while "Proportional" is selected. `bridge_config()` returns all three new fields.

## Steps

1. Build the width-scaling group inside the BRIDGES card, below the width row and above `spanning_check`, using the existing theme's widgets and spacing (`_section_title`, the row layout of `_bridges_card`). Labels: "Width scaling", "Strength", "Minimum". Tooltips:
   - strength: "How strongly bridge gaps follow stroke thickness: 0 keeps the width fixed, 100 is fully proportional"
   - minimum: "Smallest bridge gap in light masters, as percent of a reference stroke of 10% of font UPM"
   - combo: "Fixed: the same gap in every master. Proportional: gaps follow the weight of each master"

   Every change emits `parameters_changed` exactly once.
2. In `MainWindow._on_font_loaded`, call `self.controls.set_variable(bool(session.axes))` beside `set_axes`.
3. Tests in `tests/gui/test_width_scaling_controls.py`:
   - defaults equal `BridgeConfig()`;
   - selecting Proportional enables the two rows and emits once;
   - a strength or minimum change emits once and reaches `bridge_config()`;
   - Fixed disables the rows again.

   Keep `test_controls.py` green; edit it only if a widget it reads moved.
4. Render the panel offscreen in both themes and look at the result (Read the PNG). It must match the card's existing visual language. Offscreen 2x grabs need a large screen config: `QT_QPA_PLATFORM="offscreen:configfile=<scratch>/screen.json" QT_SCALE_FACTOR=2` with `{"screens":[{"name":"s","x":0,"y":0,"width":3840,"height":2160,"logicalDpi":96,"logicalBaseDpi":96}]}`. Write scratch files under `$TMPDIR`, never the worktree.

## Done when

`env QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_width_scaling_gui_contracts.py tests/gui/test_width_scaling_controls.py tests/gui/test_controls.py tests/gui/test_main_window.py --no-cov -q -p no:cacheprovider` passes, along with ruff and mypy on your files.
