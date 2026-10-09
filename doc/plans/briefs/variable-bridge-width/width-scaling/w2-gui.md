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
- `tests/gui/conftest.py`, `tests/gui/test_variable_gui_contracts.py` (the window fixture, `_load_window` pattern and save pattern), `tests/gui/test_beautify_contracts.py` (the recorder and layout patterns)

## Facts (verified at HEAD)

- `ControlPanel` (`controls.py`, 173 lines) builds the BRIDGES card in `_bridges_card` (L98-115): a Width row (spin box), `width_slider` (30-110, default 60) and `spanning_check`. `_sync_width_spin` (L152-157) makes the spin/slider pair emit `parameters_changed` once per change; mirror that pairing. `bridge_config()` (L163-168) builds the `BridgeConfig`. `_build_widgets` (L48-77) is at 29 effective lines of 50.
- `tests/gui/test_controls.py:14` asserts `panel.bridge_config() == BridgeConfig()`, so the new widgets' defaults must equal the settings defaults (fixed, 100, 30).
- The axes card shows only for variable fonts: `AxisPanel.set_axes` (axis_controls.py:89-111) toggles visibility and emits `axes_changed(bool)`. `MainWindow._on_font_loaded` (main_window.py:233-247) calls `self.controls.axes_panel.set_axes(session.axes)` at L237 and selects the glyph at L246. `session.axes` is a tuple, `()` for static fonts (session.py:196-198).
- `MainWindow` wires `controls.parameters_changed` to `_update_parameters` (connect L130, def L214-216), which hands `controls.bridge_config()` to the controller. The variable preview cache keys on `bridge.model_dump_json()` (`variable_session.py:182-185`), so the new fields refresh the preview with no change outside your files.
- `main_window.py` is 297 lines (limit 400). The controls are added to the splitter at main_window.py:75-77; the window minimum is 960x600 (L53).
- The theme already styles `QSlider`, `QSpinBox` and `QComboBox`, including `:disabled` (theme.py:129-205), and dims disabled labels through the palette (theme.py:64-68). No theme edit.

## Public surface (pinned; the frozen GUI contracts use these names)

On `ControlPanel`:

- `scaling_box: QWidget`: a container for every width-scaling widget, hidden until `set_variable(True)` (call `hide()` on it explicitly at construction).
- `scaling_combo: QComboBox`: items "Fixed" and "Proportional", in that order, with item data `BridgeWidthScaling.FIXED` and `BridgeWidthScaling.PROPORTIONAL`.
- `strength_slider: QSlider` and `strength_spin: QSpinBox`: range 0-100, default 100, suffix " %".
- `min_width_slider: QSlider` and `min_width_spin: QSpinBox`: range 10-110, default 30, suffix " %".
- `set_variable(variable: bool) -> None`:
  - `set_variable(True)` only shows `scaling_box`.
  - `set_variable(False)` hides `scaling_box` and, when `scaling_combo.currentIndex() != 0`, calls `setCurrentIndex(0)`. That emits `parameters_changed` once through the normal slot. Strength and minimum keep their values.

The strength and minimum rows (labels included) are enabled only while "Proportional" is selected. `bridge_config()` returns all three new fields, the mode built as `BridgeWidthScaling(self.scaling_combo.currentData())`: a `StrEnum` stored as `QComboBox` item data comes back as a plain `str` (PySide6 6.11.2). Do NOT make `bridge_config()` return FIXED while the box is hidden: the frozen contract builds a fresh `ControlPanel` (hidden box) and expects PROPORTIONAL back after selecting it.

## Vertical budget

At 1280x800 with Inter loaded the sidebar has 61 px of slack (`controls` `minimumSizeHint` height 660 against an actual 721). A stacked width-scaling group raises the minimum to 833 and one-line rows to 773, so the rows would render crushed and overlapping. The sidebar is a plain layout with no scroll area (main_window.py:75-77).

Wrap the controls in a `QScrollArea` in `main_window.py` (`setWidgetResizable(True)`, no frame, no horizontal scroll bar), or otherwise keep every control at its minimum height. If you use the scroll area, check that its background matches the sidebar in both themes (render step 4).

## Steps

1. Build the width-scaling group inside the BRIDGES card, below the width row and above `spanning_check`, using the existing theme's widgets and spacing (`_section_title`, the row layout of `_bridges_card`). Labels: "Width scaling", "Strength", "Minimum". Tooltips:
   - strength: "How strongly bridge gaps follow stroke thickness: 0 keeps the width fixed, 100 is fully proportional"
   - minimum: "Smallest bridge gap in light masters, as percent of a reference stroke of 10% of font UPM"
   - combo: "Fixed: the same gap in every master. Proportional: gaps follow the weight of each master. The default master is the same in both modes."

   Every change emits `parameters_changed` exactly once. Function limit: build the group in a separate `_build_scaling_widgets` method, and share one slider/spin sync helper for the three slider/spin pairs (width, strength, minimum) instead of a third copy of `_sync_width_spin`. Call `scaling_box.hide()` explicitly at construction. Disable the strength and minimum labels together with their rows (the theme dims disabled labels). Call `setBuddy` on the new labels as the Width row does (controls.py:108).
2. In `MainWindow._on_font_loaded`, call `self.controls.set_variable(bool(session.axes))` beside `set_axes` (L237), before the glyph selection (L246). Emitting there is safe: `controller._selected_glyph` is `None` (controller.py:99). Add the scroll area from the vertical budget.
3. Tests in `tests/gui/test_width_scaling_controls.py`, in addition to these four:
   - defaults equal `BridgeConfig()`;
   - selecting Proportional enables the two rows and emits once;
   - a strength or minimum change emits once and reaches `bridge_config()`;
   - Fixed disables the rows again.

   Add:
   - `set_variable(False)` after Proportional gives FIXED from `bridge_config()` with exactly one `parameters_changed` emit;
   - `strength_slider.isVisibleTo(panel)` and `min_width_slider.isVisibleTo(panel)` follow `set_variable`;
   - a recorder test following `tests/gui/test_beautify_contracts.py:174-194`: monkeypatch `window.controller.set_parameters`, select Proportional, and assert the recorded `BridgeConfig` has `width_scaling` PROPORTIONAL, strength 100.0 and minimum 30.0;
   - a layout test following `test_beautify_contracts.py:160-172`: show `MainWindow` at 1280x800 with Inter loaded and assert `window.controls.height() >= window.controls.minimumSizeHint().height()`;
   - a save test following `tests/gui/test_variable_gui_contracts.py:115-125` (`controller.save` with `qtbot.waitSignal(controller.save_finished, timeout=SAVE_TIMEOUT)`): save Inter once in Fixed and once in Proportional; glyph `o` is identical at `{}` and different at `{"wght": 1.0}` (`tests.font_helpers.glyph_at(...).to_dict()`). This test needs W1's engine; if W1 is not in yet, report it as pending.

   Keep `test_controls.py` green; edit it only if a widget it reads moved.
4. Render and look at the result (Read the PNG). It must match the card's existing visual language. Write `screen.json` and the PNGs under the session scratchpad directory named in your system prompt (not `$TMPDIR`, not the worktree). Offscreen 2x grabs need a large screen config: `QT_QPA_PLATFORM="offscreen:configfile=<scratchpad>/screen.json" QT_SCALE_FACTOR=2` with `{"screens":[{"name":"s","x":0,"y":0,"width":3840,"height":2160,"logicalDpi":96,"logicalBaseDpi":96}]}`. Render `MainWindow` at 1280x800 with `tests/fixtures/variable/Inter-VF-subset.ttf` loaded and Proportional selected, once with the light and once with the dark theme (`apply_theme`), then once with Fixed selected to see the disabled rows.

If a GUI test fails inside `src/stencilizer/variable/`, W1 may still be editing it. Report it, re-run once, and never edit `variable/`.

## Done when

```bash
env QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_width_scaling_gui_contracts.py tests/gui/test_width_scaling_controls.py tests/gui/test_controls.py tests/gui/test_main_window.py --no-cov -q -p no:cacheprovider
uv run pytest tests/regression/test_code_structure.py --no-cov -q -p no:cacheprovider
```

Both pass, along with ruff and mypy on your files.
