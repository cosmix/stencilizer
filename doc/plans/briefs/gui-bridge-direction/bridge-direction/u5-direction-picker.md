# U5: direction picker widget, with tests (gpt-6-luna)

Read `_shared.md` in this directory first ("gui/direction_picker.py" contract).

## Files owned

- `src/stencilizer/gui/direction_picker.py` (new)
- `tests/gui/test_direction_picker.py` (new; under 400 lines)

Read-only: `src/stencilizer/gui/controls.py` (widget style to mirror: `QSignalBlocker` in
`ControlPanel._sync_width_spin`, docstrings, layout), `src/stencilizer/config/settings.py`,
`tests/gui/test_controls.py` (widget test style).

## Steps

1. `DirectionPicker.__init__`: a `QVBoxLayout` holding `source_label` (`QLabel`) and `combo`
   (`QComboBox`) with three items, text then `userData` = the `BridgeDirection` value string:
   "Auto" / "auto", "Vertical (cuts top and bottom)" / "vertical",
   "Horizontal (cuts left and right)" / "horizontal". Connect `combo.currentIndexChanged` to a
   bound method that emits `direction_chosen` with `str(self.combo.currentData())`. Start in the
   cleared state.
2. `show_for(name, sources, direction)`: select the item whose data equals `direction.value`
   inside a `QSignalBlocker(self.combo)` so nothing is emitted; enable the combo only when
   `sources == (name,)`; label `f"Bridge direction for {name}"` when enabled,
   `f"Follows {', '.join(sources)}"` when `sources` is non-empty but not `(name,)`,
   `"No islands to bridge"` when `sources` is empty. `clear()`: select "Auto" without emitting,
   disable the combo, empty the label.
3. Write `tests/gui/test_direction_picker.py`:
   - `test_picker_emits_only_on_user_change`: `show_for("O", ("O",), VERTICAL)` selects the
     vertical item, enables the combo and emits nothing; `combo.setCurrentIndex` of the
     horizontal item emits exactly `["horizontal"]`.
   - `test_picker_disables_for_composites`: `show_for("Aring", ("A", "ring"), AUTO)` disables the
     combo and reads "Follows A, ring"; `show_for("space", (), AUTO)` reads "No islands to
     bridge"; `clear()` disables it and emits nothing.

## Proof (run once, report the output; never run the tests)

    .venv/bin/ruff check src/stencilizer/gui/direction_picker.py tests/gui/test_direction_picker.py && .venv/bin/mypy src/stencilizer/gui/direction_picker.py tests/gui/test_direction_picker.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_direction_picker.py
