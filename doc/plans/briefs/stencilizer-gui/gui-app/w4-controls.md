# W4: controls.py (wave 1, codex gpt-6-luna)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for
`controls.py` is binding.

## Files owned

- `src/stencilizer/gui/controls.py`
- `tests/gui/test_controls.py`

Read-only anchors: `BridgeConfig` in `src/stencilizer/config/settings.py` (width bounds 30-110,
default 60; `use_spanning_bridges` default True); the `--bridge-width` help text in
`src/stencilizer/cli/app.py` (reuse its wording for the tooltip).

## Steps

1. Build the widgets in a `QVBoxLayout`: `open_button` ("Open Font..."), `font_info_label`
   (word wrap, initial text "No font loaded"), a "Bridge width" row with `width_slider`
   (horizontal, range 30-110, value 60) and `width_spin` (range 30-110, value 60, suffix " %"),
   `spanning_check` ("Spanning bridges for stacked islands", checked), a "Workers" row with
   `workers_spin` (range 0 to `os.cpu_count() or 1`, value 0, `setSpecialValueText("Auto")`),
   `save_button` ("Stencilize && Save...", disabled), `progress_bar` (hidden), then a stretch.
2. Signals: `open_button.clicked` -> `open_requested`; `save_button.clicked` ->
   `save_requested`. Sync the width pair so each user change emits `parameters_changed` exactly
   ONCE: `width_spin.valueChanged` -> `width_slider.setValue`; `width_slider.valueChanged` ->
   a method that sets `width_spin` inside a `QSignalBlocker(self.width_spin)` and then emits
   `parameters_changed`. `spanning_check.toggled` -> emit `parameters_changed`. The workers spin
   does NOT emit it.
3. Accessors: `bridge_config()` returns `BridgeConfig(width_percent=float(width_slider.value()),
   use_spanning_bridges=spanning_check.isChecked())`; `max_workers()` returns None when the spin
   is 0, else its value. `set_font_info(text)` sets the label; `set_font_loaded(loaded)` enables
   `save_button`; `set_busy(busy)` disables `open_button` and `save_button` while busy and, when
   not busy, re-enables `open_button` and restores `save_button` to the last
   `set_font_loaded` value; `set_progress(completed, total)` shows the bar with range 0..total and
   value completed; `reset_progress()` hides it and resets it.

## Tests (`tests/gui/test_controls.py`, `qtbot`)

- Defaults: `bridge_config() == BridgeConfig()`, `max_workers() is None`, `save_button` disabled,
  `progress_bar` hidden.
- Setting `width_spin` to 80 moves the slider to 80 and emits `parameters_changed` exactly once
  (count emissions with a list-appending callback); setting the slider to 40 moves the spin to 40
  and emits exactly once; `bridge_config().width_percent == 40.0`.
- Unchecking `spanning_check` emits once and gives `use_spanning_bridges is False`.
- `workers_spin.setValue(2)` gives `max_workers() == 2` and emits nothing.
- `set_font_loaded(True)` enables save; `set_busy(True)` disables open and save;
  `set_busy(False)` re-enables both; after `set_font_loaded(False)`, `set_busy(False)` leaves save
  disabled.
- Clicking `open_button` (`qtbot.mouseClick`) emits `open_requested`.
- `set_progress(3, 10)` makes the bar visible (use `isVisibleTo(panel)`) with maximum 10 and
  value 3; `reset_progress()` hides it.

## Proof command

```bash
uv run pytest --no-cov -q tests/gui/test_controls.py && uv run mypy src/stencilizer/gui/controls.py tests/gui/test_controls.py && uv run ruff check src/stencilizer/gui/controls.py tests/gui/test_controls.py
```
