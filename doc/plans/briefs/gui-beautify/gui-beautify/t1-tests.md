# T1: Window tests for the new layout (loom-software-engineer, sonnet, wave 2)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first, then `header.py` and
`controls.py` as they stand after wave 1. For `MainWindow` use the surface in `_shared.md`
(`header`, `grid_stack`, `empty_state`, `progress_bar`, `set_progress`, `reset_progress`): U5
rewrites `main_window.py` in parallel with you, to exactly that surface.

**Owns:** `tests/gui/test_main_window.py`, `tests/gui/test_main_window_directions.py`,
`tests/gui/test_main_window_layout.py` (new).
Never edit `tests/gui/test_beautify_contracts.py` (frozen contracts), `tests/gui/test_app.py`
(D1), `tests/gui/test_controls.py` (U2), `tests/gui/test_header.py` (U1) or
`tests/gui/test_glyph_grid.py` (U3).

## Rules for moved assertions

The file actions moved from `ControlPanel` to `HeaderBar` and `MainWindow`. Retarget each affected
assertion with its meaning intact: same check, new owner. Never weaken one (no dropped assert, no
`>=` where `==` stood). The stage files one integrity dispute listing the edited assertion lines,
so report every one as `file:old line -> file:new line`.

## Tasks

1. `tests/gui/test_main_window.py`, retarget only: `window.controls.save_button` and
   `window.controls.open_button` become `window.header.save_button` / `window.header.open_button`
   (lines 75, 219, 324-325, 328-329); line 330 becomes
   `assert not window.progress_bar.isVisibleTo(window)`; `window.controls.workers_spin` becomes
   `window.controls.workers_slider` (lines 127, 149, 257, 319, 338, 362); rename
   `test_workers_spin_updates_controller_parameters` to
   `test_workers_slider_updates_controller_parameters`. Nothing else changes, and the file stays
   under 400 lines (it is 364), which is why new window tests go to a new file.
2. `tests/gui/test_main_window_directions.py`: line 72 reads
   `window.header.font_details_label.text()`; line 122 `workers_spin` becomes `workers_slider`.
3. `tests/gui/test_main_window_layout.py` (new; module docstring
   `"""Tests for the window's header, empty state and status-bar progress."""`; copy the `window`
   fixture and the `_load_font` helper from `test_main_window.py:22-36`):
   - `test_progress_can_be_shown_and_reset`, ported from the base `test_controls.py`:
     `window.set_progress(3, 10)`; `window.progress_bar.isVisibleTo(window)`, `maximum() == 10`,
     `value() == 3`; `window.reset_progress()`; `not window.progress_bar.isVisibleTo(window)`.
   - `test_empty_state_until_font_loads`: `window.grid_stack.currentWidget() is window.empty_state`
     before loading, `is window.grid` after `_load_font(window, qtbot, roboto_path)`.
   - `test_header_describes_loaded_font`: after loading Roboto,
     `window.header.font_name_label.text() == "Roboto-Regular.ttf"`, the details text contains
     `"UPM"` and `"465 composites"`, and `window.header.save_button.isEnabled()`.
   - `test_progress_lives_in_status_bar`: `window.controller.save_progress.emit(3, 10)` shows the
     bar (visible to the window, value 3, maximum 10) and
     `window.statusBar().isAncestorOf(window.progress_bar)`; then
     `window.controller.save_finished.emit(object())` hides it.
   - `test_header_open_button_opens_dialog`: monkeypatch `QFileDialog.getOpenFileName` with a
     recorder returning `("", "")`; `window.show()`;
     `qtbot.mouseClick(window.header.open_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]`;
     exactly one recorded call and `window.controller.session is None`.

Test-writing traps from past stages: quote `cast()` types (ruff TC006); prefix unused stub
parameters with `_`; give every stub and helper a docstring; `GuiController(tmp_path / "gui.log")`
only.

## Done

Your one check, run once:
`uv run ruff check tests/gui && uv run ruff format --check tests/gui && uv run pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui`
exits 0. Report the retargeted-assertion list. The orchestrator runs the tests once U5 lands.
