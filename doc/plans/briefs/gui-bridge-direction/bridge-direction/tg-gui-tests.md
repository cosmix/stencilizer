# TG: session, controller and main-window tests (loom-software-engineer, sonnet)

Read `_shared.md` in this directory first ("Measured facts" above all). You write three test
files against the session, controller and main window already in the worktree. Do not edit any
module under `src/`: report a defect you find instead of fixing it.

## Files owned

- `tests/gui/test_session_directions.py` (new)
- `tests/gui/test_controller_directions.py` (new)
- `tests/gui/test_main_window_directions.py` (new)

Each under 400 lines. Read-only: `src/stencilizer/gui/session.py`, `controller.py`,
`main_window.py`, `composites.py`, `glyph_grid.py`; `tests/gui/conftest.py` (`processor`,
`roboto_path`, `commit_mono_path`, `FIXTURES_DIR`, `controller`, `load_session`,
`build_settings`, `staging_root`, `outlines_match`); `tests/gui/test_session.py`,
`tests/gui/test_controller.py` and `tests/gui/test_main_window.py` for the fixture, save and
signal-waiting style; doc/loom/knowledge/conventions.md "GUI tests".

"No contour spans centre y" below means: no contour of the glyph has a bbox with
`ymin < 728 < ymax` (Roboto O's input centre y; its centre x is 703.5).

## tests/gui/test_session_directions.py

1. `test_display_glyphs_include_bridged_composites`: Roboto: `len(display_glyphs) == 1027`,
   `len(island_glyphs) == 562`, display names follow the font's glyph order, `Aacute` present,
   `direction_sources` gives `("O",)` for O, `("A",)` for Aacute, `("A", "ring")` for Aring,
   `()` for space; CommitMono: display names equal the 467 island glyph names.
2. `test_preview_applies_glyph_direction`: `preview("O", BridgeConfig(), geometry,
   {"O": HORIZONTAL})` equals `preview("O", BridgeConfig(direction=HORIZONTAL), geometry)`
   contour for contour and differs from the AUTO preview; a direction stored for another glyph
   leaves O unchanged.
3. `test_composite_preview_follows_base_direction`: with `{"A": HORIZONTAL}` the stenciled
   Aacute contains every stenciled contour of `preview("A", ..., {"A": HORIZONTAL})` (A is drawn
   with the identity transform; compare `to_dict()`), its `bridges_added` equals A's, it differs
   from the AUTO Aacute, and `preview("Aacute", ...).original is session.glyph("Aacute")`.
4. `test_unbridged_lists_glyphs_without_bridges`: Roboto default `unbridged(...)` equals the 14
   names in `_shared.md`; with `{"A": HORIZONTAL}` it still excludes A and Aacute.
5. `test_saved_font_changes_only_displayed_glyphs` (the regression test for the reported bug),
   parametrized over Roboto and `FIXTURES_DIR / "Lato-Black.ttf"`: save with default settings into
   `tmp_path` (`error_count == 0`); for every glyph name draw input and output with
   `DecomposingRecordingPen` on each font's glyph set; the glyphs whose recordings differ are a
   subset of `display_names`, and every display name whose recording is unchanged is in
   `unbridged(...)`.
6. `test_save_applies_directions`: save Roboto with `directions={"O": HORIZONTAL}`; the saved O
   has no contour spanning centre y, and the saved Oacute decomposed through the saved font's
   glyph set has none either (it references O).

## tests/gui/test_controller_directions.py

1. `test_set_direction_refreshes_preview`: load Roboto, `select_glyph("O")`, then
   `set_direction("O", HORIZONTAL)` inside `qtbot.waitSignal(controller.preview_ready)`: the new
   preview differs from the previous one; `direction_changed` fired with ("O", "horizontal");
   `direction_for("O")` is HORIZONTAL; setting AUTO restores AUTO.
2. `test_set_direction_rejects_composite`: `set_direction("Aacute", VERTICAL)` emits `error`
   containing "follows A" and leaves A and Aacute at AUTO.
3. `test_font_load_resets_directions`: a direction set on Roboto is gone after loading it again.
4. `test_survey_reports_unbridged_glyphs`: after loading Roboto,
   `qtbot.waitSignal(controller.unbridged_changed, timeout=10_000)` delivers a frozenset holding
   "four" and "AEacute" and not "O"; `busy_changed` emissions are only the load's True, False.
5. `test_stale_survey_result_is_dropped`: with a session loaded, calling
   `controller._on_survey_finished((controller._survey_generation - 1, frozenset({"O"})))` emits
   nothing (`qtbot.assertNotEmitted(controller.unbridged_changed)`); the current generation is
   emitted.
6. `test_save_uses_directions`: HORIZONTAL on O, save through the controller into `tmp_path`,
   wait for `save_finished` (`error_count == 0`); the saved O has no contour spanning centre y.

## tests/gui/test_main_window_directions.py

1. `test_grid_lists_bridged_composites`: after loading Roboto the grid has 1027 items including
   `Aacute`, and the font info label mentions "465 composites".
2. `test_choosing_direction_updates_preview_and_marker`: select O through the grid, choose the
   "horizontal" item in `window.direction_picker.combo` with `setCurrentIndex`: the after-canvas
   glyph has no contour spanning centre y, the grid item for O reads "O ↔", and
   `controller.direction_for("O")` is HORIZONTAL.
3. `test_composite_selection_shows_following_picker`: set A to VERTICAL, select Aacute: the combo
   is disabled, shows "vertical", and the label reads "Follows A".
4. `test_unbridged_glyphs_marked_in_grid`: after `controller.unbridged_changed`, the item for
   "four" has `UNBRIDGED_ROLE` True and O's False.
5. `test_saved_font_uses_chosen_direction`: HORIZONTAL for O, `window.save_font(tmp_path /
   "out.ttf")`, wait for `save_finished` (`error_count == 0`): the saved O has no contour spanning
   centre y, and the saved Oacute's `RecordingPen` recording still holds an `addComponent` of "O".

## Check (run once at the end, report the output)

    uv run ruff check tests/gui/test_session_directions.py tests/gui/test_controller_directions.py tests/gui/test_main_window_directions.py && uv run mypy tests/gui/test_session_directions.py tests/gui/test_controller_directions.py tests/gui/test_main_window_directions.py && uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_session_directions.py tests/gui/test_controller_directions.py tests/gui/test_main_window_directions.py
