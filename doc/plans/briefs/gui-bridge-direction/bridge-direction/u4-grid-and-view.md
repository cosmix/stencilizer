# U4: grid markers and the "no bridge" preview label, with tests (gpt-6-luna)

Read `_shared.md` in this directory first ("gui/glyph_grid.py additions").

## Files owned

- `src/stencilizer/gui/glyph_grid.py`
- `src/stencilizer/gui/glyph_view.py`
- `tests/gui/test_grid_marks.py` (new; under 400 lines)

Read-only: `src/stencilizer/config/settings.py` (`BridgeDirection`),
`src/stencilizer/gui/session.py` (`PreviewResult`), `tests/gui/test_glyph_grid.py` (how glyphs
are loaded: `_load_glyphs`), `tests/gui/test_glyph_view.py` (how a `PreviewResult` is built).

## Steps

1. `glyph_grid.py`: module constants `UNBRIDGED_ROLE = Qt.ItemDataRole.UserRole + 1` and
   `BASE_TOOLTIP_ROLE = Qt.ItemDataRole.UserRole + 2`. `set_glyphs` also stores its tooltip
   under `BASE_TOOLTIP_ROLE` and `False` under `UNBRIDGED_ROLE`; item text stays the glyph name.
   Add `set_direction_marker(name, direction)`: find the item whose `UserRole` data is `name` (as
   `select_glyph` does) and set its text to `name` (AUTO), `f"{name} ↕"` (VERTICAL) or
   `f"{name} ↔"` (HORIZONTAL); unknown names are ignored. Add `set_unbridged(names)`: for EVERY
   item set `UNBRIDGED_ROLE` to whether its name is in `names`; a marked item gets the tooltip
   `base + "\nNo bridge could be placed"` and the foreground `QColor("#c62828")`; an unmarked item
   gets back its base tooltip and the default foreground
   (`item.setData(Qt.ItemDataRole.ForegroundRole, None)`). Item text is untouched.
2. `glyph_view.py`, `ComparisonView._preview_text`: when `result.stenciled is not None` and
   `result.bridges_added == 0`, return `f"{name}{code} - no bridge could be placed,
   {result.duration_ms:.1f} ms"`; the other cases keep their current text.
3. Write `tests/gui/test_grid_marks.py`:
   - `test_grid_direction_marker`: after `set_glyphs` with O, B, eight,
     `set_direction_marker("O", HORIZONTAL)` gives text "O ↔", VERTICAL "O ↕", AUTO "O"; the
     `UserRole` data stays "O"; an unknown name changes nothing.
   - `test_grid_marks_unbridged_glyphs`: `set_unbridged({"B"})` sets `UNBRIDGED_ROLE` True on B
     only and appends "No bridge could be placed" to B's tooltip; `set_unbridged(set())` clears
     the role and restores the base tooltip; a direction marker set before survives both calls.
   - `test_preview_text_reports_no_bridge`: a `PreviewResult` with `stenciled` set and
     `bridges_added == 0` puts "no bridge could be placed" in `info_label`; with
     `bridges_added == 1` it shows "1 island(s) bridged".

## Proof (run once, report the output; never run the tests)

    .venv/bin/ruff check src/stencilizer/gui/glyph_grid.py src/stencilizer/gui/glyph_view.py tests/gui/test_grid_marks.py && .venv/bin/mypy src/stencilizer/gui/glyph_grid.py src/stencilizer/gui/glyph_view.py tests/gui/test_grid_marks.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_grid_marks.py
