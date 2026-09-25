# U8: main window wiring (gpt-5.6-terra)

Read `_shared.md` in this directory first.

## Files owned

- `src/stencilizer/gui/main_window.py`

Read-only, in the worktree from earlier waves (read with `cat`):
`src/stencilizer/gui/direction_picker.py` (U5), `src/stencilizer/gui/glyph_grid.py` (U4:
`set_direction_marker`, `set_unbridged`), `src/stencilizer/gui/controller.py` (U7:
`direction_for`, `set_direction`, `direction_changed`, `unbridged_changed`),
`src/stencilizer/gui/session.py` (U6: `display_glyphs`, `direction_sources`, `composites`).

## Steps

1. Layout. Replace `splitter.addWidget(self.comparison)` with a right-hand pane: a `QWidget`
   holding a `QVBoxLayout` of `self.comparison` (stretch 1) and `self.direction_picker =
   DirectionPicker(...)`; add that pane to the splitter. `self.comparison` and every other
   public attribute keep their names and types.
2. Signals, in `__init__` next to the existing connections:
   `self.grid.glyph_selected.connect(self._on_glyph_selected)` (in addition to the existing
   connection to `controller.select_glyph`), `self.direction_picker.direction_chosen.connect(
   self._on_direction_chosen)`, `self.controller.direction_changed.connect(
   self._on_direction_changed)`, `self.controller.unbridged_changed.connect(
   self._on_unbridged_changed)`. Handlers: `_on_glyph_selected(name)` returns when
   `controller.session` is None, else stores `self._current_glyph = name` and calls `direction_picker.show_for(name, sources, controller.direction_for(
   sources[0]) if sources else BridgeDirection.AUTO)` with `sources =
   session.direction_sources(name)`; `_on_direction_chosen(value)` calls
   `controller.set_direction(self._current_glyph, BridgeDirection(value))` when a glyph is
   current; `_on_direction_changed(name, value)` calls `grid.set_direction_marker(name,
   BridgeDirection(value))` and, when `name` is the current glyph, re-runs `_on_glyph_selected`
   so the picker shows the stored choice; `_on_unbridged_changed(names)` calls
   `grid.set_unbridged(names)`.
3. `_on_font_loaded`: fill the grid from `session.display_glyphs` (not `island_glyphs`), call
   `direction_picker.clear()` and reset `self._current_glyph = None` before selecting the first
   display glyph, and set the font info to
   `f"{session.path.name}\n{session.font_format}, {session.units_per_em} UPM\n"
   f"{session.glyph_count} glyphs, {len(session.island_glyphs)} with islands, "
   f"{len(session.composites)} composites using them"`.

`__init__` must stay under 50 lines: move the new connections into a private
`_connect_direction_signals()` if needed.

## Done when

Loading Roboto fills the grid with 1027 items; picking "Horizontal" for `O` changes the preview
and the grid label to "O ↔"; selecting `Aacute` shows a disabled picker reading "Follows A".

## Proof (run once, report the output)

    .venv/bin/ruff check src/stencilizer/gui/main_window.py && .venv/bin/mypy src/stencilizer/gui/main_window.py
