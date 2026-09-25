# U3: Glyph grid cells, palette-aware thumbnails and marks (codex gpt-5.6-terra, wave 1)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first.

**Owns:** `src/stencilizer/gui/glyph_grid.py`, `tests/gui/test_glyph_grid.py`.
**Start from:** `GlyphGrid.__init__`, `set_glyphs` and `set_unbridged` in
`src/stencilizer/gui/glyph_grid.py`; `render_glyph_image` and `glyph_frame` in
`src/stencilizer/gui/outline.py` (read-only). Keep `THUMBNAIL_SIZE`, `UNBRIDGED_ROLE`,
`BASE_TOOLTIP_ROLE`, `glyph_selected`, `set_direction_marker`, `select_glyph` and item texts as
they are: `tests/gui/test_glyph_grid.py` asserts `item.text()` equals the glyph name.

## Why

The theme follows the system light/dark setting while the app runs. Thumbnails are rasterized once
with the palette's text and base colours, so after a switch they keep the old colours; and the
unbridged mark `#c62828` reads at about 2:1 contrast on a dark base.

## Steps

1. Cells, and thumbnails that follow the palette. In `__init__`, directly after
   `super().__init__(parent)`, initialise `self._glyphs: list[Glyph] = []`, `self._ascender = 0`,
   `self._descender = 0`, `self._icon_colors: tuple[QColor, QColor] | None = None` (before any
   other call, so an early `changeEvent` finds them). Then, besides the existing settings:
   `self.setObjectName("glyphGrid")`, `self.setGridSize(QSize(THUMBNAIL_SIZE + 28, THUMBNAIL_SIZE + 30))`,
   `self.setWordWrap(False)`, `self.setTextElideMode(Qt.TextElideMode.ElideRight)`,
   `self.setFrameShape(QFrame.Shape.NoFrame)`.
   `set_glyphs` clears the grid, stores `list(glyphs)`, `ascender`
   and `descender` on the instance, adds one item per glyph exactly as today (text, `UserRole`
   name, tooltip, `BASE_TOOLTIP_ROLE`, `UNBRIDGED_ROLE` False) but without an icon, then calls
   `self._render_icons()`. New `_render_icons(self) -> None`: reads
   `foreground = self.palette().text().color()` and `background = self.palette().base().color()`,
   stores `(foreground, background)` in `self._icon_colors`, and for each `index, glyph` in
   `enumerate(self._glyphs)` sets `self.item(index)`'s icon from
   `render_glyph_image(glyph, glyph_frame(glyph, self._ascender, self._descender), THUMBNAIL_SIZE, foreground, background)`
   (skip a `None` item). New override
   `def changeEvent(self, event: QEvent) -> None:  # noqa: N802`: call `super().changeEvent(event)`;
   when `event.type() == QEvent.Type.PaletteChange`, call `_render_icons()` only if the palette's
   `(text, base)` colours differ from `self._icon_colors`, then call `self._recolor_unbridged()`.
   The literal `QEvent.Type.PaletteChange` must appear in `changeEvent` (the plan's wiring check).
2. Readable marks. Module constants `_UNBRIDGED_ON_LIGHT = "#c62828"` and
   `_UNBRIDGED_ON_DARK = "#ff8a80"`. New `_unbridged_color(self) -> QColor`: `QColor(_UNBRIDGED_ON_LIGHT)`
   when `self.palette().base().color().lightness() >= 128`, else `QColor(_UNBRIDGED_ON_DARK)`.
   `set_unbridged` uses it in place of `QColor("#c62828")` (tests/gui/test_grid_marks.py:69 still
   expects `#c62828` under the default light palette). New `_recolor_unbridged(self) -> None`: for
   every item whose `UNBRIDGED_ROLE` is True, set `ForegroundRole` to `self._unbridged_color()`.
   Imports: add `QEvent` to `PySide6.QtCore` and `QFrame` to `PySide6.QtWidgets`.
3. `tests/gui/test_glyph_grid.py`: add three tests after the existing ones, each building
   `grid = GlyphGrid()` with `qtbot.addWidget(grid)`, glyphs from the existing
   `_load_glyphs(roboto_path, ["O", "B"])` helper, and `grid.set_glyphs(glyphs, 1900, -500)`. The
   dark palette is hand-built (the theme module does not exist yet): `palette = grid.palette()`,
   `palette.setColor(QPalette.ColorRole.Base, QColor("#121418"))`,
   `palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))`, `grid.setPalette(palette)`.
   - `test_thumbnails_rerender_when_palette_changes`: set glyphs under the default palette, switch to
     the dark palette, then
     `grid.item(0).icon().pixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE).toImage().pixelColor(0, 0) == QColor("#121418")`
     (the two-pixel inset of `render_glyph_image` keeps the corner pixel on the background).
   - `test_unbridged_mark_is_light_red_on_dark_base`: dark palette first, then
     `grid.set_unbridged(["O"])`; the "O" item's `ForegroundRole` is `QColor("#ff8a80")`.
   - `test_unbridged_mark_recolours_on_palette_switch`: `grid.set_unbridged(["O"])` under the default
     palette gives `QColor("#c62828")`; after switching to the dark palette "O" gives
     `QColor("#ff8a80")` and "B" keeps `ForegroundRole` `None`.
   Import `THUMBNAIL_SIZE` from `stencilizer.gui.glyph_grid` and `QColor`, `QPalette` from
   `PySide6.QtGui`. Every test gets a docstring; `grid.item(0)` may be `None` to mypy, so assert it
   is not `None` first.

## Done

Your one check:
`.venv/bin/mypy src/stencilizer/gui/glyph_grid.py tests/gui/test_glyph_grid.py && .venv/bin/ruff check src/stencilizer/gui/glyph_grid.py tests/gui/test_glyph_grid.py && .venv/bin/ruff format --check src/stencilizer/gui/glyph_grid.py tests/gui/test_glyph_grid.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_glyph_grid.py`
exits 0. The module stays under 400 lines and every function under 50.
