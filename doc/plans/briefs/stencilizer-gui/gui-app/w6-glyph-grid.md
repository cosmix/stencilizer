# W6: glyph_grid.py (wave 2, codex gpt-6-luna)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for
`glyph_grid.py` is binding. `outline.py` was written in wave 1: read it with
`cat src/stencilizer/gui/outline.py` (the source graph does not show it).

## Files owned

- `src/stencilizer/gui/glyph_grid.py`
- `tests/gui/test_glyph_grid.py`

Read-only anchors: `render_glyph_image` and `glyph_frame` in `src/stencilizer/gui/outline.py`;
`Glyph.name` and `GlyphMetadata.unicode` in `src/stencilizer/domain/glyph.py`.

## Steps

1. `GlyphGrid.__init__`: `setViewMode(QListView.ViewMode.IconMode)`,
   `setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))`,
   `setResizeMode(QListView.ResizeMode.Adjust)`, `setMovement(QListView.Movement.Static)`,
   `setUniformItemSizes(True)`, `setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)`.
   Connect `currentItemChanged` to a method that emits `glyph_selected` with the new item's
   `Qt.ItemDataRole.UserRole` data when the new item is not None.
2. `set_glyphs(glyphs, ascender, descender)`: `clear()`, then per glyph one `QListWidgetItem`
   with text = glyph name, `UserRole` data = glyph name, tooltip = name plus `f" U+{code:04X}"`
   when the glyph has a unicode value, icon = `QIcon(QPixmap.fromImage(render_glyph_image(glyph,
   glyph_frame(glyph, ascender, descender), THUMBNAIL_SIZE, palette().text().color(),
   palette().base().color())))`. Keep the input order.
3. `select_glyph(name)`: find the item whose `UserRole` data equals `name`, make it current
   (this emits `glyph_selected` through step 1), return True; return False when absent.

## Tests (`tests/gui/test_glyph_grid.py`, `qtbot`)

- Load `O`, `B`, `eight` from Roboto with `FontReader` (the `roboto_path` fixture) and call
  `set_glyphs(glyphs, 2146, -555)`: `count() == 3`, item texts in input order, every item's icon
  is not null, the `O` tooltip contains `"U+004F"`.
- `select_glyph("B")` returns True and emits `glyph_selected` with `"B"`
  (`qtbot.waitSignal(..., check_params_cb=...)` or collect emissions).
- `select_glyph("missing")` returns False and emits nothing.
- A second `set_glyphs` call with one glyph replaces the contents (`count() == 1`).
- Keyboard: after `set_glyphs` with `O`, `B`, `eight` and `select_glyph("O")`,
  `qtbot.keyClick(grid, Qt.Key.Key_Right)` emits `glyph_selected` with `"B"`.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/glyph_grid.py tests/gui/test_glyph_grid.py && .venv/bin/ruff check src/stencilizer/gui/glyph_grid.py tests/gui/test_glyph_grid.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_glyph_grid.py
```
