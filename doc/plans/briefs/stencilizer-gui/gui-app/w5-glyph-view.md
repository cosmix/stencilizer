# W5: glyph_view.py (wave 2, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for
`glyph_view.py` is binding. `session.py` and `outline.py` were written in wave 1: read them with
`cat src/stencilizer/gui/session.py src/stencilizer/gui/outline.py` (the source graph does not
show them).

## Files owned

- `src/stencilizer/gui/glyph_view.py`
- `tests/gui/test_glyph_view.py`

Read-only anchors: `glyph_path`, `glyph_frame`, `font_to_widget_transform` in
`src/stencilizer/gui/outline.py`; `PreviewResult` and `FontSession` in
`src/stencilizer/gui/session.py`.

## Steps

1. `GlyphCanvas`: `setMinimumSize(160, 160)`; `set_glyph(glyph, frame)` stores both and calls
   `self.update()`; `glyph` and `frame` are read-only properties. `paintEvent`: `QPainter(self)`,
   fill `self.rect()` with `self.palette().base()`; when both glyph and frame are set, enable
   antialiasing, set the transform `font_to_widget_transform(frame,
   QRectF(self.rect()).adjusted(8, 8, -8, -8))`, draw the baseline with a cosmetic pen as
   `painter.drawLine(QPointF(frame.left(), 0.0), QPointF(frame.right(), 0.0))` (the four-number
   `drawLine` overloads take ints, so floats fail mypy strict) in `palette().mid().color()`,
   then `fillPath(glyph_path(glyph), self.palette().text())`. End the painter.
2. `ComparisonView`: a grid layout with the titles "Original" and "Stencilized" over
   `before_canvas` and `after_canvas`, and `info_label` spanning below. `show_preview(result,
   ascender, descender)`: frame = `glyph_frame(result.original, ascender, descender)`, united
   (`QRectF.united`) with the stenciled glyph's frame when `result.stenciled` is not None; set BOTH
   canvases to that same frame so they share one scale. The after canvas gets `result.stenciled`
   (None clears it).
3. Info text: `f"{name}{code} - {n} island(s) bridged, {ms:.1f} ms"` where `code` is
   `f" (U+{unicode:04X})"` when `result.original.metadata.unicode` is set, else empty; on failure
   (`stenciled is None`) `f"{name}: transform failed: {error}"`. `clear()` empties both canvases
   and the label.

## Tests (`tests/gui/test_glyph_view.py`, `qtbot`, fixtures from `tests/gui/conftest.py`)

- Build a `FontSession` from Roboto (`FontSession.open(roboto_path, processor)`) and a
  `PreviewResult` for `O` with default configs.
- `test_show_preview_shares_one_frame`: `show_preview` sets `before_canvas.glyph is
  result.original`, `after_canvas.glyph is result.stenciled`, and `before_canvas.frame ==
  after_canvas.frame`; the info text contains `"O (U+004F)"` and `"1 island(s) bridged"`.
- A synthetic `PreviewResult(stenciled=None, error="boom", ...)` leaves `after_canvas.glyph is
  None` and the info text contains `"transform failed: boom"`.
- Painting: `canvas.resize(200, 200)`, `canvas.set_glyph(o_glyph, frame)`, then
  `canvas.grab().toImage()` contains at least one pixel equal to the palette text colour
  (`canvas.palette().text().color().rgb()`, compare `QColor(image.pixel(x, y)).rgb()`); with
  `set_glyph(None, None)` no pixel does.
- `clear()` empties both canvases and the label.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/glyph_view.py tests/gui/test_glyph_view.py && .venv/bin/ruff check src/stencilizer/gui/glyph_view.py tests/gui/test_glyph_view.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_glyph_view.py
```
