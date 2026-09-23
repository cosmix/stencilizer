# W2: outline.py (wave 1, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for `outline.py`
is binding.

## Files owned

- `src/stencilizer/gui/outline.py`
- `tests/gui/test_outline.py`

Read-only anchors: `_update_truetype_glyph` and `_update_cff_glyph` in
`src/stencilizer/io/converter.py` (the point walk to mirror), `_recording_to_contours` in the
same file (how points are produced), `Contour.bounding_box` in
`src/stencilizer/domain/contour.py` (returns `(min_x, min_y, max_x, max_y)`),
`fontTools/pens/qtPen.py` in `.venv` (QtPen over BasePen, which expands TrueType implied
on-curve points).

## Steps

1. `draw_glyph(glyph, pen)`: for each contour with points, `pen.moveTo(first)`, then walk the
   remaining points as the two writers do, WITHOUT the CFF reversal: `ON_CURVE` -> `lineTo`; a
   run of `OFF_CURVE_QUAD` points plus the point after the run -> one `qCurveTo(*run)`;
   `OFF_CURVE_CUBIC` at index i with `i + 2 < len(points)` -> `curveTo(p[i], p[i+1], p[i+2])`
   and advance 3; anything else advances 1. End each contour with `closePath()`. Rendering what
   the writers would write is the point: do not "fix" the walk.
2. `glyph_path(glyph)`: `path = QPainterPath()`, `path.setFillRule(Qt.FillRule.WindingFill)`,
   `draw_glyph(glyph, QtPen(None, path=path))`, return `path`. Winding fill is required: fonts
   rasterize nonzero, and a hole whose winding broke during surgery must show filled (odd-even
   fill would hide exactly that defect). `glyph_frame(glyph, ascender, descender)`: over all
   non-empty contours' bounding boxes take x0 = min(0, min_x), x1 = max(advance_width, max_x),
   y0 = min(descender, min_y), y1 = max(ascender, max_y); a glyph without points uses
   0..advance_width and descender..ascender; return `QRectF(x0, y0, x1 - x0, y1 - y0)` in font
   units (y up).
3. `font_to_widget_transform(frame, target)`: s = min(target.width() / frame.width(),
   target.height() / frame.height()); `t = QTransform()`; `t.translate(target.center().x(),
   target.center().y())`; `t.scale(s, -s)`; `t.translate(-frame.center().x(),
   -frame.center().y())`. Guard a zero-width or zero-height frame by returning the identity
   transform. `render_glyph_image(glyph, frame, size, foreground, background)`: a
   `QImage(size, size, QImage.Format.Format_ARGB32)` filled with `background`, painted with
   antialiasing, transform `font_to_widget_transform(frame, QRectF(2, 2, size - 4, size - 4))`,
   `fillPath(glyph_path(glyph), foreground)`; end the painter before returning.

## Tests (`tests/gui/test_outline.py`, use the `qapp` fixture for anything painting)

- TrueType round trip, exact: for `O`, `B`, `eight`, `a`, `g`, `at` of Roboto (non-composite
  glyphs; composites such as `Aring` record `addComponent` and are excluded),
  `draw_glyph(reader.get_glyph(name), RecordingPen())` produces a `.value` equal to recording
  `reader.font.getGlyphSet()[name].draw(RecordingPen())`. Measured: equal for all six.
- CFF bounds: for CommitMono `O` and `B`, `glyph_path(...).boundingRect()` equals the fontTools
  `BoundsPen` bounds (measured: `(30, -10, 570, 710)` and `(85, 0, 546, 700)`).
- `test_winding_fill_shows_broken_hole`: a 100x100 outer square wound clockwise (points (0,0),(0,100),(100,100),(100,0)) plus
  a 30..70 inner square. Rendered at 32 px with white background and black foreground, the
  centre pixel is white when the inner square winds counter-clockwise and black when it winds
  clockwise. Measured: odd-even fill gives white for both, so this test fails a wrong fill rule.
- Transform landmarks with frame `QRectF(0, -500, 1000, 2500)` and target `QRectF(0, 0, 200,
  100)`: `frame.center()` maps to `(100, 50)`; `(500, frame.bottom())` (highest font y) maps to
  y 0; `(500, frame.top())` maps to y 100.
- `glyph_frame` on Roboto `O` spans x from 0 to at least the advance width and y from -555 to
  2146; a synthetic glyph whose points reach y 2500 extends the frame top to 2500.

## Proof command

```bash
.venv/bin/mypy src/stencilizer/gui/outline.py tests/gui/test_outline.py && .venv/bin/ruff check src/stencilizer/gui/outline.py tests/gui/test_outline.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_outline.py
```
