# W1: TrueType gvar writer (stage variable-writers)

Tier: sonnet (`loom-software-engineer`). Never run git.

You own:

- `src/stencilizer/variable/write_gvar.py`
- `tests/unit/test_write_gvar.py`

Read-only:

- `src/stencilizer/variable/model.py`, `transform.py`, `reader.py`, `rounding.py` (merged);
- the frozen `tests/unit/test_variable_writer_contracts.py`.

W3 calls your function from `FontWriter.update_variable_glyph`, and only for glyphs the engine bridged. Untouched glyphs keep their original glyf and gvar data byte for byte: never rebuild a glyph's tuples from `vg.deltas()` unless you are writing that glyph (a glyph whose supports share a peak cannot even be solved).

## Pinned signature

```python
def write_truetype_variable_glyph(font: TTFont, vg: VariableGlyph) -> None
```

## Facts checked while planning (fontTools 4.66.0)

- `font["glyf"]._getCoordinatesAndControls(name, hMetrics, vMetrics=None, *, round=otRound)` returns (coordinates including the 4 phantom points, controls); `controls[1]` is `endPts` without the phantoms. `calcInferredDeltas` and IUP take those `endPts` (`varLib/iup.py:151`).
- `TupleVariation.optimize(origCoords, endPts, tolerance=0.5, isComposite=False)` does nothing on a tuple that already holds `None` entries; a `None` phantom delta means (0, 0) (`iup.py:103-105`).
- `gvar.variations` is a lazy dict. Read every original tuple you need **before** replacing glyf: lazy decoding uses the current glyf point count (`_g_v_a_r.py:171-189`).
- fontTools draws a glyf glyph at offset `hmtx lsb - glyph.xMin` (`ttLib/ttGlyphSet.py:240-247`), and phantom pp1 is `xMin - lsb`. The domain glyph you get from the engine already includes that offset, which is 0 in all three fixtures.
- Domain points do not map 1:1 to glyf points: the converter rotates each contour to its first on-curve point, appends a copy of point 0 when the last segment is a curve, and prepends an implied midpoint when a contour has no on-curve point. Store every domain point as written (Ubuntu `o`: 34 stored, 34 read back); `TTGlyphPen` drops the closing copy in `closePath`, so never use it.

## Steps

0. **Round first.** `r = round_variable_glyph(vg)` (`variable/rounding.py`, engine stage). It is idempotent, so a glyph from `transform_variable_glyph`, which the engine already rounded and validated, is stored exactly as validated; an untransformed glyph read from a font is rounded by the same rule. Use `r` everywhere below. **No gvar table:** a font with fvar and no gvar is legal (`tests/gui/conftest.py` fixture `variable_font_path`). Then `r.supports` must be `()` (raise `ValueError` otherwise); do step 2 only and never create a gvar table.

1. **Capture before replacing anything.** Record the old glyph's `xMin`, the hmtx entry `(advance, lsb)`, and, for each original `TupleVariation` in `font["gvar"].variations.get(name, [])`: deepcopy it, call `calcInferredDeltas(origCoords, endPts)` with `origCoords, controls = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)` and `endPts = controls[1]`, and keep the last 4 coordinates (phantom deltas, `None` → (0, 0)) keyed by the full sorted `(tag, start, peak, end)` tuple that `reader.py` builds.

2. **New glyf glyph.**
   - Build a `fontTools.ttLib.tables._g_l_y_f.Glyph` from `r.default` directly. Every domain point becomes a stored point, so point indices equal `r.deltas()` indices.
   - `coordinates = GlyphCoordinates([(otRound(x), otRound(y)), ...])`; `flags = array("B", ...)` with bit 0 set for ON_CURVE and clear for OFF_CURVE_QUAD, other bits 0; `endPtsOfContours`, `numberOfContours`, and `program = ttProgram.Program()` with `fromBytecode(b"")`.
   - Assign `font["glyf"][name] = glyph`, then `glyph.recalcBounds(font["glyf"])`.
   - Keep pp1 fixed so the copied phantom deltas stay valid: `font["hmtx"][name] = (advance, lsb + glyph.xMin - old_xMin)`. Planning measured a 3-unit xMin shift with hmtx untouched moving the outline by 3.0-3.44 units at every location. HVAR in all three fixtures has advance-width data only (LsbMap/RsbMap are None), so it stays valid.
   - The engine flattens modified glyphs, so `r.default` is all ON_CURVE in practice. Still support OFF_CURVE_QUAD; OFF_CURVE_CUBIC raises `ValueError`.

3. **New tuples from the rounded glyph.** `RD_k = otRound(r.deltas()[k])` per point and coordinate (integral within 1e-9 already; `round_variable_glyph` rounded them against each other because rounding each solved delta on its own measured 2.0 units at Ubuntu `{wdth: -1, wght: ±1}`, where corner tuples add up). Never re-round from `vg` here: the engine validated `r`'s values.
   - Build `TupleVariation({tag: (start, peak, end)}, RD_k + phantom_k)` per support, where `phantom_k` is the step-1 phantom deltas for that support or four (0, 0).
   - Call `tv.optimize(new_coords, new_end_pts, tolerance=0.0)` with `new_coords, controls = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)` after step 2. Never a positive tolerance: inferred deltas may then differ by up to the tolerance between points of one bridge line, which reopens a hairline gap (planning: 1.2-1.47 units of error at tolerance 0.5).
   - Replace `font["gvar"].variations[name]` with the list, in the original support order.

   This guarantees 0.5 units at every support peak; every Ubuntu validation location is a peak.

4. Touch no other glyph, no advance width, HVAR, MVAR, avar, STAT or fvar.

Split the work into named helpers (capture, glyph build, rounding, tuple build); a single function would pass the 50-line limit.

## Tests (`tests/unit/test_write_gvar.py`)

Use `tests/fixtures/variable/Ubuntu-VF-subset.ttf`; resolve glyphs by `getBestCmap()`. Compare outlines through `fonttools_glyph_to_domain(name, font.getGlyphSet(location=loc, normalized=True)[name], font)` point by point, never through raw `RecordingPen` values (their structure differs from domain points).

- Writing an untransformed `read_variable_glyph(font, "o")` back and saving to `tmp_path` reproduces the outline at every support peak within 0.5 units.
- Phantom deltas survive: after save and reload, each output tuple's `calcInferredDeltas` result has the same last 4 coordinates as the input tuple with the same support. Do not compare `.width`: with HVAR present the glyph set takes the width from HVAR, so a writer that zeroes phantom deltas still passes a width check.
- A glyph whose xMin moves gets its hmtx lsb moved by the same amount.
- Instructions are dropped for the rewritten glyph only.
- An fvar-only font (Roboto plus fvar, built as the `variable_font_path` fixture does): a constant glyph (`supports=()`) writes and saves, and the output has no `gvar`; a glyph with supports raises `ValueError` there.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_write_gvar.py
```

The contract tests need W3's `FontWriter.update_variable_glyph`; the orchestrator runs them after both return.
