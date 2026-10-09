# W2: surgery replay, validation, transform (stage variable-engine, wave 2)

Tier: opus, effort xhigh (`loom-senior-software-engineer`). Never run git.

You own:

- `src/stencilizer/variable/replay.py`
- `src/stencilizer/variable/validate.py`
- `src/stencilizer/variable/transform.py`
- `tests/unit/test_variable_replay.py`

Read-only:

- `src/stencilizer/variable/model.py`, `solver.py`, `reader.py`, `flatten.py` and `rounding.py` (W1, merged before you start; call `flatten_compatible(vg, upm) -> VariableGlyph` and `round_variable_glyph(vg) -> VariableGlyph`);
- `src/stencilizer/variable/overlaps.py` (W3, in parallel; call only `remove_overlaps_compatible(vg) -> VariableGlyph | None`);
- `src/stencilizer/core/**`: never edit it, because `tests/regression` pins the static pipeline;
- the frozen `tests/unit/test_variable_engine_contracts.py`.

Reference spike (evidence only; rewrite to repo standards) in `doc/plans/briefs/variable-fonts/spike/`:

- `vlib.py`: `Flat` pen and `mapping`;
- `vlib3.py`: `replay3`, realignment and projection;
- `vlib4.py`: `lines`, bridge-line grouping by edge axis;
- `run4.py`: the driver.

Read the plan's Evidence section first: fixed-t replay fails (554 of 561 Ubuntu glyphs), realignment works (23 of 561 fail).

## Pipeline: `transform_variable_glyph(vg, bridge, geometry, upm)`

1. `flat = flatten_compatible(vg, upm)` (W1's `variable/flatten.py`).
2. `merged = remove_overlaps_compatible(flat)`. If it returns `None`, return the no-op outcome (counted on the flattened default, below).
3. `hierarchy = GlyphAnalyzer().analyze(merged.default, upm)`. With no islands, return `VariableOutcome(vg, 0, 0)`.
4. Run surgery on the default only:

   ```python
   GlyphTransformer(analyzer=GlyphAnalyzer(), bridge_config=bridge, geometry_config=geometry).transform_with_outcome(merged.default, upm=upm)
   ```

   With `bridge_count == 0`, return `VariableOutcome(vg, 0, outcome.unbridged_count)`. A partially successful outcome (`bridge_count > 0` and `unbridged_count > 0`) is allowed, as in the static pipeline, which writes such glyphs (plan Goals): replay what was bridged.
5. `smap = map_surgery(merged.default, outcome.glyph)`. If it returns `None`, return the no-op outcome.
6. For every master `m` in `merged.masters`, `replay(smap, merged.default, outcome.glyph, m)`. Any `None` gives the no-op outcome.
7. Build `result = round_variable_glyph(VariableGlyph(outcome.glyph, merged.supports, replayed, vg.axis_tags, vg.cff2))`. Rounding comes before validation: the writers store exactly this glyph, so validation proves the saved font.
8. `validate(result, upm, allowed_islands=outcome.unbridged_count)`. On failure, return the no-op outcome.
9. Return `VariableOutcome(result, outcome.bridge_count, outcome.unbridged_count)`.

The **no-op outcome** is `VariableOutcome(vg, 0, n)`, where n is the island count of step 3's hierarchy when step 3 ran; when overlap removal returned `None`, of `GlyphAnalyzer().analyze(flat.default, upm)`; when flattening raised, of `GlyphAnalyzer().analyze(vg.default, upm)` (the analyzer accepts curves). n is never 0 for a glyph with islands: stage `variable-surfaces` adds it to `ProcessingStats.unbridged_count`, and classification counts the same way. The input `vg` is returned unchanged; never return half-bridged output. A `VariationDataError` raised by any step (flattening past 64 subdivisions, a master-structure mismatch when building a `VariableGlyph`, a singular solve) returns the no-op outcome; `transform_variable_glyph` never raises it.

Contours that surgery leaves untouched in a bridged glyph come out flattened, because step 1 flattens every contour; the static pipeline re-appends those contours with their curves (`core/surgery.py:75-77`). Planning measured this on 3 Ubuntu glyphs (uni0221, uni0247, uni2116) and 22 Inter glyphs (Ohorn, ohorn, Q_rthook, ...). It is an accepted non-goal of this plan; do not try to restore curves.

`process_variable_glyph(vg_dict, config_dict, upm, geometry_dict) -> dict` is module-level and picklable. It mirrors `core/processor.py` `process_glyph` (lines 76-104): the same keys `glyph`, `bridges_added`, `unbridged_count`, `duration_ms`, plus `error`, `glyph_name` and `traceback` on exception. `"glyph"` carries `VariableGlyph.to_dict()`. Its positional order differs from `process_glyph` (whose ignored 4th parameter is `reference_stroke_width`); the error dict's `glyph_name` comes from `vg_dict["default"]["metadata"]["name"]`.

## replay.py

**`map_surgery(input_glyph, output_glyph) -> SurgeryMap | None`.** For each output point, record one of:

- `Vertex(index)`: equal to an input vertex. Compare at 1e-6 after rounding to 6 decimals; indices are into the concatenated input point list.
- `EdgePoint(a, b, t)`: on input edge a→b within contour order, distance < 1e-4, t in [0, 1].

If any point is neither, return `None` (the spike found 4 such points in 2 of 563 Ubuntu glyphs). When a coordinate matches several input vertices, prefer the one whose input contour already supplies the neighbouring mapped points of the same output contour. If still ambiguous and the candidates' master positions differ, return `None`.

**Bridge-line grouping.** Put every `EdgePoint` on a line key:

- `("x", round(x, 5))` when its source edge's |dx| ≥ |dy|: a vertical bridge line crosses a mostly-horizontal edge;
- `("y", round(y, 5))` otherwise.

Keep only keys with ≥2 members. A grouping that checks shared x first, regardless of edge direction, mis-grouped `p.sc`: it put cut points on a vertical stem into "vertical lines". That is why the edge-axis rule exists (spike `vlib4.py` `lines`).

**`replay(smap, input_default, output_default, master_input) -> Glyph | None`.** Per master:

1. Compute each `EdgePoint` at fixed t.
2. **Line coordinate per group:** `c_m` is the mean of the group's fixed-t coordinate on its axis.
3. **Re-intersect:** for each group point, search the master polyline of the point's source contour for an edge crossing `c_m`, nearest first, within ±12 edges of the default edge. Set the point's axis coordinate exactly to `c_m`, so collinearity is exact. If no edge crosses, return `None`.
4. **Projection:** walk from each group point along its output contour in both directions over `Vertex` points while the master coordinate lies on the opposite side of `c_m` from the default coordinate's side of the default line. Set that coordinate to `c_m`. Stop at the first point on the correct side, at another group point, or at an `EdgePoint`.
5. Ungrouped `EdgePoint`s keep fixed t. `Vertex` points take the master vertex.

Point types and direction come from `output_default`.

## validate.py

**`validation_locations(vg)`** returns, de-duplicated, in a stable order:

- the default `{}`;
- every support peak;
- the grid: per axis in `vg.axis_tags`, the value set {0.0} ∪ {−1.0 if any support peak on that axis is negative} ∪ {+1.0 if any is positive} ∪ {every intermediate peak coordinate on that axis}. `VariableGlyph` has no fvar access, so the range comes from the supports; Ubuntu's wdth default equals its maximum, so wdth has no +1 there.
- When the full product of those value sets has at most 64 locations, include all of it. Otherwise include, instead of the product, each axis's non-zero values alone (other axes 0), all axes at their minimum value, and all axes at their maximum value. A 13-axis font would otherwise need up to 3^13 analyzer runs per glyph, and the GUI runs this on its thread.

The fixtures stay far under 64 (Ubuntu `o`: wdth {−1, 0} × wght {−1, 0, +1} = 6 locations), so they take the full product.

**`validate(vg, upm, allowed_islands) -> bool`.** For every location, `GlyphAnalyzer().analyze(vg.instance(loc), upm)` must report at most `allowed_islands` islands. Every bridge-line pair must keep its order: for the default's grouped lines, the sign of `c1 - c2` between consecutive lines of the same axis must not flip. A degenerate or zero-width bridge fails. If `vg.deltas()` raises `VariationDataError`, the result is False.

## Tests (`tests/unit/test_variable_replay.py`)

- `map_surgery` round-trips at the default: replaying with the default as master returns the surgery output exactly.
- Grouping uses the edge axis: build a synthetic case with two cut points on one vertical stem edge.
- Projection snaps a wrong-side vertex.
- `validation_locations` on Ubuntu-VF-subset `o` has no wdth +1; a synthetic 8-axis `VariableGlyph` with ±1 supports on every axis gets at most 1 + 16 + 16 + 2 locations, never the 3^8 product.
- A `VariationDataError` from `flatten_compatible` (monkeypatch it to raise) gives the no-op outcome with `unbridged_count == 1` for Ubuntu-VF-subset `o`.
- `remove_overlaps_compatible` monkeypatched to return `None` gives the no-op outcome with `unbridged_count == 1` for Ubuntu `o` (counted on the flattened default, never 0).
- Partial success, deterministic: build a synthetic `VariableGlyph` (one clockwise outer square, two counter-clockwise inner squares far apart, one `wght` support (0, 1, 1) whose master scales the outline by 1.1). Monkeypatch `GlyphTransformer.transform_with_outcome` with a wrapper that runs the real method on the glyph without the second hole, appends the second hole's contour unchanged (as static surgery does for an island it cannot bridge), and returns `TransformOutcome` (`core/surgery.py`) with that glyph, the real `bridge_count` and `unbridged_count=1`. Assert the result has `bridge_count >= 1`, `unbridged_count == 1`, and exactly 1 island at every location in `validation_locations`; and that `validate(result.glyph, upm, allowed_islands=0)` is False.
- `validate` with `allowed_islands`: passes with 1 and fails with 0 on a glyph that keeps one island everywhere.

Fixtures are under `tests/fixtures/variable/`; resolve glyphs by character through `getBestCmap()`.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_variable_replay.py tests/unit/test_variable_engine_contracts.py tests/regression/test_code_structure.py
```

Size limits: files ≤400 lines, functions ≤50 effective lines. Split helpers across your three files rather than growing one. The verifier runs lint, format and mypy.
