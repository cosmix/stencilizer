# Variable Replay

> Variable-font stencil replay, validation, delta solve, rounding, measured rates

## Replay pipeline

`variable/transform.py` `transform_variable_glyph(vg, bridge, geometry, upm)` stencils one `VariableGlyph` (default `Glyph`, `Support` regions, one full master `Glyph` per support; `variable/model.py`). Steps, each of which can end in an unchanged glyph whose default islands are counted as unbridged:

1. `flatten_compatible` (flatten.py): every master becomes a polygon with one structure; a curve needing more than `MAX_SUBDIVISIONS = 64` pieces raises `VariationDataError`.
2. `remove_overlaps_compatible` (overlaps.py): skia-pathops merges overlaps once on the default, every output vertex is mapped back to an input vertex or the crossing of two input edges, and the map is replayed on each master. Returns None when it cannot.
3. Default surgery: core `GlyphTransformer.transform_with_outcome` on the merged default; zero bridges ends the pipeline as a no-op.
4. `map_surgery(merged.default, outcome.glyph, snap)` (replay.py) records the source of every output point and groups bridge-line points into `BridgeLine`s; `align_to_lines` (align.py) puts every line point exactly on its line in the default; `replay` rebuilds the glyph per master.
5. `round_variable_glyph` (rounding.py) rounds to what the target format stores; `validate` (validate.py) checks the rounded glyph.

`VariationDataError` (singular supports, incompatible masters, unsupported CFF2 `vsindex`) never escapes a glyph: the glyph is written unchanged and counted. `process_variable_glyph` is the picklable worker wrapper; `variable/processing.py` is the `FontProcessor` hook (classification, selection, save) and `core/pool.py` spawns the workers.

## Bridge-line grouping and realignment

Cross-master replay must keep every point of one bridge line on one axis-aligned line in each master. Fixed edge parameters do not (mistakes.md "Fixed-parameter replay breaks bridge-cut coincidence across masters").

- A cut point's line axis comes from its surgery-created output segment (both ends not on one input edge; mostly vertical means an x line); the source-edge axis is the fallback. The edge-axis rule alone put cuts on steep slanted edges on y lines.
- An output vertex that ends a surgery segment joins the nearest same-axis line within `snap_distance(geometry, upm)` (the core's contour gap or point dedup tolerance, scaled by UPM) and sits exactly on it in the default and every master. Core surgery keeps an input vertex in place of a cut point within `bridge_tolerance` of the line (`core/bridge_segments.py:78-100`), which left one cut edge leaning off its line; `align.py` fixes that on the default before replay, and a fixed 0.5-unit snap missed 2048-UPM fonts where offsets reach 0.8.
- `LineMember` carries no offset; `map_surgery` keeps its default `_SNAP_DISTANCE = 0.5` for callers that pass none.
- Per master, `_place` (replay.py:281) puts each cut point at its default edge parameter, then `_realign` (replay.py:361) sets the line's master coordinate to the mean of its cut points' placed positions and moves every member onto the nearest crossing of its edge with that coordinate (`_crossing`, searching up to `_SEARCH_EDGES = 12` edges); `_project` snaps wrong-side neighbours. Each line is placed independently and nothing ties the two lines of one bridge, so the gap between them is not held at the default width in other masters (concerns.md "Variable bridge gaps drift across masters").
- Overlap-union crossings slide along flattened curves from master to master and may pass neighbouring union vertices, which folds the outline into spikes, bow-ties and sliver holes. `variable/crossings.py` collapses those vertices onto a re-intersected crossing. `overlaps.py` accepts crossings past their edge ends (only parallel edges and flipped orientation return None), because a stricter on-segment check rejected Inter A/R/e/x and 14 Cantarell glyphs.
- The overlap result is rejected when XOR area against its own pathops union exceeds `FIDELITY_RELATIVE = 3%` of the union plus `FIDELITY_FLOOR = 1` unit squared (overlaps.py:38-39, 192). Measured: legitimate sliding merges deviate up to 1.92% (Cantarell `a`), a pulled-apart box pair 37%. Slivers up to about 5 units wide are accepted.

## Validation

`validate(vg, upm, allowed_islands, lines=())` runs on the rounded glyph at `validation_locations`: the default, every support peak and a grid over the used axis values (full product up to `MAX_GRID_LOCATIONS = 64`, otherwise each axis end alone plus the all-minimum and all-maximum corners). At each location it requires the analyzer island count and `holes.enclosed_counters` (skia-pathops non-zero union, pinch points split, both contour directions) to stay at or under `allowed_islands`, and `_bridges_intact`. `GlyphAnalyzer` misses counters closed by a bow-tie or a hairline wall, and pathops resolves near-coincident tangles differently by contour direction and treats a hole touching the outer contour at one point inconsistently, hence the both-direction max. `lines` is optional; line order is compared only between same-axis lines whose cut points overlap on the other axis, since bridges of different counters (an `8`) legitimately cross.

## Delta solve and rounding

`solver.solve_deltas` solves `M . D = masters - default` with `M[j][k] = supports[k].scalar(peak_j)` against the ORIGINAL supports, by Gauss-Jordan; a pivot at or under 1e-12 raises `VariationDataError("singular variation supports")`. Solving master minus default independently is wrong where regions overlap. `round_variable_glyph` stores an integer default and integer deltas chosen so each peak lands within 0.5 of the exact outline (`_round_deltas`, the gvar algorithm), for CFF2 as well: fontTools' CFF2 instancer rounds every blended relative operand, so fractional deltas make coincident edges on separate contours drift apart. `write_cff2` re-solves deltas from the rebuilt masters and snaps them with `round_coords` because solver noise (about 1e-13) defeats the operand's zero-delta check. Glyphs the engine leaves alone keep their original glyf/gvar data or charstring byte for byte; a source glyph can already hold half-integer CFF2 blends (Cantarell `four`).

## Measured outcomes

Full system fonts, all validation locations, rounded: Ubuntu[wdth,wght] 209 of 229 island glyphs bridged (6 no static bridge, 13 replay finds no crossing, 1 unmappable); InterVariable 248 of 283 (24 no crossing, 7 rejected by line order or piece orientation, 2 unmappable, 1 island, 1 no static bridge); Cantarell-VF 394 of 413 (13 no static bridge, 6 no crossing). "No crossing" glyphs have counters that close or narrow below the bridge width in a bold or condensed master (oe, percent.osf, registered, superscripts); leaving them unbridged is correct. Worst cold GUI preview over the fixture island glyphs is about 18 ms (ampersand, q, eight).

## Checking output

- `GlyphAnalyzer` uses exact float comparisons (core/analyzer.py:247 strict bbox nesting, :273 `a*b <= 0`), so fontTools runtime-evaluated CFF2 instances with 5e-13 noise show false islands at some weights, and `island_count` on stenciled Cantarell `o` reports 1 at off-grid weights (150, 250, 275, 300) where `enclosed_counters` reports 0. The pipeline validates on `vg.instance`, so output is unaffected; external sweeps must snap coordinates or use `enclosed_counters`.
- `instantiateVariableFont` on stenciled CFF2 output drifts bridge-line points apart by up to about 10 units at intermediate weights (wght 170) and closes counters in `o`, `B`, `D`. Integer deltas make masters and axis extremes exact; the drift between them is inherent to the fontTools CFF2 instancer. gvar fonts instantiate cleanly.
- Tests that monkeypatch names imported into `variable.transform` (`replay`, `validate`, `map_surgery`) take the real function from its defining module; reading it back as `transform.replay` fails mypy `attr-defined`. A `replay()` that returns None through tied vertices needs ties inside one input contour; `_resolve` drops ties across contours (replay.py:157-160).
