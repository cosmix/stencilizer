# W3: compatible overlap removal (stage variable-engine, wave 2)

Tier: opus (`loom-senior-software-engineer`, effort high). Never run git.

You own:

- `src/stencilizer/variable/overlaps.py`
- `tests/unit/test_variable_overlaps.py`

Read-only:

- `src/stencilizer/variable/model.py` and `flatten.py` (W1, merged);
- the frozen `tests/unit/test_variable_engine_contracts.py`.

W2 calls exactly `remove_overlaps_compatible(vg: VariableGlyph) -> VariableGlyph | None`. Its input is already flattened by W1's `flatten_compatible(vg, upm)`: every point is ON_CURVE, and the structure is identical across masters. Build every test input with that real flattener, never a stand-in: its subdivision counts are usually not powers of two, and that is what breaks naive float32 matching (step 2).

Reference spike: `doc/plans/briefs/variable-fonts/spike/ovl.py` (`union`, `umap`, `ureplay`). Read the plan's Evidence section. On the probe set, Inter `A`, `D`, `P`, `R` and `e` went from no island to bridged and valid at 8 of 8 locations.

## Why

Variable fonts keep overlapping contours, so many counters are not separate contours (Inter `P` and `e` are a single self-overlapping contour). The static analyzer then finds no island. Merging the overlaps per master independently would give incompatible structures. Merge once on the default and replay the result.

## Algorithm

1. Union the default with skia-pathops:
   - build a `pathops.Path` from the default's polygons with `path.getPen()` (`moveTo`, `lineTo`, `closePath`);
   - `result = pathops.simplify(path, clockwise=True)`;
   - read the result back through a `RecordingPen`. Only `moveTo`, `lineTo` and `closePath` are expected; anything else returns `None`.
2. **Map each output vertex:**
   - `Vertex(i)`: the nearest input vertex within 1e-3 font units, found with a grid-bucket lookup (cell size 1e-3, check the 3×3 neighbourhood). pathops round-trips coordinates through float32, which is off by up to 1.2e-4 at Inter's 2048 UPM; planning measured 20-23% of union vertices missing a "1e-4 after rounding to 4 decimals" match at 7 or 13 subdivisions per curve, which made every Inter probe glyph unmappable. With 1e-3, 7 and 13 subdivisions give the same results as 8.
   - `Crossing((a, b), (c, d))`: else, exactly two input edges pass within 2e-3 of it.
   - Any other count returns `None`.
3. **Replay in every master and in the default:**
   - a `Vertex` takes that glyph's input vertex;
   - a `Crossing` is the line-line intersection of the same two edges in that glyph;
   - a parallel pair (|denominator| < 1e-12) returns `None`.
   Build `merged_default` the same way, by replaying the map on the default input. Never use pathops' float32 output coordinates: the static surgery and W2's replay map compare coordinates at 1e-6.
4. Check that each replayed master contour keeps the default's orientation sign. A flipped signed area means a crossing moved past an edge end; return `None`.
5. Return `VariableGlyph(merged_default, vg.supports, merged_masters, vg.axis_tags)`. Construction re-checks structure.
6. **Fast path:** when the default has no self-intersection and no pair of contours that intersect, return `vg` unchanged. Ubuntu-style fonts with overlaps already removed take this path. Use `pathops.simplify` and compare: if the output equals the input polygons up to start point and orientation, return `vg`. pathops drops a closing point equal to the first (Ubuntu `o` goes from 97 to 96 points per contour), so remove consecutive duplicate points, including a last point equal to the first, from both sides before comparing; without that the fast path matched 5 of 23 Ubuntu glyphs, with it 22 of 23. Then glyphs without overlaps keep their contour order and start points, and the surgery sees exactly the static font's contours.

Winding: `clockwise=True` gives clockwise outer contours. The repo's internal convention is the TrueType one (outer clockwise; see `core/analyzer.py` `ContourHierarchy`: negative signed area is outer). Verify it on Inter `P` by checking that `GlyphAnalyzer` reports one island after the union.

## Tests (`tests/unit/test_variable_overlaps.py`)

- Inter-VF-subset `P` and `e`: union yields 2 contours, one island at the default, and an identical structure in every master.
- Ubuntu-VF-subset `o`: the fast path returns the same object.
- A synthetic pair of overlapping squares whose overlap shrinks to a touch in a master returns `None` or a valid structure, never an exception.

Resolve glyphs through `getBestCmap()`.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_variable_overlaps.py
```

`test_overlap_built_counter_bridged` also needs W2's `transform.py`; the orchestrator runs it after wave 2. `pathops` ships no type stubs or `py.typed`; import it with `# type: ignore[import-untyped]`. `pathops.simplify` is `simplify(path, fix_winding=True, keep_starting_points=True, clockwise=False)`; `clockwise=True` gives outer contours negative signed area, which is the analyzer's convention. The verifier runs lint, format and mypy. Size limits: files ≤400 lines, functions ≤50 effective lines.
