# Bridge algorithm
> Island detection, bridge placement, contour surgery, multi-island cases

## Island detection

`GlyphAnalyzer.analyze()` (src/stencilizer/core/analyzer.py:94) classifies contours by signed area: `area < 0` (clockwise) is outer, positive (counter-clockwise) is a hole, under TrueType convention (sign definition: `signed_area`, core/geometry_polygon.py:8, re-exported by core/geometry.py). `_build_nesting_tree` (analyzer.py:230) picks the smallest-bbox containing parent, so nested outers such as the R inside ® get their own subtree. An island is an inner contour whose ON_CURVE points (off-curve handles ignored on purpose) all lie inside an outer contour (`_is_island`, analyzer.py:202).

## Candidate placement

The live placement logic is in `ContourMerger` and its axis-generic contour builders. `GlyphTransformer.transform()` computes width from `BridgeConfig.width_percent` and a reference stroke of 10% of UPM. The unused candidate scorer and rectangular geometry generator were removed with their domain types and tests.

## Contour surgery

`ContourMerger.merge_contours_with_bridges` (core/merger.py:12, checks in merger_checks.py, orientation choice in merger_dispatch.py) cuts notches into the outer contour that reach the inner contour instead of adding extra hole contours, which avoids black rendering artifacts. It measures stroke on all four sides, checks real edge crossings (`find_edge_crossing`, core/geometry_crossings.py:124) and clear paths, and picks the thinner-stroke orientation unless the asymmetry rule (ratio > 2.5) or multi-island grouping forces the other.

`GlyphTransformer.transform()` (core/surgery.py:32) returns the glyph unchanged without islands, otherwise builds a `SurgeryContext` and runs `process_groups` then `process_nested`, then copies unprocessed contours. Correction: an earlier entry placed the protecting, grouping and gap tests in `transform()` itself (and cited surgery.py:42); they live in `core/surgery_groups.py` (`protected_indices` for nested-outer descendants, `group_islands` by parent, `arrangement` choosing vertical, horizontal or single from the bounding-box gaps) and `core/surgery_nested.py` (`process_nested`, nested children and inverted islands).

## Multi-island cases

- Vertically stacked islands (Θ-like): horizontal bars with the same winding as the outer contour are structural, not obstructions, and are split with the rest (`has_spanning_obstruction`, `merge_multi_island_vertical` in core/multi_island.py; tests/unit/test_surgery.py).
- Filled encircled digits (⑧): three winding levels, outer CW → digit hole CCW → inverted islands CW inside the hole; the inverted islands need bridges to the hole boundary (tests/unit/test_glyph_transformer.py).

## Axis duplication

Horizontal and vertical bridges share one implementation parameterized by `Axis` (core/axis.py: `coord`, `cross`, `point`, bbox accessors; `VERTICAL` = bridge line at fixed x). Generic code: bridge_contours.py (`create_bridge_contours_for_axis`), bridge_portions.py (`build_outer_portion`, `build_inner_portion`), bridge_segments.py, bridge_nested.py, multi_island_merge.py (`merge_multi_island_axis`), multi_island_obstruction.py, multi_island_outer.py, multi_island_portions.py. horizontal_bridge.py, vertical_bridge.py, multi_island.py and horizontal_multi_island.py are thin wrappers keeping the public names. Real asymmetries are explicit parameters, e.g. `detect_internal_holes` (true only for horizontal outer portions). tests/regression/test_code_structure.py fails if axis-mirrored copies appear.

## UPM scaling

Every absolute font-unit threshold is defined at 1000 UPM in `GeometryConfig` (src/stencilizer/config/settings.py:20) and scaled with `GeometryConfig.scaled(field, upm)` = value * upm / 1000; dimensionless ratios (1.5, 2.5, 3, 0.9, percents) stay literal because they multiply already-scaled lengths. `upm` flows from `process_glyph` into `GlyphTransformer.transform(glyph, upm)` and down through the merger, bridge and multi-island helpers via keyword params (`epsilon`, `edge_margin`, `min_gap`, tolerances). tests/regression/test_upm_scaling.py requires bit-exact output under 2x and 0.5x scaling; tests/regression/test_behavior_golden.py pins 1000-UPM output against the pre-refactor behavior.
