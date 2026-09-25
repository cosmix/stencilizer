# Bridge algorithm
> Island detection, bridge placement, contour surgery, multi-island cases

## Island detection

GlyphAnalyzer flattens curves with UPM-scaled tolerance before measuring signed area and containment. Parents are the smallest strictly enclosing contours; touching or crossing edges are rejected. The nesting tree is reused for island classification and guarded against cycles.

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

GeometryConfig thresholds and adaptive curve-flattening tolerance scale with UPM. Scale regression tests require matching normalized geometry. Behavior goldens now pin the corrected curve-aware geometry, regenerated after analytical regression tests and visual inspection.

## Bridge direction

`BridgeConfig.direction` (`BridgeDirection` AUTO / VERTICAL / HORIZONTAL, config/settings.py:76) is per glyph: the GUI builds one config per previewed or saved glyph and `FontProcessor.process(directions=...)` overrides it by glyph name. `SurgeryContext.direction` (core/surgery_context.py) reaches the merge as follows:

| Case | AUTO | Explicit D |
| --- | --- | --- |
| Single island, nested child, inverted island | `MergeDispatch.preferred()` picks the axis | `SurgeryContext.merge` forces D through `force_horizontal`/`force_vertical`; `MergeDispatch.forced_*` falls back to the other axis when D cannot be built |
| Island group, arrangement equals D | spanning iff `use_spanning_bridges` | spanning always tried, sequential if it fails |
| Island group, arrangement differs from D | as AUTO | sequential only |

The group rule is `_spanning_allowed` (core/surgery_groups.py:133); `_split_child` is unchanged. A glyph unbuildable on both axes stays unbridged, with no further fallback. Auto output is pinned bit for bit by tests/regression.

## Truthful bridge counts

process_glyph reports bridges_added from accepted surgery operations through TransformOutcome. unbridged_count records unresolved hole contours, including partially successful glyphs. No-op transformations preserve the original outlines. _islands_bridged remains available for compatibility with direction tests.

## Open issue: CommitMono `.notdef` under Auto

The CommitMono `.notdef` defect is tracked in concerns.md. For the synthetic filled encircled digit (tests/unit/test_glyph_transformer.py:161) `_process_inverted` (core/surgery_nested.py) is never reached at any width or direction: the hole merges with the outer as one island and that merge marks both inverted bowls processed first, so `test_inverted_islands_follow_direction` asserts the force flags of the outer merge.

## Bridge contour parameter grouping

BridgeRequest carries bridge construction parameters throughout the contour helper pipeline, with inner and outer contours explicit. BridgeSide is a frozen dataclass with named line, crossing-list, and lower-side fields. The axis wrapper derives hole detection from the axis; the general entry point preserves its scalar signature and explicit override as a compatibility adapter.

## Curve-aware geometry

Geometry uses adaptive de Casteljau subdivision with a maximum control-point distance to the finite chord segment of 0.25 font units at 1000 UPM, scaled by UPM. Finite-segment distance preserves collinear overshoot; tolerance is positive and finite and subdivision depth is bounded. A no-op returns the original glyph. Successfully modified contours use the flattened working geometry; untouched contours retain their original curves.

## Containment and bridge outcomes

Contour parents require proper boundary containment; intersecting or touching outlines are not parents. Island containment reuses the nesting tree and selects the closest enclosing filled contour. Accepted surgery operations record connected source contours, while unbridged islands are reported separately from operational failures.

## Corrected geometry regression baseline

Golden snapshots are refreshed for adaptive curve geometry and cyclic serialization after analytical curve tests, all 45 real-font integration tests, and all 15 UPM-scaling checks passed. Visual inspection covered 27 representative original/output pairs across Roboto, Lato, and CommitMono. Glyph membership is unchanged; normalized geometry changes for 500 Roboto, 434 Lato, and 397 CommitMono glyphs, and 397 glyphs in the full CommitMono pipeline. Modified curved outlines are line approximations; snapshots grow because they record the additional vertices.

## Preserve CFF preview vertices

CFF command specialization can remove vertices that become collinear after integer rounding. Disable getCharString optimization for rewritten glyphs so saved outlines preserve preview point structure; keep normal coordinate rounding. This costs some output size and is covered by the GUI CFF round-trip regression.
