# W1: width-scaling replay engine (stage width-scaling)

Tier: opus, effort xhigh (`loom-senior-software-engineer`). Never run git. Read `_shared.md` in this directory first: it pins the formula, the engine rules (pairing, centre and extent, ink, ratio, placement, fallback chain) and the HEAD measurements. The plan (`doc/plans/IN_PROGRESS-PLAN-variable-bridge-width.md`) wins over this brief.

You own:

- `src/stencilizer/variable/bridge_width.py` (new)
- `src/stencilizer/variable/realign.py` (new: `_realign`, `_project` and their helpers move there from `replay.py`, which is at 397 lines)
- `src/stencilizer/variable/replay.py`
- `src/stencilizer/variable/transform.py`
- `src/stencilizer/variable/validate.py`
- `tests/unit/test_width_scaling.py` (new)
- `tests/unit/test_variable_replay.py`
- `tests/unit/test_variable_transform.py`
- `tests/unit/test_variable_validate.py`

Read-only:

- `src/stencilizer/variable/align.py`, `rounding.py`, `model.py`, `overlaps.py`, `flatten.py`
- `src/stencilizer/core/**`
- `src/stencilizer/config/settings.py` (F0 wrote the new fields)
- `tests/unit/_variable_cases.py`, `tests/font_helpers.py`
- the frozen `tests/unit/test_width_scaling_contracts.py`: it drives your code through `transform_variable_glyph`

## Facts you build on (verified at HEAD; reread the lines before relying on them)

- Pipeline: `variable/transform.py` `_bridged` (L35-67) runs core surgery on the merged default, then `map_surgery` → `align_to_lines` → `replay` per master → `round_variable_glyph` → `validate`. `replay` receives no config today. `BridgeConfig` reaches only the default surgery (`GlyphTransformer`).
- `replay.py`: `_place` (L281-295) puts each cut at its default edge parameter. `_realign` (L361-372) sets each line's master target to the mean of its cuts and moves every member onto the nearest crossing via `_crossing` (L308-334), searching up to `_SEARCH_EDGES = 12` edges. `_project` (L337-358) snaps wrong-side neighbours. `replay` (L375-397) loops over `smap.lines`. `_SEARCH_EDGES` is at L23 and `_contour_spans` at L101-107. The file is at 397 lines: move `_realign`, `_project` and their helpers into the new `variable/realign.py` and import them back; do not grow `replay.py`.
- `BridgeLine(axis, coordinate, members)` and `LineMember(slot, edges, cut)`: nothing records which two lines form one bridge. `validate._facing_pairs` (validate.py:81-97) returns every same-axis pair whose cross spans overlap, a superset of the true pairs (6 for the 4 lines of B, `eight` and `ampersand` with horizontal bridges; 91 for the 14 lines of Inter `.notdef`). It stays the validation rule, unchanged. Do not reuse it for pairing: implement the disjoint pairing rule (`_shared.md` rule 1: same set of input contours, overlapping cross span, walking upward from the lowest line) in `bridge_width.py`.
- `transform.py:57-58` gives `merged.default` and `merged.masters`; `transform.py:94` catches only `VariationDataError`, so a replay spy or call with a fifth argument escapes as `TypeError`.
- In the default output the two lines of a bridge sit exactly `bridge_width` apart (`core/bridge_contours.py:130-131`: `center ± half_width`). Ubuntu o 262.5/322.5, Ubuntu 8 252/312, Inter B 615.56/738.44. Exceptions are the multi-island clamp and nested gaps (see `_shared.md`), which is why `base` comes from the default line coordinates, not from `width_percent`.
- With the default config, spanning bridges and multi-counter glyphs (8, B) give ONE pair of lines, with cut points from every input contour on both lines (horizontal bridges on Inter B give two pairs). A member's input contour is `spans[member.edges[0]]` (`_contour_spans`, L101-107). Inter B slot (5,31) is a `cut=False` member.
- `align_to_lines` (align.py:27-44) replays the default through itself, calling `replay` with 4 arguments at align.py:37, and rejects any point that moves more than `snap + 1e-3`. Align stays on mean targets; do not change `align.py`.
- `validate._bridges_intact` requires facing lines to keep their default order (sign) at every validation location (validate.py:113-115). A positive gap preserves it.
- Rounding (`rounding.py`) needs every member of one line on one coordinate before rounding; that already holds.

## Public surface (pinned; the plan's `reachable` check names `scaled_gap`)

In `src/stencilizer/variable/bridge_width.py`:

- `scaled_gap(base: float, ratio: float, bridge: BridgeConfig, upm: int) -> float`: the formula in `_shared.md`, exactly. Fixed mode returns `base`. The name and signature are pinned.
- Pairing and ink functions (names are your choice): the disjoint pairs of a `SurgeryMap` (rule 1), and the ink along a centre line through a polygon glyph, clipped to an extent (rules 2-3). A zero default ink gives ratio 1; a master ink of 0 gives ratio 0 (rule 4).

`transform_variable_glyph(vg, bridge, geometry, upm)` keeps its signature. `process_variable_glyph` keeps its signature, and the new fields reach it through `BridgeConfig(**config_dict)`.

## Pins (existing tests and spies; none of these may change except the one sanctioned edit below)

- `replay(smap, input_default, output_default, master_input)` keeps four positional parameters and returns today's mean-target result when `smap` carries no width rule. Carry the pairs and the gap rule in a new optional `SurgeryMap` field (default `None`). `SurgeryMap((row,), (line,))` at `test_variable_align.py:84` must keep working.
- `test_variable_align.py:20-29` imports `BridgeLine`, `EdgePoint`, `LineMember`, `SurgeryMap`, `Vertex`, `map_surgery`, `replay` and `slot_values` from `replay.py`: keep them importable there.
- `transform.py` builds `dataclasses.replace(smap, ...)` per attempt and calls `replay`, `round_variable_glyph`, `validate` and `map_surgery` through its own module names. Tests monkeypatch `transform.replay`, `validate`, `map_surgery`, `round_variable_glyph`, `flatten_compatible` and `remove_overlaps_compatible` (`test_variable_transform.py:59,95`; `test_variable_replay.py:166,176`): keep those names bound in `transform.py`.
- `_bridged(vg, merged, bridge, geometry, upm)` stays ONE attempt (a keyword-only `step` parameter is allowed). The fallback chain loops outside it in `transform.py`.
- `test_variable_validate.py:59-78` (a 4-argument `replay` spy, one replay per master) and `test_variable_replay.py:90-102` (the 4-argument `replay` returns today's mean target) pass unchanged.
- The sanctioned test change, exact, in `tests/unit/test_variable_transform.py` `test_replay_failing_in_the_last_master_writes_no_replayed_master` (L74-92): the stub uses `last = len(results) % len(vg.masters) == len(vg.masters) - 1`; the assertions become `assert len(results) == 2 * len(vg.masters)` and `assert all(glyph is not None for i, glyph in enumerate(results) if i % len(vg.masters) != len(vg.masters) - 1)`; `_assert_unchanged(...)` stays. That is the only assertion change allowed. Report it, so the orchestrator files `dispute-integrity`.

## Steps

1. Before editing anything, run `env QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_variable_session.py::test_cold_preview_time_per_island_glyph --no-cov -q -s -p no:cacheprovider` and note the `worst cold preview` time it prints.
2. Write `scaled_gap`, the pairing and the ink measurement in `bridge_width.py`, with unit tests in `tests/unit/test_width_scaling.py`:
   - `scaled_gap` at strength 0, 50 and 100, the minimum clamp, and the minimum never above base;
   - `ValidationError` for `BridgeConfig(scaling_strength=101)` and `BridgeConfig(min_width_percent=9)`;
   - ink on the square ring, and on a glyph with a curved outer (Inter `o` from `tests.font_helpers.INTER`): vertical-bridge ratios 0.27 +/- 0.03 at Thin and 1.88 +/- 0.05 at Black;
   - pairing on Inter `B` with `direction=BridgeDirection.HORIZONTAL` returns exactly 2 disjoint pairs, each 122.88 apart in the default; in fixed mode each pair's lines are 122.88 +/- 1 apart in every master;
   - `test_fixture_glyphs_keep_baseline_outcome`: per island glyph of the three fixtures, in both modes, the same `bridge_count`/`unbridged_count` as the plan's per-glyph Baseline table (embed the table), and every glyph bridged in fixed mode is bridged in proportional mode;
   - `test_spanning_pairs_share_contour_set`: Inter `eight` with `use_spanning_bridges=True` and `direction=BridgeDirection.HORIZONTAL`: disjoint pairs, same input-contour set per pair, fixed-mode gap within 1 of the default gap in every master (the plan's REQUIRED W1 TESTS give the exact wording).
3. Move `_realign`, `_project` and their helpers into `realign.py`. Give `replay` per-pair targets through the new optional `SurgeryMap` field; lines that do not pair keep today's mean target, and a `smap` without the field replays exactly as today.
4. Wire `transform.py`: compute each master's ratios from `merged` (the flattened, overlap-removed masters) and run the fallback chain (rule 6) in a loop outside `_bridged`, in its order. Keep `transform.py` and every function within the size limits; helpers belong in `bridge_width.py`.
5. Update `tests/unit/test_variable_replay.py` only where a signature changed. In `test_variable_transform.py` make only the sanctioned change (Pins), and add two new tests: `test_replay_failing_once_retries_next_step` (the sanctioned test's stub failing the last master on the first attempt only: `bridge_count >= 1`, the glyph changed, `len(results) == 2 * len(vg.masters)`) and `test_validation_failure_retries_next_step` (wrap `transform.validate` so its first call returns False and later calls delegate: the glyph is bridged and validate ran twice). `test_variable_validate.py` stays unchanged. Keep every other assertion: test integrity flags removed or changed assertion lines.
6. Measure and report:
   - for each fixture in `tests/fixtures/variable/` (Inter, Ubuntu, Cantarell) and each mode (fixed, and proportional at strength 100): the glyphs bridged and how many went down each fallback step, against the Baseline;
   - the Inter `o` gap at Thin, default and Black;
   - the worst cold preview time from step 1, before and after (run the same command again at the end).

   Fixed mode must bridge at least as many glyphs as HEAD did (the fallback guarantees it). Record the numbers with `loom memory note`.

## Traps

- Thin-master gaps get small. A gap at or below rounding noise (about 1 unit) can collapse after integer rounding and fail `_bridges_intact`. The minimum clamp keeps the gap above that unless `min_width_percent` is tiny. Let validation reject it and the fallback recover.
- Wider gaps in bold masters can miss the counter (`test_bridge_line_missing_a_narrowed_counter_gives_no_replay`). That is what the fallback exists for; do not widen `_SEARCH_EDGES` to compensate.
- Do not touch `core/`: the default surgery and every static path must stay byte for byte unchanged (`tests/regression` goldens pin static output).
- The integration-verify stage no longer greps `transform.py` for `scaled_gap`. The reachable check (`scaled_gap` reachable from `transform_variable_glyph`) is what must hold: some call path from `transform_variable_glyph` has to end in `scaled_gap`.

## Done when

The engine's share of the plan's width-scaling acceptance lines passes, plus the code-structure test:

```bash
uv run pytest tests/unit/test_width_scaling_contracts.py tests/unit/test_width_scaling.py tests/unit/test_variable_replay.py tests/unit/test_variable_transform.py tests/unit/test_variable_validate.py tests/unit/test_variable_align.py tests/unit/test_variable_engine_contracts.py tests/regression/test_code_structure.py --no-cov -q -p no:cacheprovider
```

Ruff check, ruff format --check and mypy pass on your files. The contract tests that need W3's CLI flags may still fail when you finish; say which ones in your report.
