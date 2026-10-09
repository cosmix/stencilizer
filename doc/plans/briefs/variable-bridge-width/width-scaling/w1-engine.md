# W1: width-scaling replay engine (stage width-scaling)

Tier: opus, effort xhigh (`loom-senior-software-engineer`). Never run git. Read `_shared.md` in this directory first: it pins the formula, the fallback chain and the HEAD measurements.

You own:

- `src/stencilizer/variable/bridge_width.py` (new)
- `src/stencilizer/variable/replay.py`
- `src/stencilizer/variable/transform.py`
- `src/stencilizer/variable/validate.py`
- `tests/unit/test_width_scaling.py` (new)
- `tests/unit/test_variable_replay.py`

Read-only:

- `src/stencilizer/variable/align.py`, `rounding.py`, `model.py`, `overlaps.py`, `flatten.py`
- `src/stencilizer/core/**`
- `src/stencilizer/config/settings.py` (F0 wrote the new fields)
- `tests/unit/_variable_cases.py`, `tests/font_helpers.py`
- the frozen `tests/unit/test_width_scaling_contracts.py`: it drives your code through `transform_variable_glyph`

## Facts you build on (verified at HEAD; reread the lines before relying on them)

- Pipeline: `variable/transform.py` `_bridged` (L35-67) runs core surgery on the merged default, then `map_surgery` → `align_to_lines` → `replay` per master → `round_variable_glyph` → `validate`. `replay` receives no config today. `BridgeConfig` reaches only the default surgery (`GlyphTransformer`).
- `replay.py`: `_place` (L281-295) puts each cut at its default edge parameter. `_realign` (L361-372) sets each line's master target to the mean of its cuts and moves every member onto the nearest crossing via `_crossing` (L308-334), searching up to `_SEARCH_EDGES = 12` edges. `_project` (L337-358) snaps wrong-side neighbours. `replay` (L375-397) loops over `smap.lines`. The file is at 397 lines, so move code out (for example `_realign`/`_project` into a new module) rather than grow it.
- `BridgeLine(axis, coordinate, members)` and `LineMember(slot, edges, cut)`: nothing records which two lines form one bridge. `validate._facing_pairs` (validate.py:250-266) pairs same-axis lines whose cross spans overlap; its docstring says the two sides of one bridge always face each other. Reuse that rule instead of writing a second one. Move it into `bridge_width.py` if both modules need it, and keep `validate` behaviour identical.
- In the default output the two lines of a bridge sit exactly `bridge_width` apart (`core/bridge_contours.py:130-131`: `center ± half_width`). Ubuntu o 262.5/322.5, Ubuntu 8 252/312, Inter B 615.56/738.44. Exceptions are the multi-island clamp and nested gaps (see `_shared.md`), which is why `base` comes from the default line coordinates, not from `width_percent`.
- Spanning bridges and multi-counter glyphs (8, B) give ONE pair of lines, with cut points from every input contour on both lines. A member's input contour is `spans[member.edges[0]]` (`_contour_spans`, L101-107). Inter B slot (5,31) is a `cut=False` member.
- `align_to_lines` (align.py:152-169) replays the default through itself and rejects any point that moves more than `snap + 1e-3`. With ratio 1 at the default, the pair targets equal the default coordinates. Keep it that way, or the default stops replaying.
- `validate._bridges_intact` requires facing lines to keep their default order (sign) at every validation location. A positive gap preserves it.
- Rounding (`rounding.py`) needs every member of one line on one coordinate before rounding; that already holds.

## Public surface (pinned; the plan's `reachable` check names `scaled_gap`)

In `src/stencilizer/variable/bridge_width.py`:

- `scaled_gap(base: float, ratio: float, bridge: BridgeConfig, upm: int) -> float`: the formula in `_shared.md`, exactly. Fixed mode returns `base`.
- Pairing: a function that returns the facing line pairs of a `SurgeryMap` for a default output glyph (name it; it replaces or wraps `_facing_pairs`).
- Stroke measurement: the ink length along a line at a coordinate through a polygon glyph, within a cross range.

`transform_variable_glyph(vg, bridge, geometry, upm)` keeps its signature. `process_variable_glyph` keeps its signature, and the new fields reach it through `BridgeConfig(**config_dict)`.

## Steps

1. Write `scaled_gap` and the stroke measurement with unit tests in `tests/unit/test_width_scaling.py`. Cover the formula at strength 0, 50 and 100, the minimum clamp, the minimum never exceeding base, and the measurement on a square ring and on an 8-like glyph with two counters.
2. Pair lines and give `replay` per-pair targets. The default replay (`align_to_lines`) must stay exact. Lines that do not pair keep today's mean target.
3. Wire `transform.py`: compute each master's ratios from `merged` (the flattened, overlap-removed masters) and run the fallback chain in `_shared.md`, in its order. Keep `transform.py` and every function within the size limits; helpers belong in `bridge_width.py`.
4. Update `tests/unit/test_variable_replay.py` only where a signature changed. Keep every assertion: test integrity flags removed or changed assertion lines.
5. Measure and report, for each fixture in `tests/fixtures/variable/` (Inter, Ubuntu, Cantarell) and each mode (fixed, and proportional at strength 100):
   - the number of glyphs bridged;
   - how many glyphs went down each fallback step;
   - the Inter `o` gap at Thin, default and Black.

   Fixed mode must bridge at least as many glyphs as HEAD did (the fallback guarantees it). Record the numbers with `loom memory note`.

## Traps

- Thin-master gaps get small. A gap at or below rounding noise (about 1 unit) can collapse after integer rounding and fail `_bridges_intact`. The minimum clamp keeps the gap above that unless `min_width_percent` is tiny. Let validation reject it and the fallback recover.
- Wider gaps in bold masters can miss the counter (`test_bridge_line_missing_a_narrowed_counter_gives_no_replay`). That is what the fallback exists for; do not widen `_SEARCH_EDGES` to compensate.
- Do not touch `core/`: the default surgery and every static path must stay byte for byte unchanged (`tests/regression` goldens pin static output).

## Done when

`uv run pytest tests/unit/test_width_scaling_contracts.py tests/unit/test_width_scaling.py tests/unit/test_variable_replay.py tests/unit/test_variable_transform.py tests/unit/test_variable_validate.py tests/unit/test_variable_align.py tests/unit/test_variable_engine_contracts.py --no-cov -q -p no:cacheprovider` passes, along with ruff and mypy on your files. The contract tests that need W3's CLI flags may still fail when you finish; say which ones in your report.
