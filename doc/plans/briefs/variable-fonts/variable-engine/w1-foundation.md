# W1: variable model, solver, reader, flattening, dependency (stage variable-engine, wave 1)

Tier: sonnet (`loom-software-engineer`). Never run git.

You own:

- `src/stencilizer/variable/__init__.py`
- `src/stencilizer/variable/model.py`
- `src/stencilizer/variable/solver.py`
- `src/stencilizer/variable/reader.py`
- `src/stencilizer/variable/flatten.py`
- `src/stencilizer/variable/rounding.py`
- `pyproject.toml`
- `uv.lock`
- `tests/unit/test_variable_model.py`
- the `VariationDataError` addition in `src/stencilizer/exceptions.py`

Read-only:

- `tests/fixtures/variable/` (built by the contract session);
- `tests/unit/test_variable_engine_contracts.py` (frozen, never edit);
- `src/stencilizer/io/converter.py`;
- `src/stencilizer/core/curve.py`;
- `src/stencilizer/domain/`.

Wave 2 (W2 replay, W3 overlaps) builds on your public surface, including `flatten_compatible`: W3 must test its vertex matching against the real flattener, so flattening is a wave-1 deliverable. The surface is pinned in the stage description; implement it exactly.

Winding: internally TrueType winding holds, outer contours clockwise (negative signed area, `core/analyzer.py:131`). The root `CLAUDE.md` line "TrueType: CCW=outer" is wrong; trust the analyzer.

## Steps

1. **Dependency.** Run `uv add skia-pathops` (a runtime dependency, never hand-edit `pyproject.toml`). Confirm with `uv run python -c "import pathops"`. 0.9.2 ships a `cp310-abi3` wheel for Linux x86_64 and macOS arm64, so CI and the PyInstaller build need nothing else.

2. **Exception.** In `src/stencilizer/exceptions.py`, add `VariationDataError(GlyphError)` with `__init__(self, glyph_name: str, reason: str)`, message `f"Variation error for '{glyph_name}': {reason}"`, and attributes `.glyph_name` and `.reason`. Follow the style of `GlyphProcessingError` (exceptions.py:57-63). Every raise of it in the engine is per glyph; callers (W2's transform, stage `variable-surfaces`' classification) turn it into a skipped or unbridged glyph, never a font-level failure.

3. **`model.py`.**
   - `Support`: a frozen dataclass with `axes: tuple[tuple[str, float, float, float], ...]` holding (tag, start, peak, end), sorted by tag.
     - `peak()` returns `{tag: peak}`.
     - `scalar(location)` returns `fontTools.varLib.models.supportScalar(location, {tag: (start, peak, end) for ...})` (OpenType rules, `ot=True`). Do not reimplement it: fontTools ignores an axis whose peak is 0, whose start > peak or peak > end, or whose start < 0 < end (`varLib/models.py:180-187`), and the glyph sets that produce the masters use exactly that function. An axis absent from `location` counts as 0.0.
   - `VariableGlyph`: a dataclass with `default: Glyph`, `supports: tuple[Support, ...]`, `masters: tuple[Glyph, ...]` (same length as supports; `masters[i]` is the full outline at `supports[i].peak()`) and `axis_tags: tuple[str, ...]` (fvar axis order).
     - `cff2: bool = False`, the last field: the outline format. `read_variable_glyph` sets it to `"CFF2" in font`; `to_dict`/`from_dict` carry it; `round_variable_glyph` dispatches on it.
     - `name` property: `default.name`.
     - `deltas()`: calls `solve_deltas` on the flattened point lists (all contours, in order) and returns one list of (dx, dy) per support. Compute it once and cache it in a field declared `field(default=None, init=False, repr=False, compare=False)`; `to_dict` never includes it. The GUI calls `instance()` on every slider tick and validation calls it once per location.
     - `instance(location)`: returns a `Glyph` with the default's metadata and point types, at `default + sum(support.scalar(location) * delta)`.
     - `to_dict()` / `from_dict()`: reuse `Glyph.to_dict` / `Glyph.from_dict`. Supports serialize as lists.
   - Validate in `__post_init__` that every master has the default's contour count, per-contour point count and point types. On mismatch, raise `VariationDataError(name, "incompatible master structure")`.

4. **`solver.py`.** `solve_deltas(supports, default, masters, *, glyph_name="<unknown>")`:
   - Build M with `M[j][k] = supports[k].scalar(supports[j].peak())`, then solve `M · D = (masters[j] - default)` per coordinate.
   - Pure-Python Gaussian elimination with partial pivoting. numpy is not a dependency, so do not add it.
   - A pivot with absolute value ≤ 1e-12 raises `VariationDataError(glyph_name, "singular variation supports")`. Two supports with the same peak are legal OpenType (GoogleSansCode `dollar` has (0, 0.5, 1) and (0.5, 0.5, 1)) and are singular here; such glyphs stay unbridged.
   - Zero supports returns `[]`.
   - Planning measured: masters read at each peak and solved this way reproduce Ubuntu `o`'s and `8`'s gvar deltas after `calcInferredDeltas`, corner tuples included, within 2.8e-14.

5. **`reader.py`.**
   - `is_variable(font)` returns `"fvar" in font`.
   - `read_variable_glyph(font, name)`:
     - Return `None` if the glyph is empty or composite. Use `font["glyf"][name].isComposite()` for TrueType; for CFF2 a glyph is never composite.
     - **Supports, TrueType:** `variations = font["gvar"].variations.get(name, []) if "gvar" in font else []`. A font with fvar and no gvar is legal (the GUI fixture `tests/gui/conftest.py:80-95` `variable_font_path` is exactly that) and gives every glyph `supports=()`. For each `TupleVariation`, build `Support(tuple(sorted((tag, start, peak, end) for tag, (start, peak, end) in tv.axes.items())))`. Keep the gvar tuple order.
     - **Supports, CFF2:** add a public helper `cff2_vsindex(font: TTFont, name: str) -> int | None` that the CFF2 writer also uses. Draw the charstring through `charstring.draw(NullPen(), blender)` with a `blender(vs_index, deltas)` callable that records `vs_index` and returns `0` (fontTools calls it once per blended operand, after following subroutines, and adds the return value to that operand: `misc/psCharStrings.py:497-516`). Return `None` when the blender is never called: the glyph does not vary (12 of 1,322 Cantarell glyphs), so it gets `supports=()` and `masters=()`. Read the glyph's FD private through `_, fd = top_dict.CharStrings.getItemAndSelector(name)` and `top_dict.FDArray[fd or 0].Private`; when its `vsindex` attribute is set to a value different from the recorded index, raise `VariationDataError(name, "Private vsindex not honoured by fontTools glyph sets")` (fontTools 4.66 blends with the charstring's own index and ignores Private.vsindex, `misc/psCharStrings.py:338, 512`). Scanning the top-level program for `vsindex` is wrong: Cantarell is subroutinized and its `blend` operators live only in subrs. Map each region index in `VarData[vsindex].VarRegionIndex`, in that order, to `VarRegionList.Region[i].VarRegionAxis` (StartCoord, PeakCoord, EndCoord) by fvar axis order. Leave out axes whose peak is 0. When the blender records more than one distinct `vs_index` for the glyph, raise `VariationDataError(name, "more than one vsindex in one charstring")`: the CFF2 spec allows one `vsindex` per charstring, before the first `blend`. Never substitute VarData 0 for an index you did not record.
     - **Outlines:** draw `font.getGlyphSet(location=support.peak(), normalized=True)[name]` and the default (`font.getGlyphSet()`) through `stencilizer.io.converter.fonttools_glyph_to_domain(name, glyph, font)`. Do not reverse anything yourself: the `cff2-static` stage made that function reverse CFF2 winding, and it is merged. `normalized=True` skips avar (`ttFont.py:1321-1322`), so supports and locations both live in post-avar normalized space.
     - **axis_tags:** `tuple(a.axisTag for a in font["fvar"].axes)`.
     - Point structure matches across masters: the converter rotates each contour to its first on-curve point, appends a duplicate of point 0 when the last segment is a curve, and prepends an implied on-curve midpoint when a contour has no on-curve point (converter.py:75-128), identically for every glyph set. Planning found 0 structure mismatches over 1,023 simple Inter glyphs. So domain points do not map 1:1 to glyf points (Ubuntu `o`: 34 domain points against 32 glyf points); the writers handle that.
     - `VariationDataError` from `VariableGlyph` construction or `cff2_vsindex` propagates; callers handle it per glyph.
   - `variable/__init__.py` re-exports `Support`, `VariableGlyph`, `read_variable_glyph`, `is_variable`, `solve_deltas` and `flatten_compatible`. It must import nothing from `stencilizer.core.processor` (stage `variable-surfaces` lazy-imports `variable.processing` from there).

6. **`flatten.py`: `flatten_compatible(vg: VariableGlyph, upm: int) -> VariableGlyph`.** Choose one subdivision count per curve segment that holds in every master: the smallest n such that uniform-t sampling of the segment stays within `curve_tolerance(upm)` (`core/curve.py:8`) in the default and in every master. Then emit identical structure everywhere, with all points ON_CURVE:
   - **Quadratic runs with consecutive off-curves:** expand implied on-curve midpoints first. The midpoint is linear in the control points, so it is master-compatible.
   - **Uniform t:** the sampled points are Bernstein combinations of the control points, and the same t list is used in every master.
   - **Contour start:** keep the contour's first point. When a contour starts off-curve, rotate exactly as `core/curve.py` `flatten_contour` does (curve.py:47-60), identically in all masters.
   - **Cap:** at most 64 subdivisions per segment. Beyond that, raise `VariationDataError(name, "curve needs more than 64 subdivisions")`.
   - Contours that are already polygons pass through unchanged. `core.curve.flatten_contour` must then be a no-op on the result: it returns all-ON_CURVE contours as they are (curve.py:52-53).

7. **`rounding.py`: `round_variable_glyph(vg: VariableGlyph) -> VariableGlyph`.** The values the target format stores, returned as a `VariableGlyph` whose masters are rebuilt from them, so `deltas()` returns the stored deltas. The engine (W2) validates this rounded glyph and the writers (stage `variable-writers`) store it, so what passes validation is what the font contains.
   - `vg.cff2` False (gvar): `rdef` = default coordinates through `fontTools.misc.roundTools.otRound`. Visit supports sorted by (number of axes, then largest |peak| first). For support k with peak p_k: `acc = rdef + Σ over visited j of supports[j].scalar(p_k) · RD_j`; `RD_k = otRound(vg.instance(p_k) - acc)` per point and coordinate. When a not-yet-visited support has a non-zero scalar at p_k, use `RD_k = otRound(deltas[k])` instead. Then `masters[k] = rdef + Σ over all j of supports[j].scalar(p_k) · RD_j`. Rounding each delta on its own measured 2.0 units of error at Ubuntu `{wdth: -1, wght: ±1}`, where corner tuples add up; this rule keeps every support peak within 0.5.
   - `vg.cff2` True: default through `otRound`, deltas unchanged (CFF2 stores them as 16.16 fixed); masters rebuilt as default + delta.
   - Idempotent: rounding a rounded glyph returns equal coordinates within 1e-9. Points with equal default coordinates and equal deltas get equal rounded values, so the points of one bridge line stay collinear in every master.
   - `supports=()` returns the rounded default with no masters.

8. **Tests** in `tests/unit/test_variable_model.py`:
   - `Support.scalar` cases: inside, outside, at the peak, missing axis, peak 0, and region (-1, 0.5, 1) at `{}` giving 1.0 (the OpenType start < 0 < end rule).
   - Round-trip of `to_dict`; the cached deltas are not in it.
   - `instance` at a peak equals that master within 1e-6 for Ubuntu-VF-subset `o`.
   - Mismatched master structure raises.
   - `solve_deltas` on a hand-built two-region case, and on two supports sharing a peak (raises).
   - A font with fvar and no gvar (Roboto from `tests/fixtures/` plus an fvar table, as `tests/gui/conftest.py:80-95` builds it) reads with `supports=()`.
   - `cff2_vsindex` on Cantarell-VF-subset `o` returns 0, and `supports` has 2 entries.
   - `round_variable_glyph`: idempotent on Ubuntu-VF-subset `o` (gvar) and Cantarell-VF-subset `o` (CFF2); integral default and (gvar) integral deltas; every support peak within 0.5 of the unrounded instance; two points with equal default and equal deltas stay equal.
   - `cff2_vsindex` raises `VariationDataError` when a blender records two indices (drive it with a stub charstring whose `draw` calls the blender with 0 and then 1).
   - `flatten_compatible` keeps structure identical across masters, stays within tolerance, and uses a subdivision count that is not a power of two for at least one Inter-VF-subset glyph (W3's float32 matching must work for any count).

   Resolve fixture glyphs by character: `font.getBestCmap()[ord("o")]`.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_variable_model.py tests/unit/test_variable_engine_contracts.py::test_solver_reproduces_original_deltas tests/unit/test_variable_engine_contracts.py::test_singular_supports_raise tests/unit/test_variable_engine_contracts.py::test_cff2_variable_reader_uses_varstore_regions tests/unit/test_variable_engine_contracts.py::test_cff2_vsindex_selection
```

fontTools imports carry `# type: ignore[import-untyped]`, as elsewhere in `src/`. mypy is strict. Limits: files ≤400 lines, functions ≤50 effective lines. The verifier runs lint, format and mypy.
