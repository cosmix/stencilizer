# W2: CFF2 blend writer (stage variable-writers)

Tier: opus (`loom-senior-software-engineer`, effort high). Never run git.

You own:

- `src/stencilizer/variable/write_cff2.py`
- `tests/unit/test_write_cff2.py`

Read-only:

- `src/stencilizer/io/converter.py` (`_update_cff2_glyph` from stage `cff2-static`: private lookup, winding reversal);
- `src/stencilizer/variable/model.py`, `rounding.py` (`round_variable_glyph`) and `reader.py` (`cff2_vsindex(font, name) -> int | None`, CFF2 support extraction: which VarData, region order);
- the frozen `tests/unit/test_variable_writer_contracts.py`.

W3 calls your function only for glyphs the engine bridged; untouched glyphs keep their original charstrings.

## Pinned signature

```python
def write_cff2_variable_glyph(font: TTFont, vg: VariableGlyph) -> None
```

## Requirement

The new charstring for `vg.name` must evaluate, through `font.getGlyphSet(location=s.peak(), normalized=True)` after save and reload, to `vg.instance(s.peak())` within 1 unit for every support `s`, and to `vg.default` at `{}`. Width is not encoded (CFF2).

## Recipe (probed on Cantarell-VF-subset `o`, fontTools 4.66.0)

0. **Round first and handle constant glyphs.** `r = round_variable_glyph(vg)` (idempotent: the engine already rounded and validated a transformed glyph, so it is stored as validated). Use `r` below. When `r.supports == ()` (`cff2_vsindex` returned `None`: the glyph never blends), emit plain operands, no `blend`, no `vsindex`, and skip the region check.

1. **Regions.** `n = cff2_vsindex(font, vg.name)` (W1 of the engine stage; it follows subroutines, where Cantarell keeps every `blend`). Take `region_indices = VarData[n].VarRegionIndex` from `font["CFF2"].cff.topDictIndex[0].VarStore.otVarStore`. Match `vg.supports` to those regions by axes (the reader built them in that order) and raise `ValueError` if the count or any axes differ. Blend needs exactly one delta per region, in region order.
2. **Absolute outlines per support.** `deltas = r.deltas()`. Reverse every contour back to CFF winding with plain `list(reversed(points))`, as `converter.py:254` does, for the default and for each support's absolute delta list, so point i means the same point everywhere. Keeping the start point first (`[p0] + reversed(rest)`) shifts the reread by one point (planning: 7.9 units) and emits a zero-length `rlineto`.
3. **Rounding.** `r` already has the absolute default rounded and float deltas (CFF2 encodes them as 16.16 fixed point). Then take relative differences of consecutive points, for the default and for each support's delta list separately: operand i of support k is `delta_k[i] - delta_k[i-1]`. Planning measured 13.03 units of drift when relative arguments were rounded after differencing, 0.5 when absolute positions were rounded first, and 0.003 with float deltas.
4. **Commands.** Build `rmoveto` / `rlineto` / `rrcurveto` commands with every domain point emitted. Each numeric argument is either a plain number (when every delta is 0) or the list `[default_rel, d_0_rel, ..., d_{n-1}_rel, 1]`. `commandsToProgram` only flattens that list and appends `blend` (`cffLib/specializer.py:131-151`), so a list without the count, such as `[100, 10, 20]`, emits `100 10 20 blend` and reads 20 as the blend count: the trailing `1` is the blend count that `specializeCommands` asserts (`cffLib/specializer.py:498`) and that `CFF2CharStringMergePen.reorder_blend_args` produces (`varLib/cff.py:558-613`). Without it the font fails to reload with "CFF2 CharStrings must not have an initial width value".
5. **Encode.** `program = commandsToProgram(specializeCommands(cmds, generalizeFirst=False, preserveTopology=True, maxstack=maxStackLimit))` with `maxStackLimit` from `fontTools.cffLib` (513). `preserveTopology=True` keeps the zero-length segments rounding creates (planning: 514 points read back with it, 488 without). The specializer kept stack use at 512 for a 514-point glyph with 2 regions.
6. **Store.** When `n` is neither `None` nor 0, prepend `[n, "vsindex"]` (operand, then operator). Create `T2CharString(program=program, private=old.private, globalSubrs=old.globalSubrs)` from `old = top_dict.CharStrings[vg.name]` (this picks the right FD under FDSelect) and assign it to `top_dict.CharStrings[vg.name]`. Replacing one charstring in the subroutinized font with one that calls no subrs saves, and every glyph still draws.

`CFF2CharStringMergePen` needs a `VariationModel` in `getCharString` (`varLib/cff.py:615-631`); building commands directly, as above, is the chosen route.

## Tests (`tests/unit/test_write_cff2.py`)

Use `tests/fixtures/variable/Cantarell-VF-subset.otf`. Compare outlines through `fonttools_glyph_to_domain` (which reverses CFF2 winding after stage `cff2-static`) point by point.

- Writing the untransformed `read_variable_glyph(font, cmap[ord("o")])` back and saving to `tmp_path` reproduces outlines at `{"wght": -1.0}`, `{}` and `{"wght": 1.0}` within 0.5 units, with the same point count.
- A flattened `o` (all ON_CURVE, several hundred points) round-trips with the same point count.
- Encoding unit test, before any font round trip: a two-region command list with one blended `rlineto` (`[[100, 10, 20, 1], 0]`) goes through `specializeCommands(..., generalizeFirst=False, preserveTopology=True)` and `commandsToProgram`; the program ends `100 10 20 1 blend 0 rlineto` (allowing the specializer's operator choice), and a `T2CharString` built from it on the Cantarell private draws, through the blender `lambda _vs_index, deltas: 0.5 * deltas[0]` (region weights 0.5 and 0), a line to (105, 0).
- Non-zero vsindex: on an in-memory Cantarell copy with a second VarData appended (a copy of VarData[0] with its regions reversed; see the engine contract `test_cff2_vsindex_selection`) and `[1, "vsindex"]` prepended to `o`, writing `read_variable_glyph(font, o)` back produces a program starting `1 vsindex` that reproduces the outline at both peaks within 0.5 units after save and reload.
- A glyph with `supports == ()` writes a charstring with no `blend` and no `vsindex`.
- A `VariableGlyph` whose support count differs from the region count raises `ValueError`.
- The saved font reloads, and every glyph draws at `{}`, `{"wght": -1.0}` and `{"wght": 1.0}`.

## Proof

```bash
uv run pytest --no-cov -q -p no:cacheprovider tests/unit/test_write_cff2.py
```

The contract test `test_cff2_blend_output_reproduces_instances` needs W3; the orchestrator runs it. Size limits: files ≤400 lines, functions ≤50 effective lines. If the encoding approach fails twice, stop and report the evidence; the orchestrator escalates.
