# U3: composite glyph resolution, with tests (gpt-5.6-terra)

Read `_shared.md` in this directory first ("gui/composites.py" contract and the measured facts).

## Files owned

- `src/stencilizer/gui/composites.py` (new, Qt-free: never import PySide6)
- `tests/gui/test_composites.py` (new; under 400 lines)

Read-only: `src/stencilizer/io/reader.py` (`FontReader.font`, `FontReader.get_glyph`),
`src/stencilizer/domain/contour.py` (`Point(x, y, point_type)` is frozen; `Contour(points,
direction)`; `WindingDirection`), `src/stencilizer/domain/glyph.py` (`Glyph`, `GlyphMetadata`),
`src/stencilizer/io/converter.py` (`_recording_to_contours`), `tests/gui/conftest.py`
(`roboto_path`, `processor`, `FIXTURES_DIR`).

## Steps

1. `component_parts(glyph_set, name)`: draw `glyph_set[name]` into a
   `fontTools.pens.recordingPen.RecordingPen`. If the recording is empty or holds any operation
   other than `"addComponent"`, return `()`: the glyph is not a composite. Otherwise, for each
   `("addComponent", (base, transform))`, compose with the parent's transform using
   `fontTools.misc.transform.Transform`: `total = Transform(*parent).transform(transform)` (child
   first, then parent: `Transform(2,0,0,2,10,0).transform((1,0,0,1,5,0))` is
   `(2,0,0,2,20,0)`; the top level starts from the identity). When `base` is itself a composite,
   extend with its parts under `total`; otherwise append `ComponentPart(base, tuple(float(v) for
   v in total))`, in recording order. `find_bridged_composites(reader, island_names)`: one
   `reader.font.getGlyphSet()`, then for every name in `reader.font.getGlyphOrder()` keep the
   glyph when at least one part's base is in `island_names`; `sources` = the island bases among
   the parts in first-appearance order without repeats; `metadata` =
   `reader.get_glyph(name).metadata`; return a tuple in glyph order.
   `load_component_outlines(reader, composites)` returns `{base: reader.get_glyph(base)}` for
   every base of every part (skip a base whose `get_glyph` returns None).
2. `compose(composite, outlines)`: build `Glyph(metadata=composite.metadata, contours=...)` by
   mapping every point of every contour of `outlines[part.base]` through
   `Transform(*part.transform).transformPoint((point.x, point.y))`, keeping `point_type`. When the
   transform mirrors (`xx * yy - xy * yx < 0`), reverse each contour's point list and swap its
   `direction` (CLOCKWISE <-> COUNTER_CLOCKWISE, None stays None) so the TrueType convention
   (clockwise outer, counter-clockwise hole) survives; otherwise keep the direction. Build new
   `Contour`/`Point` objects; never mutate the inputs (previews share them). Import fontTools with
   `# type: ignore[import-untyped]`; every public name gets a docstring.
3. Write `tests/gui/test_composites.py`:
   - `test_finds_bridged_composites_in_glyph_order`: Roboto classified with the `processor`
     fixture; 465 composites in glyph order; `Aacute` has sources `("A",)` and parts `A`
     (identity) then `acute` (dx 447, dy 310); `Aring` sources `("A", "ring")`; none has empty
     sources; `O` and `space` are absent.
   - `test_composed_outlines_match_fonttools_decomposition`: for EVERY composite found in Roboto,
     the oracle is `_recording_to_contours(pen.value)` with `pen =
     DecomposingRecordingPen(font.getGlyphSet())` drawn with the glyph; `compose(...)` has the
     same contour count and, point by point, the same `point_type` and coordinates within 1e-6.
     (At planning time all 1433 Roboto composites matched this oracle.)
   - `test_nested_and_mirrored_components`: build a TrueType font in `tmp_path` with
     `fontTools.fontBuilder.FontBuilder` and `TTGlyphPen`: `O` (clockwise outer square,
     counter-clockwise inner square), `acute` (a triangle), `Oacute` = `O` + `acute` at (0, 500),
     `Onested` = `Oacute` at (100, 0), `Omirror` = `O` with transform (-1, 0, 0, 1, 600, 0).
     `component_parts` of `Onested` flattens to `O` at dx 100 and `acute` at (100, 500); `compose`
     of `Omirror` keeps the outer contour clockwise (`signed_area() < 0`) and the hole
     counter-clockwise (`> 0`).
   - `test_compose_leaves_inputs_untouched`: composing twice returns equal glyphs and the outlines
     passed in keep their `to_dict()`.

## Done when

On Roboto, `find_bridged_composites` returns 465 composites with Aacute as above, and the proof
command is clean.

## Proof (run once, report the output; never run the tests)

    .venv/bin/ruff check src/stencilizer/gui/composites.py tests/gui/test_composites.py && .venv/bin/mypy src/stencilizer/gui/composites.py tests/gui/test_composites.py && .venv/bin/python -c "import sys, stencilizer.gui.composites; sys.exit('PySide6' in sys.modules)" && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_composites.py
