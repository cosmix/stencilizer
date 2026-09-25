# U6: session lists composites, previews with directions, surveys unbridged glyphs (gpt-5.6-terra)

Read `_shared.md` in this directory first ("gui/session.py" contract, "Direction semantics",
measured facts).

## Files owned

- `src/stencilizer/gui/session.py` (stays Qt-free)

Read-only, already in the worktree from wave 1 (NOT in the source graph; read them with `cat`):
`src/stencilizer/gui/composites.py` (U3), `src/stencilizer/core/processor.py` (U2: `process`
takes `directions=`, `process_glyph` reports islands actually bridged).

## Steps

1. Composites at open. `FontSession` gains the dataclass fields `composites`,
   `component_outlines` and `display_names` (types in the contract), declared after
   `source_sha256` and before `_glyph_index`. In `open`, inside the `with FontReader(...)` block:
   classify as today, then `composites = find_bridged_composites(reader, island_names)`,
   `component_outlines = load_component_outlines(reader, composites)`, and `display_names` = the
   island glyph names plus the composite names, ordered by their index in
   `reader.font.getGlyphOrder()`. `__post_init__` keeps `_glyph_index` for island glyphs and adds
   `_composite_index: dict[str, CompositeGlyph]` and `_composed: dict[str, Glyph]` (each
   composite's `compose(composite, component_outlines)`, built once), both `field(init=False,
   repr=False)`. `glyph(name)` returns the island glyph, else the composed composite, else None.
   `display_glyphs` returns `[self.glyph(n) for n in display_names]` (never None there).
   `direction_sources(name)`: `(name,)` for an island glyph, the composite's `sources` for a
   composite, `()` otherwise.
2. `preview(name, bridge, geometry, directions=None)`. For an island glyph: run
   `process_glyph` as today with the config
   `bridge.model_copy(update={"direction": directions.get(name, bridge.direction)}).model_dump()`
   (`directions` None means `{}`); keep the existing result mapping (`error` -> stenciled None,
   `bridges_added` 0). For a composite: preview each source that way; if any source fails,
   return `stenciled=None`, `error=f"{source}: {error}"`; otherwise `stenciled` =
   `compose(composite, {**component_outlines, **stenciled_sources})`, `bridges_added` and
   `duration_ms` = the sums over the sources, `original` = the cached composed glyph (the same
   object `glyph(name)` returns). Unknown names raise `GlyphNotFoundError(name)` as today. Keep
   each function under 50 lines: a private `_preview_island` helper both paths use.
3. `unbridged(bridge, geometry, directions=None)`: preview every island glyph as in step 2; an
   island glyph is unbridged when its preview has an error or `bridges_added == 0`; a composite
   is unbridged when all of its sources are. Return a `frozenset` of names.
   `save(..., directions=None)` passes `directions=directions` to `self.processor.process`;
   every other line of `save` stays as it is (the digest checks, the staging directory and
   `_publish` are the save-safety contract in doc/loom/knowledge/architecture/gui.md).

Constraints: `island_glyphs` keeps returning `classification.glyphs_to_process` (the save and
existing tests use it). Never import PySide6 here. The file must stay under 400 lines.

## Done when

On Roboto: `len(display_glyphs) == 1027`, `direction_sources("Aring") == ("A", "ring")`,
`preview("Aacute", ...)` returns a composed stenciled outline, and `"four" in unbridged(...)`.

## Proof (run once, report the output)

    .venv/bin/ruff check src/stencilizer/gui/session.py && .venv/bin/mypy src/stencilizer/gui/session.py && .venv/bin/python -c "import sys, stencilizer.gui.session; sys.exit('PySide6' in sys.modules)"
