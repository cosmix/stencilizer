# U1: thread the direction through glyph surgery, with its tests (gpt-5.6-terra)

Read `_shared.md` in this directory first ("FOUNDATION", "Direction semantics", "Measured facts").

## Files owned

- `src/stencilizer/core/surgery.py`
- `src/stencilizer/core/surgery_groups.py`
- `tests/integration/test_bridge_direction.py` (new; under 400 lines)

Read-only: `src/stencilizer/core/surgery_context.py` (already has `direction` after the
FOUNDATION step), `src/stencilizer/config/settings.py` (`BridgeDirection`),
`tests/integration/conftest.py` (`FIXTURES_DIR`), `src/stencilizer/core/processor.py`
(`process_glyph`, `FontProcessor.classify_glyphs`).

## Steps

1. `GlyphTransformer.transform` (`core/surgery.py`) builds `SurgeryContext(...)` positionally.
   Pass the glyph's direction as a keyword: `direction=self.bridge_config.direction`. Nothing else
   in this file changes.
2. In `core/surgery_groups.py` add a private helper directly above `process_groups` (import
   `BridgeDirection` from `stencilizer.config.settings`):

       def _spanning_allowed(ctx: SurgeryContext, axis: str) -> bool:
           """Span along ``axis`` when the glyph's explicit direction matches it, else follow the setting."""
           if ctx.direction is BridgeDirection.AUTO:
               return ctx.use_spanning
           return ctx.direction.value == axis

   In `process_groups`, replace the two conditions `ctx.use_spanning and _spanning(...)` with
   `_spanning_allowed(ctx, axis) and _spanning(...)`, keeping the arguments passed to `_spanning`
   (`indices` in the "horizontal" branch, `sorted_y` in the "vertical" branch) and everything else
   byte for byte. Do not touch `_sequential`, `arrangement`, `_spanning` or `group_islands`: a
   "single" arrangement reaches `SurgeryContext.merge` with no forced flags, and the FOUNDATION
   default applies the direction there.
3. Write `tests/integration/test_bridge_direction.py`. Drive glyphs through
   `process_glyph(glyph.to_dict(), BridgeConfig(...).model_dump(), upm)` and rebuild them with
   `Glyph.from_dict(result["glyph"])`; load glyphs with `FontReader` from `FIXTURES_DIR`.
   - `test_explicit_direction_splits_o_along_axis`: Roboto `O`. HORIZONTAL: 4 contours and no
     contour bbox spans the input's centre y; VERTICAL and AUTO: 4 contours and none spans the
     input's centre x. Compute both centres from the input bbox (728 and 703.5); never hard-code
     output coordinates.
   - `test_stacked_islands_follow_direction`: Roboto `B` and `eight`. HORIZONTAL equals AUTO with
     `use_spanning_bridges=False`; VERTICAL with `use_spanning_bridges=False` equals AUTO with
     `use_spanning_bridges=True`; HORIZONTAL differs from default AUTO. Compare
     `[c.to_dict() for c in glyph.contours]`.
   - `test_every_island_glyph_survives_explicit_directions`: for Roboto and Lato, every island
     glyph from `FontProcessor(StencilizerSettings(logging=LoggingConfig(log_file=tmp_path /
     "p.log"))).classify_glyphs(reader)` returns no "error" key under VERTICAL and under
     HORIZONTAL (about 1 s per font and direction).
   No `skip`/`xfail`/`importorskip` in this file.

Constraint the graph cannot show: with AUTO the new code must take exactly the old branches
(`tests/regression/test_behavior_golden.py` pins the output bit for bit), and no axis-mirrored
duplicate may appear (`tests/regression/test_code_structure.py`).

## Done when

`process_groups` consults `_spanning_allowed`, `GlyphTransformer` passes `direction=`, the three
tests are collected, and the proof command prints no errors.

## Proof (run once, report the output; never run the tests)

    .venv/bin/ruff check src/stencilizer/core/surgery.py src/stencilizer/core/surgery_groups.py tests/integration/test_bridge_direction.py && .venv/bin/mypy src/stencilizer/core/surgery.py src/stencilizer/core/surgery_groups.py tests/integration/test_bridge_direction.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/integration/test_bridge_direction.py
