# Entry Points

> Key files agents should read first to understand the codebase.
> Keep current: correct or delete entries the code no longer supports.

(Add entry points as you discover them)

## CLI

Script `stencilizer = "stencilizer.cli.app:main"` (pyproject.toml). One Typer command, `stencilize` (src/stencilizer/cli/app.py): positional `input_font`, `--output/-o`, `--bridge-width/-w` (30-110, default 60, percent of a reference stroke of 10% of font UPM), `--workers/-j`, `--instance` (pin a variable font to a static instance, `wght=700,wdth=90`; `io/instance.py` `instantiate_static` writes a temporary font and messages name the input path), `--list-islands`, `--dry-run`, `--log-file`, `--log-level`, `--verbose/-v` / `--quiet/-q`, `--version/-V`. `--list-islands` and `--dry-run` short-circuit; their handlers (`_handle_list_islands`, `_handle_dry_run`) are in `cli/handlers.py`, and `cli/output.py` prints the font info (axes included for variable fonts). Text taken from a font (axis tags, glyph names, paths) must go through `rich.markup.escape` before `Console.print`.

## Where to start reading

- `src/stencilizer/core/processor.py`: orchestration and the picklable `process_glyph` worker (processor.py:67).
- `src/stencilizer/core/surgery.py`: `GlyphTransformer.transform()` (surgery.py:42), delegating to surgery_groups.py and surgery_nested.py.
- `src/stencilizer/core/axis.py`: `Axis` (VERTICAL / HORIZONTAL), the parameter that replaced the mirrored horizontal/vertical modules.
- `src/stencilizer/core/analyzer.py`: contour hierarchy and island detection.

## GUI

Script `stencilizer-gui = "stencilizer.gui.app:main"` (pyproject.toml:20), usage `stencilizer-gui [font]`. Start at `src/stencilizer/gui/session.py` (`FontSession`, Qt-free), `composites.py` (composites shown in the grid) and `controller.py` (`GuiController`, per-glyph directions); `main_window.py` composes the widgets (`header.py` is the top bar), `theme.py` holds every colour and QSS rule, `direction_picker.py` holds the Auto / Vertical / Horizontal choice. Module table: [architecture/gui](architecture/gui.md). Tests: `tests/gui/` (`conftest.py` has the `processor`, font-path and `outlines_match` fixtures and `build_settings`; `test_beautify_contracts.py` pins the theme contracts); direction tests also in `tests/integration/test_bridge_direction.py` and `test_processor_directions.py`.

## Variable fonts

Start at `src/stencilizer/variable/transform.py` (`transform_variable_glyph`, `process_variable_glyph`), which chains `flatten.py`, `overlaps.py` (with `crossings.py`), `replay.py` (`map_surgery`, `replay`), `align.py`, `rounding.py` and `validate.py` (with `holes.py`); `reader.py` (`is_variable`, `read_variable_glyph`), `solver.py` and `model.py` (`VariableGlyph`, `Support`) read the font; `write_gvar.py` and `write_cff2.py` write it. `variable/processing.py` is the `FontProcessor` hook (`classify_variable_glyphs`, `process_variable_font`). `io/instance.py` implements `--instance`; the GUI side is `gui/variable_session.py` and `gui/axis_controls.py`. Fixtures: `tests/fixtures/variable/` (Ubuntu and Inter with gvar, Cantarell with CFF2); shared builders and `island_count` in `tests/font_helpers.py`. How the pieces fit: [patterns/variable-replay](patterns/variable-replay.md).
