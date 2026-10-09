# W2: CLI `--instance` (stage variable-surfaces, wave 2, codex unit)

Tier: codex gpt-5.6-terra, effort xhigh. Do not run git. Do not touch `.loom/`.

You own exactly:

- `src/stencilizer/io/instance.py` (create)
- `src/stencilizer/cli/app.py`
- `src/stencilizer/cli/output.py`
- `src/stencilizer/cli/handlers.py` (create)
- `tests/unit/test_instance.py` (create)

Anchors:

- `stencilize` (`cli/app.py` line 75): options are declared as `Annotated` parameters; module-level aliases such as `BridgeWidthOption` sit at lines 47-71;
- `_run_command` (line 124): the parameters are forwarded positionally (lines 110-121 and 124-135);
- `_classify_font` (line 186), `_run_standard` (line 206), `_process_font` (line 240);
- `_handle_list_islands` (line 299), `_scan_islands` (line 324), `_handle_dry_run` (line 336);
- `print_font_info` (`cli/output.py` line 76);
- W1 (merged before you start): `stencilizer.variable.processing.variable_island_counts(reader) -> list[tuple[str, int]]` and `stencilizer.variable.reader.is_variable(font)`.

Size limits, measured with `tests/regression/test_code_structure.py`'s own helper: `cli/app.py` is 393 of 400 lines, `stencilize` is 46 of 50 effective lines and `_run_command` 42 of 50. An inline multi-line `Annotated[...]` option plus one forwarded argument puts `stencilize` at 51 and fails the gate. This unit adds roughly 25 lines to `app.py`, so step 2 first moves the list-islands and dry-run handlers out.

## Steps

1. `io/instance.py`:
   - `parse_instance_spec(spec: str, font: TTFont) -> dict[str, float]`:
     - split on `,`; each item is `tag=value` with a float value;
     - every fvar axis not named takes its `defaultValue`;
     - raise `FontFormatError(<font path or "<font>">, message)` when the font has no `fvar` ("--instance requires a variable font"), when an item has no `=` or a non-numeric value ("malformed --instance item '<item>'"), when a tag is not an fvar axis ("unknown axis '<tag>'"), or when a value is outside [minValue, maxValue] ("axis '<tag>' value <v> outside <min>..<max>").
     - These checks are the only validation: fontTools' instancer silently clamps an out-of-range value (wght=5000 gives a static font and exit 0) and raises a bare `KeyError` for an unknown axis.
   - `instantiate_static(font_path: Path, spec: str, workdir: Path) -> Path`:
     - load `TTFont(font_path)` (the instancer needs a `TTFont`; a path raises `ValueError`);
     - `limits = parse_instance_spec(spec, font)`;
     - call `fontTools.varLib.instancer.instantiateVariableFont(font, limits, static=True, overlap=OverlapMode.REMOVE, downgradeCFF2=True, updateFontNames="STAT" in font)` with `OverlapMode` from `fontTools.varLib.instancer`.
       - `overlap=OverlapMode.REMOVE` (skia-pathops, a runtime dependency since stage `variable-engine`) merges overlapping contours. Without it the static instance keeps them and the static pipeline finds no island in overlap-built counters: planning measured 0 islands in Inter `A D P R e 4 &` at wght=700, and 1, 1, 1, 1, 1, 1, 2 with it.
       - `downgradeCFF2=True` makes a CFF2 input produce a static `CFF ` font, the widely supported static format; without it Cantarell stays CFF2.
       - Naming never decides success. With `updateFontNames=True`, fontTools raises `ValueError: Cannot find Axis Values {'wght': 650}` for an in-range value that no STAT AxisValue names (Inter: 700 works, 650 raises; `varLib/instancer/names.py:73-77, 123-124, 161-164`). On that `ValueError`, reload `TTFont(font_path)` and call again with `updateFontNames=False`; the instance keeps the variable font's names and `FontWriter` adds the Stenciled suffix later. A `ValueError` from the second call propagates.
     - save to `workdir / f"{font_path.stem}-instance{font_path.suffix}"` and return the path.
2. `cli/app.py` and `cli/handlers.py`:
   - move `_handle_list_islands`, `_scan_islands` and `_handle_dry_run` (app.py:299 to the end of `_handle_dry_run`) unchanged into a new `src/stencilizer/cli/handlers.py`, with the imports they need, and import them back into `app.py`. No test patches those three or anything they use through `stencilizer.cli.app` (`tests/unit/test_review_processing.py` patches `stencilizer.cli.app.FontReader` only for the standard path's `_classify_font`). The stage wiring check greps `variable_island_counts\(` in `cli/handlers.py`;
   - declare `InstanceOption = Annotated[str | None, typer.Option("--instance", help="Pin a variable font to a static instance, e.g. wght=700,wdth=90")]` at module level beside `BridgeWidthOption`; add `instance: InstanceOption = None` to `stencilize` and thread it through `_run_command` (one line each);
   - put the instance resolution in one small helper inside `cli/app.py` that enters `tempfile.TemporaryDirectory(prefix="stencilizer-instance-")` and calls `instantiate_static(input_font, instance, Path(tmp))` literally. Keep that call in `cli/app.py`, never in a new module: the stage's wiring check greps `instantiate_static\(` there and its `reachable` check starts at `stencilize`. Use the returned path as the font for classification, dry run, list-islands and processing; the context covers `typer.Exit` and Ctrl-C (`Exit(130)`, app.py:281), so the temp dir is always removed;
   - the default output name must still derive from the original `input_font` (`FontWriter.get_stenciled_path(input_font)`), because the temp dir is gone after the run;
   - `FontFormatError` already maps to exit 1 through the `StencilizerError` clause (lines 152-165). `_handle_list_islands` and `_handle_dry_run` catch every `Exception` themselves; make sure `--instance` errors there also exit 1 with the message;
   - `_scan_islands(reader)` in `cli/handlers.py`: when `is_variable(reader.font)`, return `variable_island_counts(reader)` (import it inside the function); otherwise keep today's loop.
3. `cli/output.py` `print_font_info`: add an optional `axes: str | None = None` parameter. When it is given, print a line `Variable axes: wght 100–900, wdth 75–100` (fvar axis order, `minValue–maxValue`, integers printed without a decimal point). Its three callers (`_classify_font` in `app.py`, and the list-islands and dry-run handlers now in `handlers.py`) pass `axes=` for fonts with `fvar`; with `--instance` they see the static instance, which has no `fvar`, so no axes line. The frozen contract `test_cli_shows_variable_axes` runs `--dry-run` with and without `--instance` and looks for `Variable axes`, `wght 100` and `900`; the stage wiring check greps `axes=` in `cli/app.py`.
4. Tests in `tests/unit/test_instance.py`, using `tests/fixtures/variable/Inter-VF-subset.ttf` (axes opsz 14..32 default 14, wght 100..900 default 400) and `Cantarell-VF-subset.otf`:
   - parse fills defaults;
   - each error case raises with the message;
   - `instantiate_static` output has no `fvar`, and its `P` has one island under `GlyphAnalyzer().analyze(glyph, upm)` (read through `FontReader`);
   - `instantiate_static` on Cantarell produces a `CFF ` table and no `CFF2`;
   - `instantiate_static(Inter, "wght=650", ...)` succeeds (STAT has no AxisValue for 650) and the output has no `fvar`;
   - `print_font_info(..., axes="wght 100–900")` prints the `Variable axes` line, and without `axes` it does not.

Test-writing rules units have missed before: quote `cast()` type arguments (ruff TC006), prefix unused stub parameters with `_` (ARG001), keep `ruff format` clean.

## Done

One static check, run once (codex's sandbox has no network and a read-only uv cache, so `uv run` fails there; call the worktree venv directly):

```bash
.venv/bin/ruff check src/stencilizer/cli src/stencilizer/io/instance.py tests/unit/test_instance.py && .venv/bin/mypy src/stencilizer/cli src/stencilizer/io/instance.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/unit/test_instance.py
```

The orchestrator runs the tests, `uv run stencilizer --help` and `tests/regression/test_code_structure.py`.
