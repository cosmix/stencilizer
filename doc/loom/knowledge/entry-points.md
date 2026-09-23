# Entry Points

> Key files agents should read first to understand the codebase.
> Keep current: correct or delete entries the code no longer supports.

(Add entry points as you discover them)

## CLI

Script `stencilizer = "stencilizer.cli.app:main"` (pyproject.toml). One Typer command, `stencilize` (src/stencilizer/cli/app.py): positional `input_font`, `--output/-o`, `--bridge-width/-w` (30-110, default 60, percent of a reference stroke of 10% of font UPM), `--workers/-j`, `--list-islands`, `--dry-run`, `--log-file`, `--log-level`, `--verbose/-v` / `--quiet/-q`, `--version/-V`. `--list-islands` and `--dry-run` short-circuit.

## Where to start reading

- `src/stencilizer/core/processor.py`: orchestration and the picklable `process_glyph` worker (processor.py:67).
- `src/stencilizer/core/surgery.py`: `GlyphTransformer.transform()` (surgery.py:42), delegating to surgery_groups.py and surgery_nested.py.
- `src/stencilizer/core/axis.py`: `Axis` (VERTICAL / HORIZONTAL), the parameter that replaced the mirrored horizontal/vertical modules.
- `src/stencilizer/core/analyzer.py`: contour hierarchy and island detection.
