# W3: width-scaling CLI options and stencil-first `--instance` wiring (stage width-scaling)

Tier: sonnet (`loom-software-engineer`). Never run git. Read `_shared.md` in this directory first. The plan (`doc/plans/IN_PROGRESS-PLAN-variable-bridge-width.md`) wins over this brief.

You own:

- `src/stencilizer/cli/app.py`
- `src/stencilizer/cli/handlers.py`
- `tests/unit/test_cli_width_scaling.py` (new)
- `tests/unit/test_cli_variable.py`

Read-only:

- `src/stencilizer/cli/pinning.py`: written in parallel by codex unit U1, with the interface pinned below; code against it.
- `src/stencilizer/io/instance.py`, `core/processor.py`, `variable/reader.py`
- the frozen `tests/unit/test_width_scaling_contracts.py` (its CLI contracts drive your code)
- `tests/unit/test_variable_surface_contracts.py` (the `_cli` CliRunner pattern)

## Facts (verified at HEAD)

- `cli/app.py` is 340 lines. Options are `Annotated` aliases at L57-72 (`BridgeWidthOption` is `min=30.0, max=110.0`). `stencilize` (def at L86, forwarding call L122-134) is at 48 of 50 effective lines, and forwards everything to `_run_command` (L137-181, 45 of 50). The three new parameters alone put `stencilize` at 51, so:
  - define the three options as module-level `Annotated` aliases and convert the inline options to aliases too (`--dry-run` saves 5 lines, `--workers` 3, `--output` 3);
  - build the `BridgeConfig` in a `_bridge_config(...)` helper;
  - put the warning/settings switch and the stencil-first flow in helpers.
- `_run_command` builds `StencilizerSettings(bridge=BridgeConfig(width_percent=bridge_width), ...)` at L153-157, then opens `with _instance_font(input_font, instance) as font_path:` (L159), which pins first. It then dispatches to `_handle_list_islands`, `_handle_dry_run` or `_run_standard` (L236).
- `_instance_font` (L202-208) moves to `cli/pinning.py` as `pinned_input`, with identical behaviour; delete it from `app.py` and import `pinned_input`.
- `app.py` module budget: about 390 of 400 lines when done. Put helpers that do not need `app.py` internals in `cli/handlers.py` (110 lines).
- Tests patch `stencilizer.cli.app.FontProcessor` and `stencilizer.cli.app._classify_font` (`test_review_processing.py:69-117`, `test_cli_variable.py:175`): keep `_classify_font`, `_process_font`, `_run_standard` and the `FontProcessor` import in `app.py`.
- `cli/handlers.py` `_report_dry_run` (L92-110) prints `Bridge width          {width_percent}% of a reference stroke of 10% of font UPM`, with 2-space indented lines at L97-103.

## Pinned interface of `cli/pinning.py` (U1 writes it)

```python
@contextmanager
def pinned_input(input_font: Path, instance: str | None) -> Iterator[Path]: ...
def validate_instance(input_font: Path, instance: str) -> None: ...
def pin_stenciled(stenciled: Path, source: Path, instance: str, workdir: Path) -> Path: ...
def publish_pinned(pinned: Path, output_path: Path) -> Path: ...
def is_variable_font(path: Path) -> bool: ...
def is_cff2_font(path: Path) -> bool: ...
```

- `validate_instance` raises `InstanceSpecError` for a bad spec or a static input ("--instance requires a variable font").
- `pin_stenciled` swaps the source font's `name` table into the stenciled font, instantiates that at `instance` inside `workdir`, and returns the pinned path. It raises `InstanceSpecError` on a bad spec.
- `publish_pinned` writes the pinned font to `output_path` through `update_font_names` (it adds " Stenciled" once to the source names), staged in `output_path.parent`; it raises `FontSaveError`.
- `is_variable_font` and `is_cff2_font` raise `FontLoadError` when the file cannot be opened.

## Public surface (pinned; the frozen CLI contracts use it)

- Options:
  - `--width-scaling [fixed|proportional]`, default `fixed`;
  - `--scaling-strength FLOAT`, range 0-100, default 100;
  - `--min-bridge-width FLOAT`, range 10-110, default 30.

  All three go into `BridgeConfig(width_scaling=..., scaling_strength=..., min_width_percent=...)`. Help texts: "Variable fonts: keep bridge gaps fixed or scale them with each master's stroke weight", "Proportional width scaling: 0 (fixed) to 100 (fully proportional)", "Proportional width scaling: smallest gap as percent of a reference stroke (10-110)".
- Help: Typer 0.27.2 renders `--width-scaling` as `<fixed|proportional>`; tests assert option names only.
- Settings helper, called before `pinned_input` (never read `font_path`, the pinned temporary file, for these checks):
  - `PROPORTIONAL` and `not is_variable_font(input_font)`: `console.print` the static warning in yellow (unless `--quiet`): `Width scaling applies only to variable fonts; using fixed width.`
  - `elif` `PROPORTIONAL` and `instance` and `is_cff2_font(input_font)`: print, in yellow and unless `--quiet`, `Proportional width scaling with --instance is not supported for CFF2 fonts; pinning first with fixed width.`
  - In both cases switch the settings to fixed: `settings = settings.model_copy(update={"bridge": settings.bridge.model_copy(update={"width_scaling": BridgeWidthScaling.FIXED})})`. The run then proceeds as fixed mode, so a static `--instance` run fails at once with today's "--instance requires a variable font".
- Stencil-first flow, only when `width_scaling` is `PROPORTIONAL`, `instance` is set, and neither `--list-islands` nor `--dry-run` is given (those pin first in every mode; fixed mode keeps today's flow for every command):
  1. `validate_instance(input_font, instance)` first, so a bad spec fails before any processing with today's message.
  2. `with tempfile.TemporaryDirectory(prefix="stencilizer-instance-") as tmp:`
  3. Pass 1: stencil `input_font` into `Path(tmp) / f"{stem}-variable{suffix}"` through the same classify/process path as `_run_standard`, with progress and "Variable axes" as today and no success report for the temporary file. Split `_run_standard`'s body into a function that returns `ProcessingStats`.
  4. `pinned = pin_stenciled(stenciled, input_font, instance, Path(tmp))`.
  5. Pass 2 (put it in `handlers.py`): `FontProcessor` built from the same settings switched to fixed, with `quiet=True`, and the classification taken from `FontReader(pinned)`. When `glyphs_to_process` is non-empty, `processor.process(font_path=pinned, output_path=output_path, max_workers=workers, classification=...)` and `raise FontProcessingError(stats.errors)` on errors. Otherwise `publish_pinned(pinned, output_path)` and an empty `ProcessingStats()`.
  6. Map `KeyboardInterrupt` during the pin and pass 2 to `typer.Exit(code=130)` inside the `with`, as `_process_font` does.
  7. Report once for `output_path` with `dataclasses.replace(pass1, bridges_added=pass1.bridges_added + pass2.bridges_added, unbridged_count=pass2.unbridged_count)`. The quiet-mode unbridged warning uses the same count.

  A variable input with no islands keeps today's "Nothing to process" exit. Pass 2 writes through `FontWriter`, which adds " Stenciled" once to the source names `pin_stenciled` swapped in, so the names equal today's pin-first output ("Inter Variable Text Black Stenciled" at `wght=900`).
- Dry run: after the bridge-width line, print exactly `  Width scaling         fixed` or `  Width scaling         proportional (strength {s}%, minimum {m}% of a reference stroke)`, with `s` and `m` the floats from `settings.bridge` (for example `strength 40.0%, minimum 45.0%`) and the 2-space indent of `handlers.py:97-103`.

## Steps

1. Add the options (module-level `Annotated` aliases, inline options converted) and the settings wiring through `_bridge_config(...)`, keeping `app.py` under 400 lines and `stencilize` and `_run_command` under 50 effective lines. Moving `_instance_font` out frees 8 lines. Put helpers that do not need `app.py` internals in `cli/handlers.py`, never in `pinning.py` (U1 owns it).
2. Add the settings helper (static and CFF2 warnings) and the stencil-first `--instance` flow with its second pass.
3. Add the dry-run line.
4. Tests in `tests/unit/test_cli_width_scaling.py` (fixtures from `tests.font_helpers`: `INTER`, `CANTARELL`, `ROBOTO`; `CliRunner` pattern from `tests/unit/test_variable_surface_contracts.py` `_cli`):
   - `--help` lists the three options;
   - `--scaling-strength 150` exits 2, and `--min-bridge-width 5` exits 2;
   - `--dry-run --width-scaling proportional --scaling-strength 40 --min-bridge-width 45` prints "strength 40.0%, minimum 45.0%";
   - `ROBOTO --dry-run --width-scaling proportional` prints the static warning and "Width scaling         fixed";
   - `INTER --instance wght=900 --width-scaling proportional`: the output has no `fvar`, `o` and `ampersand` have no island (`tests.font_helpers.island_count` after `removeOverlaps`), name ID 1 is "Inter Variable Text Black Stenciled", and the non-quiet report names the `-o` path (flatten whitespace as `test_cli_variable.py:24-26` does);
   - a bad spec (`--instance nope=1` with proportional) fails before any processing with today's message;
   - `CANTARELL --instance wght=250 --width-scaling proportional` prints the CFF2 warning.

   Keep every existing test in `test_cli_variable.py` passing; change it only where `_instance_font`'s move forces an import change. Do not add proportional runs to `test_cli_variable.py:148-167`: it asserts "Variable axes" is absent for fixed `--instance` runs.

## Done when

```bash
uv run pytest tests/unit/test_width_scaling_contracts.py tests/unit/test_cli_width_scaling.py tests/unit/test_cli_variable.py tests/unit/test_variable_surface_contracts.py tests/regression/test_code_structure.py --no-cov -q -p no:cacheprovider
env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help
```

Both pass once W1 and U1 are in, along with ruff and mypy on your files. If W1's engine is not in yet when you finish, report which contracts still fail and why.
