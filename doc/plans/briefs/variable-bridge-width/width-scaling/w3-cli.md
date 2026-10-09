# W3: width-scaling CLI options and pin-after-stencil wiring (stage width-scaling)

Tier: sonnet (`loom-software-engineer`). Never run git. Read `_shared.md` in this directory first.

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

- `cli/app.py` is 340 lines (60 lines of room). Options are `Annotated` aliases at L57-72 (`BridgeWidthOption` is `min=30.0, max=110.0`). `stencilize` (L122-134) forwards everything to `_run_command` (L137-181). That builds `StencilizerSettings(bridge=BridgeConfig(width_percent=bridge_width), ...)` at L153-157, then opens `with _instance_font(input_font, instance) as font_path:` (L159), which pins first. It then dispatches to `_handle_list_islands`, `_handle_dry_run` or `_run_standard` (L236).
- `_instance_font` (L202-208) moves to `cli/pinning.py` as `pinned_input`, with identical behaviour; delete it from `app.py` and import `pinned_input`.
- `cli/handlers.py` `_report_dry_run` (L92-110) prints `Bridge width          {width_percent}% of a reference stroke of 10% of font UPM`.

## Pinned interface of `cli/pinning.py` (U1 writes it)

```python
@contextmanager
def pinned_input(input_font: Path, instance: str | None) -> Iterator[Path]: ...
def pin_stenciled(stenciled: Path, instance: str, output_path: Path) -> Path: ...
def is_variable_font(path: Path) -> bool: ...
```

`pin_stenciled` instantiates an already-stenciled variable font at `instance` and writes the result to `output_path`. It raises `InstanceSpecError` on a bad spec, like `instantiate_static`.

## Public surface (pinned; the frozen CLI contracts use it)

- Options:
  - `--width-scaling [fixed|proportional]`, default `fixed`;
  - `--scaling-strength FLOAT`, range 0-100, default 100;
  - `--min-bridge-width FLOAT`, range 10-110, default 30.

  All three go into `BridgeConfig(width_scaling=..., scaling_strength=..., min_width_percent=...)`. Help texts: "Variable fonts: keep bridge gaps fixed or scale them with each master's stroke weight", "Proportional width scaling: 0 (fixed) to 100 (fully proportional)", "Proportional width scaling: smallest gap as percent of a reference stroke (10-110)".
- With `--width-scaling proportional` and a static input font, print exactly `Width scaling applies only to variable fonts; using fixed width.` (unless `--quiet`), then proceed in fixed mode.
- With `--width-scaling proportional` and `--instance`, the write path (`_run_standard`) stencils the VARIABLE input into a temporary directory (`tempfile.TemporaryDirectory(prefix="stencilizer-instance-")`), then calls `pin_stenciled(temp_output, instance, output_path)`. The success report names `output_path`. `--list-islands` and `--dry-run` keep pinning first; they only analyse. Fixed mode keeps today's pin-first flow for every command.
- Dry run: after the bridge-width line, print `Width scaling         fixed`, or `Width scaling         proportional (strength {s}%, minimum {m}% of a reference stroke)`.

## Steps

1. Add the options and the settings wiring, keeping `app.py` under 400 lines. Moving `_instance_font` out frees 8 lines. If the pin-after-stencil branch does not fit, put a helper in `cli/handlers.py`, never in `pinning.py` (U1 owns it).
2. Add the static-font warning and the proportional `--instance` branch.
3. Add the dry-run line.
4. Tests in `tests/unit/test_cli_width_scaling.py`:
   - `--help` lists the three options;
   - an out-of-range `--scaling-strength 150` exits 2;
   - `--dry-run --width-scaling proportional --scaling-strength 40` prints the proportional line;
   - with `--instance wght=900 --width-scaling proportional` on Inter, the output has no `fvar`, `o` has no island after `removeOverlaps`, and the run log or report names the given output path.

   Keep every existing test in `test_cli_variable.py` passing; change it only where `_instance_font`'s move forces an import change.

## Done when

`uv run pytest tests/unit/test_width_scaling_contracts.py tests/unit/test_cli_width_scaling.py tests/unit/test_cli_variable.py tests/unit/test_variable_surface_contracts.py --no-cov -q -p no:cacheprovider` passes once W1 and U1 are in, along with ruff and mypy on your files and `env NO_COLOR=1 COLUMNS=200 uv run stencilizer --help`. If W1's engine is not in yet when you finish, report which contracts still fail and why.
