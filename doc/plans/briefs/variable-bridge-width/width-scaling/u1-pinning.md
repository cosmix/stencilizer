# U1 (codex unit): cli/pinning.py and its test (stage width-scaling)

Lane: codex, `--model gpt-5.6-terra --effort xhigh`, through `loom-codex-forwarder` in the foreground. Do not run git. Do not touch any path under `.loom/`.

Files owned (write): `src/stencilizer/cli/pinning.py` (new), `tests/unit/test_cli_pinning.py` (new).
Files read: `src/stencilizer/cli/app.py` (`_instance_font`, about L202-208), `src/stencilizer/io/instance.py` (`instantiate_static`, `InstanceSpecError`), `src/stencilizer/variable/reader.py` (`is_variable`), `tests/font_helpers.py` (`INTER`, `units_per_em`, `island_count`), `tests/unit/test_cli_variable.py` (style).

## Steps

1. Create `src/stencilizer/cli/pinning.py` with a module docstring and exactly these three functions:
   - `pinned_input(input_font: Path, instance: str | None) -> Iterator[Path]`, decorated with `contextlib.contextmanager`. Its body is identical to `_instance_font` in `app.py`: with no instance it yields `input_font`; otherwise it yields `instantiate_static(input_font, instance, Path(tmp))` inside `tempfile.TemporaryDirectory(prefix="stencilizer-instance-")`. Do not edit `app.py`; another worker removes the old function.
   - `pin_stenciled(stenciled: Path, instance: str, output_path: Path) -> Path`. Inside a `tempfile.TemporaryDirectory(prefix="stencilizer-pin-")`, call `instantiate_static(stenciled, instance, Path(tmp))`, move the result to `output_path` with `shutil.move` (replacing an existing file), and return `output_path`. Let `InstanceSpecError` propagate.
   - `is_variable_font(path: Path) -> bool`. Open the font with `TTFont(path, lazy=True)`, return `is_variable(font)`, and close the font in a `finally` block.
2. Create `tests/unit/test_cli_pinning.py`:
   - `pinned_input` with `None` yields the same path;
   - with `"wght=700"` on `INTER`, the yielded file exists inside the block, has no `fvar`, and is gone after the block;
   - `pin_stenciled` on `INTER` with `"wght=900"` writes `tmp_path / "out.ttf"`, returns that path, and the output has no `fvar`;
   - `pin_stenciled` with `"nope=1"` raises `InstanceSpecError` and writes no output file;
   - `is_variable_font` is True for `INTER` and False for `tests/fixtures/Roboto-Regular.ttf`.
3. Prove it: `uv run ruff check src/stencilizer/cli/pinning.py tests/unit/test_cli_pinning.py` and `uv run ruff format --check src/stencilizer/cli/pinning.py tests/unit/test_cli_pinning.py` pass, and `uv run python -c "import stencilizer.cli.pinning"` exits 0. The orchestrator runs the pytest file afterwards. It needs fixture fonts, which the codex sandbox cannot run.

Constraints: mypy strict (annotate everything; fontTools imports carry `# type: ignore[import-untyped]`, as in `io/instance.py`). Line length 100. No mention of any AI system.
