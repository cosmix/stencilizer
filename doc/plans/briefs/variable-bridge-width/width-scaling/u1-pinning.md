# U1 (codex unit): cli/pinning.py and its test (stage width-scaling)

Lane: codex, `--model gpt-5.6-terra --effort xhigh`, through `loom-codex-forwarder` in the foreground. Do not run git. Do not touch any path under `.loom/`.

Files owned (write): `src/stencilizer/cli/pinning.py` (new), `tests/unit/test_cli_pinning.py` (new).
Files read:

- `src/stencilizer/cli/app.py` (`_instance_font`, about L202-208)
- `src/stencilizer/io/instance.py` (`instantiate_static` L65, `parse_instance_spec` L25, `InstanceSpecError`)
- `src/stencilizer/io/writer.py` (`update_font_names`, L30-76)
- `src/stencilizer/exceptions.py` (`FontLoadError`, `FontSaveError`, both `(path: str, reason: str)`)
- `src/stencilizer/variable/reader.py` (`is_variable`)
- `tests/font_helpers.py` (`INTER`, `CANTARELL`, `ROBOTO`, `units_per_em`, `island_count`)
- `tests/unit/test_cli_variable.py` (style)

## Steps

1. Create `src/stencilizer/cli/pinning.py` with a module docstring and exactly these six functions:
   - `pinned_input(input_font: Path, instance: str | None) -> Iterator[Path]`, decorated with `contextlib.contextmanager`. Its body is identical to `_instance_font` in `app.py` (app.py:202-208): with no instance it yields `input_font`; otherwise it yields `instantiate_static(input_font, instance, Path(tmp))` inside `tempfile.TemporaryDirectory(prefix="stencilizer-instance-")`. Do not edit `app.py`; another worker removes the old function.
   - `validate_instance(input_font: Path, instance: str) -> None`. Open `TTFont(input_font, lazy=True)`, call `stencilizer.io.instance.parse_instance_spec(instance, font)`, and close the font in a `finally` block. `InstanceSpecError` propagates; any other open error becomes `FontLoadError(str(input_font), str(error))`.
   - `pin_stenciled(stenciled: Path, source: Path, instance: str, workdir: Path) -> Path`. Load `stenciled`, replace its `name` table with a `copy.deepcopy` of the `source` font's `name` table (`update_font_names` edits existing records only, so the IDs match), save it to `workdir / f"{stenciled.stem}-renamed{stenciled.suffix}"`, and return `instantiate_static(that_path, instance, workdir)`. `instantiate_static` writes `workdir / f"{stem}-instance{suffix}"` and raises `InstanceSpecError` before writing. Close every `TTFont`.
   - `publish_pinned(pinned: Path, output_path: Path) -> Path`. Inside `tempfile.TemporaryDirectory(prefix=".stencilizer-pin-", dir=output_path.parent)`: load `pinned`, call `stencilizer.io.writer.update_font_names(font)`, save into the temporary directory, close the font, and `Path(staged).replace(output_path)` (never `shutil.move`: it moves into a directory destination and copies non-atomically across filesystems). Wrap `OSError` (a missing parent directory and a directory `output_path` included) in `FontSaveError(str(output_path), str(error))`. Return `output_path`.
   - `is_variable_font(path: Path) -> bool` and `is_cff2_font(path: Path) -> bool`. Open `TTFont(path, lazy=True)` and return `"fvar" in font` (or `stencilizer.variable.reader.is_variable(font)`) and `"CFF2" in font` respectively; close the font in a `finally` block. An open failure becomes `FontLoadError(str(path), str(error))`.
2. Create `tests/unit/test_cli_pinning.py` (fixtures `INTER`, `CANTARELL`, `ROBOTO` from `tests.font_helpers`):
   - `pinned_input` with `None` yields the same path;
   - `pinned_input` with `"wght=700"` on `INTER` yields an existing file without `fvar` inside the block, and the file is gone after it;
   - `validate_instance` raises `InstanceSpecError` for `"nope=1"` on `INTER` and for `ROBOTO` ("--instance requires a variable font");
   - `pin_stenciled` on `INTER` (as both `stenciled` and `source`) with `"wght=900"` and `tmp_path` as `workdir` returns a path inside `tmp_path` without `fvar`;
   - `publish_pinned` writes `tmp_path / "out.ttf"` with " Stenciled" in name ID 1 and leaves no other file in `tmp_path`;
   - `publish_pinned` to a missing directory raises `FontSaveError` and writes nothing;
   - `publish_pinned` over an existing `tmp_path / "out.ttf"` replaces it (the new file has " Stenciled" in name ID 1);
   - `publish_pinned` with `output_path` an existing directory raises `FontSaveError`, and the directory and its contents are unchanged;
   - with `Path.replace` monkeypatched to raise `OSError`, `publish_pinned` over an existing `out.ttf` raises `FontSaveError`, `out.ttf` keeps its previous bytes, and no staged file or `.stencilizer-pin-*` directory is left in `tmp_path`;
   - `is_variable_font` is True for `INTER` and False for `ROBOTO`;
   - `is_cff2_font` is True for `CANTARELL` and False for `INTER`;
   - `is_variable_font` on a text file raises `FontLoadError`.
3. Prove it statically, with the project's tools called directly (the codex sandbox has no network, a read-only `~/.cache/uv` and no writable `/tmp`, so `uv run` fails there; knowledge `mistakes.md` "Codex worker briefs told to verify with uv run"):

   ```bash
   .venv/bin/mypy src/stencilizer/cli/pinning.py tests/unit/test_cli_pinning.py && .venv/bin/ruff check src/stencilizer/cli/pinning.py tests/unit/test_cli_pinning.py && .venv/bin/ruff format --check src/stencilizer/cli/pinning.py tests/unit/test_cli_pinning.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/unit/test_cli_pinning.py
   ```

   Do not run the tests themselves: the orchestrator runs them with `uv run` afterwards, because pytest's `tmp_path` is not writable in codex's sandbox.

Constraints: mypy strict (annotate everything; fontTools imports carry `# type: ignore[import-untyped]`, as in `io/instance.py`). Line length 100. Keep functions under 50 effective lines and the file under 400. No mention of any AI system.
