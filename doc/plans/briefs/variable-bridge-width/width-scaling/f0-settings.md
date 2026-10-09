# F0: width-scaling settings (stage width-scaling, foundation, runs alone first)

Tier: haiku (`loom-software-engineer`, `model: haiku`). Never run git. Read `_shared.md` in this directory first.

You own:

- `src/stencilizer/config/settings.py`
- `src/stencilizer/config/__init__.py`

Read-only: everything else.

## Steps

1. In `settings.py`, add `BridgeWidthScaling(StrEnum)` directly after `BridgeDirection` (L76-81). Match `BridgeDirection`'s docstring and inline-comment style; the members and values are pinned in `_shared.md`.
2. Add three fields to `BridgeConfig` (L84-100), after `direction`, each with a `description` in the style of `width_percent`:
   - `width_scaling`: "How a variable font's bridge gaps change across masters (fixed: the default master's gap everywhere; proportional: each gap follows the stroke it cuts)"
   - `scaling_strength`: "Proportional mode: how strongly gaps follow stroke thickness, 0 (fixed) to 100 (fully proportional)"
   - `min_width_percent`: "Proportional mode: the smallest gap, as a percentage of the reference stroke, never above the default master's gap"
3. In `config/__init__.py`, add `BridgeWidthScaling` to the `from stencilizer.config.settings import (...)` list (L13-20) and to `__all__` (L22-29), keeping both alphabetical. `BridgeDirection` is not exported there; do not add it.

## Done when

- This command exits 0. It checks the defaults, the dump round trip, and that `BridgeConfig(scaling_strength=101)` and `BridgeConfig(min_width_percent=9)` each raise `ValidationError`.

  ```bash
  uv run python -c "
  from pydantic import ValidationError
  from stencilizer.config import BridgeConfig, BridgeWidthScaling
  c = BridgeConfig()
  assert c.width_scaling is BridgeWidthScaling.FIXED and c.scaling_strength == 100.0 and c.min_width_percent == 30.0
  assert BridgeConfig(**c.model_dump()) == c
  for bad in ({'scaling_strength': 101}, {'min_width_percent': 9}):
      try:
          BridgeConfig(**bad)
      except ValidationError:
          continue
      raise SystemExit(f'no ValidationError for {bad}')
  "
  ```

- ruff check, ruff format --check and mypy pass on `src/stencilizer/config` (run through `uv run`).
- `uv run pytest tests/regression/test_code_structure.py --no-cov -q -p no:cacheprovider` passes.

Report the two changed files and the command results.
