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
3. In `config/__init__.py`, import and export `BridgeWidthScaling` beside `BridgeDirection`, matching the existing import and `__all__` pattern (L14, L23).

## Done when

- `uv run python -c "from stencilizer.config import BridgeConfig, BridgeWidthScaling; c = BridgeConfig(); assert c.width_scaling is BridgeWidthScaling.FIXED and c.scaling_strength == 100.0 and c.min_width_percent == 30.0; assert BridgeConfig(**c.model_dump()) == c"` exits 0.
- `uv run ruff check src/stencilizer/config`, `uv run ruff format --check src/stencilizer/config` and `uv run mypy src/stencilizer/config` pass.

Report the two changed files and the command results.
