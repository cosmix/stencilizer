# Shared context: stage width-scaling

Read this file and your own brief in full before anything else. Never run git. Never edit a frozen contract file:
`tests/unit/test_width_scaling_contracts.py` and `tests/gui/test_width_scaling_gui_contracts.py` (the contract session wrote them; `loom stage contracts show width-scaling` lists them).

## The feature

Variable fonts get a bridge width option. The user picks one of two modes:

- **fixed** (the default): every bridge has the same gap in every master. The gap is `width_percent / 100 * (0.1 * upm)`, as the default master gets it today.
- **proportional**: each bridge's gap in a master follows the thickness of the stroke the bridge cuts in that master. Bolder masters get wider gaps and lighter masters thinner ones.

The default master is identical in both modes: switching modes changes only the non-default masters.

## Settings (written by F0 before anyone else starts)

`src/stencilizer/config/settings.py`:

```python
class BridgeWidthScaling(StrEnum):
    """How a variable font's bridge gaps change from master to master."""

    FIXED = "fixed"  # the default master's gap in every master
    PROPORTIONAL = "proportional"  # each gap follows the thickness of the stroke it cuts


class BridgeConfig(BaseModel):
    ...existing fields unchanged...
    width_scaling: BridgeWidthScaling = Field(default=BridgeWidthScaling.FIXED, ...)
    scaling_strength: float = Field(default=100.0, ge=0.0, le=100.0, ...)
    min_width_percent: float = Field(default=30.0, ge=10.0, le=110.0, ...)
```

`BridgeWidthScaling` is exported from `stencilizer.config` beside `BridgeDirection`.

## The gap formula (W1 implements it; everyone else relies on it)

For one bridge (a pair of facing bridge lines) in one master:

- `base` is the bridge's gap in the default output: the distance between its two line coordinates. That is `width_percent`'s gap, except where core surgery narrowed it (`core/multi_island_merge.py:39-48`) or reused a previous gap (`core/surgery_nested.py:62-63`).
- `ratio` is the stroke thickness the bridge cuts in this master divided by the same in the default. Measure it along the bridge's centre line through the merged (flattened, overlap-removed) input outline. Within the cross extent of the pair's cut points, sort the crossings and add up the alternate intervals (the ink intervals).
- `minimum = min(min_width_percent / 100 * 0.1 * upm, base)`: the minimum never exceeds the base gap.
- fixed: `gap = base`
- proportional: `gap = max(minimum, base * ratio ** (scaling_strength / 100))`

The pair's master centre is the mean of the two lines' fixed-parameter targets, which is what `replay.py` `_realign` computes today. The lines go to `centre - gap / 2` and `centre + gap / 2`, and the existing crossing search and projection then run against those targets.

## Fallback (no glyph bridges less often than today)

`variable/transform.py` tries the configured mode first. When replay or validation rejects the glyph, it retries in fixed mode. When that also fails, it retries with today's per-line mean targets, with no pairing at all. Only when all three fail is the glyph left unchanged, with its islands counted. Fixed mode starts at the second step.

## Measured at HEAD (2026-10-09)

A synthetic square ring: outer (0,0)-(1000,1000), counter 200..800, UPM 1000, direction vertical, width 60%, so the base gap is 60. A bold master has its counter at 300..700, a thin master at 50..950. Today's replay gives gaps of 50 at bold, 75 at thin and 55 at wght 0.5. Under the new rules, fixed mode must give 60 everywhere. Proportional mode must give 90 at bold (ink 600 against 400) and `max(30, 15) = 30` at thin.

Inter `o` (2048 UPM, base gap 122.88) today gives 117 at Thin, 123 at default and 109 at Black. The stroke the vertical bridge cuts is 46, 161 and 300 thick in those masters.

## Code limits and gates

- `tests/regression/test_code_structure.py` fails any `src/stencilizer` file over 400 lines. `variable/replay.py` is at 397, `cli/app.py` at 340 and `gui/session.py` at 389 (do not touch it). Functions stay under 50 lines.
- Type checking: mypy strict with the pydantic plugin (`init_forbid_extra`). Lint and format: ruff (line length 100).
- Run once, scoped to your files: `uv run pytest <your test files> --no-cov -q -p no:cacheprovider`, then `uv run ruff check <your files>`, `uv run ruff format --check <your files>` and `uv run mypy src/stencilizer tests`. GUI tests need `QT_QPA_PLATFORM=offscreen` set on the command line, because the host session exports `wayland;xcb`.
- Never mention Claude or any AI system in code, comments or docs. Record mistakes, decisions and surprises with `loom memory note` / `loom memory decision`, never Claude Code auto-memory.
