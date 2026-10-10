# Shared context: stage width-scaling

Read this file and your own brief in full before anything else. Never run git. Never edit a frozen contract file:
`tests/unit/test_width_scaling_contracts.py` and `tests/gui/test_width_scaling_gui_contracts.py` (the contract session wrote them; `loom stage contracts show width-scaling` lists them).

The plan is `doc/plans/IN_PROGRESS-PLAN-variable-bridge-width.md` (loom renames it from `PLAN-variable-bridge-width.md`; read whichever exists). Ignore any `doc/plans/codex-PLAN-*` file. Where this brief and the plan disagree, the plan wins.

## The feature

Variable fonts get a bridge width option. The user picks one of two modes:

- **fixed** (the default): every bridge has the same gap in every master. The gap is `width_percent / 100 * (0.1 * upm)`, as the default master gets it today.
- **proportional**: each bridge's gap in a master follows the thickness of the stroke the bridge cuts in that master. Bolder masters get wider gaps and lighter masters thinner ones.

For every glyph both modes bridge, the default master is identical in both modes and the same as today's: switching modes changes only the non-default masters. No glyph bridges less often than today (fallback chain below).

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

`BridgeWidthScaling` is a new export of `stencilizer.config` (added to its import and `__all__`). `BridgeDirection` is not exported there today and stays unexported: import it from `stencilizer.config.settings`.

## The gap formula and engine rules (W1 implements them exactly; everyone else relies on them)

For one bridge (a disjoint pair of bridge lines, rule 1) in one master:

- `base` is the bridge's gap in the default output: the distance between the pair's two default line coordinates. That is `width_percent`'s gap, except where core surgery narrowed it (`core/multi_island_merge.py:39-48`) or reused a previous gap (`core/surgery_nested.py:62-63`).
- `ratio = master ink / default ink` along the pair's centre line (rules 2-4).
- `minimum = min(min_width_percent / 100 * 0.1 * upm, base)`: the minimum never exceeds the base gap.
- fixed: `gap = base`
- proportional: `gap = max(minimum, base * ratio ** (scaling_strength / 100))`

Rules, measured with a prototype built only from HEAD modules:

1. **Pairing.** Sort the same-axis lines of a surgery map by default coordinate. Walking upward from the lowest, pair each unpaired line with the nearest unpaired line above it that has the same set of input contours (`{spans[m.edges[0]][0] for m in line.members}`, `spans` from `replay._contour_spans`) and an overlapping cross span. Leftover lines keep today's mean target. This matches every true pair on the fixtures: Inter B horizontal gives `{0,108}`/`{0,108}` and `{0,73}`/`{0,73}`, the spanning `eight` gives `{0,143,216}` on both lines. Pairing by nearest distance fails: Inter `.notdef` lines 941.44 and 946.56 are 5.12 apart and belong to different bridges.
2. **Centre and extent.** In a master, each member of the pair's two lines has a fixed-parameter position (the `_place` output that `_realign` averages today). The master centre is the mean of the two lines' fixed-parameter targets. The extent is the min..max cross-axis coordinate of those positions. In the default, the centre is the mean of the two line coordinates and the extent comes from the default positions. A default extent applied to a wider master under-measures it: Inter `o` horizontal at Black gives a ratio of 1.884 with the default's extent against 2.261 with its own.
3. **Ink.** Take every crossing of the centre line with the merged (flattened, overlap-removed) polygon. An edge counts when its two ends straddle the line, half-open; an edge lying on the line is skipped. Sort the crossings and pair them even-odd from the first one to get the ink intervals. Clip each interval to the extent, then sum. Filtering the crossings to the extent before pairing measures the counter on round glyphs: Inter `o`'s centre line crosses at y -23.9, 137.1, 970.9 and 1131.9 inside a cut extent of -20.7..1128.7, which gives 833.9 (the counter) instead of about 316.
4. **Ratio.** A default ink of 0 or less gives ratio 1. A master ink of 0 gives ratio 0; `0.0 ** 0.0 == 1`, so strength 0 gives `base` and any other strength gives the minimum.
5. **Placement.** The pair's line with the lower default coordinate goes to `centre - gap / 2`, the other to `centre + gap / 2`. That keeps the sign `validate._bridges_intact` checks (validate.py:113-115). The existing crossing search and projection then run against those targets.
6. **Fallback chain.** Configured mode, then fixed, then today's per-line mean targets with no pairing. Only when all three fail is the glyph left unchanged, with its islands counted. Fixed mode starts at the second step. The loop lives in `variable/transform.py` outside `_bridged`, which stays one attempt.

`align_to_lines` keeps today's per-line mean targets, so the default master is unchanged (the rounded default came out identical either way on 3 fixtures x auto/vertical/horizontal); `align.py` stays untouched. `validate._facing_pairs` (validate.py:81-97) returns every same-axis pair whose cross spans overlap, a superset of the disjoint pairs (6 pairs for the 4 lines of B, `eight` and `ampersand` with horizontal bridges; 91 for the 14 lines of Inter `.notdef`), and stays the validation rule only.

## CLI and `--instance` decisions

With `--width-scaling proportional`: a glyf variable font with `--instance` is stenciled, pinned with the source font's `name` table swapped in (so names equal today's pin-first output), then run through the ordinary static stencil as a second pass. A CFF2 variable font with `--instance` warns and pins first in fixed mode (stencil-then-pin leaves most Cantarell counters closed). A static font warns and switches the settings to fixed; that check reads `input_font`, never the pinned temporary file, so a static `--instance` run still fails at once with today's "--instance requires a variable font". `--list-islands` and `--dry-run` pin first in every mode.

## Measured at HEAD (2026-10-09)

A synthetic square ring: outer (0,0)-(1000,1000), counter 200..800, UPM 1000, direction vertical, width 60%, so the base gap is 60. A bold master has its counter at 300..700, a thin master at 50..950. Today's replay gives gaps of 50 at bold, 75 at thin and 55 at wght 0.5. Under the new rules, fixed mode must give 60 everywhere. Proportional mode must give 90 at bold (ink 600 against 400) and `max(30, 15) = 30` at thin (40 with `min_width_percent=40`); at strength 50 bold gives `60 * 1.5 ** 0.5 = 73.48`.

Narrow master (counter x 465..535, y 300..700): HEAD gap 34. In proportional mode the gap 90 cannot fit the 70-unit counter, so the glyph falls back to fixed (bold gap 60, `bridge_count` 1; the 5-unit slivers pass validation). Mean-only master (counter x 475..525, y 300..700): HEAD gap 32 (lines at x 484/516). The fixed gap 60 cannot fit the 50-unit counter, so only the third fallback step bridges it (gap below 50).

Inter `o` (2048 UPM, base gap 122.88) today gives 117 at Thin, 123 at default and 109 at Black. The stroke the vertical bridge cuts is 46, 161 and 300 thick in those masters.

### Baseline (default `BridgeConfig()`, c058f98)

| Fixture | Island glyphs bridged (`bridge_count > 0`) | CLI report: glyphs processed / bridges / unbridged |
| --- | --- | --- |
| Inter (`tests/fixtures/variable/Inter-VF-subset.ttf`) | 20 of 21 | 21 / 28 / 2 |
| Ubuntu (`Ubuntu-VF-subset.ttf`) | 20 of 21 | 21 / 22 / 2 |
| Cantarell (`Cantarell-VF-subset.otf`) | 20 of 21 | 21 / 22 / 2 |

In every fixture `ampersand` gets 2 default-master bridges, then replay or validation fails and the glyph is left unchanged.

### Prototype expectations (auto direction)

| Fixture | fixed | proportional |
| --- | --- | --- |
| Inter | 20 (19 at step fixed, `e` at step mean) | 20 (18 proportional, `a` fixed, `e` mean) |
| Ubuntu | 20 (all fixed) | 20 (15 proportional, 5 fixed) |
| Cantarell | 20 (all fixed) | 20 (18 proportional, 2 fixed) |

- `ampersand` fails every step, as at HEAD. No glyph is lost or gained against HEAD in any mode or direction. Horizontal bridges: Inter 21, Ubuntu 19, Cantarell 21.
- Inter `o` vertical: ratios 0.27 (Thin) and 1.88 (Black). Gaps at Thin/default/Black: HEAD 117/123/109, fixed 123 everywhere except 122 at the opsz master, proportional 61/123/231 (opsz+Black 241).
- Fixed and proportional defaults are identical to each other and to HEAD on 3 fixtures x 3 directions.

## Code limits and gates

- `tests/regression/test_code_structure.py` fails any `src/stencilizer` file over 400 lines and any class over 300. Current file sizes: `variable/replay.py` 397, `variable/transform.py` 125, `cli/app.py` 340, `cli/handlers.py` 110, `gui/controls.py` 173, `gui/main_window.py` 297, `gui/session.py` 393 (do not touch it). Classes: `ControlPanel` 141, `MainWindow` about 261.
- Functions stay under 50 effective lines, measured by `tests/regression/test_code_structure.py::_effective_lines`. `stencilize` is at 48, `_run_command` at 45, `ControlPanel._build_widgets` at 29.
- Every worker runs `uv run pytest tests/regression/test_code_structure.py --no-cov -q -p no:cacheprovider` before reporting.
- Type checking: mypy strict with the pydantic plugin (`init_forbid_extra`). Lint and format: ruff (line length 100).
- Run once, scoped to your files: `uv run pytest <your test files> --no-cov -q -p no:cacheprovider`, then ruff check and ruff format --check on your files and mypy on `src/stencilizer tests`, each through `uv run`. GUI tests need `QT_QPA_PLATFORM=offscreen` set on the command line, because the host session exports `wayland;xcb`.
- Scratch files (renders, probes) go under the session scratchpad directory named in your system prompt, never `$TMPDIR` (the Read tool cannot open files there) and never the worktree.
- Never mention Claude or any AI system in code, comments or docs. Record mistakes, decisions and surprises with `loom memory note` / `loom memory decision`, never Claude Code auto-memory.
