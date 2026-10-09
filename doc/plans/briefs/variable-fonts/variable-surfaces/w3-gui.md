# W3: GUI variable sessions and axis sliders (stage variable-surfaces, wave 2)

Tier: sonnet (`loom-software-engineer`). This is routine UI wiring onto the existing card design. Never run git.

You own:

- `src/stencilizer/gui/session.py`
- `src/stencilizer/gui/variable_session.py` (create)
- `src/stencilizer/gui/controller.py`
- `src/stencilizer/gui/controls.py`
- `src/stencilizer/gui/axis_controls.py` (create)
- `src/stencilizer/gui/main_window.py`
- `src/stencilizer/gui/theme.py`
- `tests/gui/test_session.py`
- `tests/gui/test_axis_controls.py` (create)
- `tests/gui/test_variable_session.py` (create)

Read-only:

- `tests/gui/conftest.py`;
- `src/stencilizer/variable/*` (`classify_variable_glyphs(processor, reader)` from W1, `transform_variable_glyph`, `VariableGlyph.instance`);
- frozen: `tests/gui/test_variable_gui_contracts.py`, `tests/gui/test_beautify_contracts.py` (it monkeypatches `controller.set_parameters(bridge, max_workers)` and reads `window.controls.workers_slider`; keep both).

`variable_session.py` and `session.py` stay Qt-free: only controller, tasks and widgets import PySide6 (knowledge conventions "Qt and GUI code"). The stage acceptance checks that importing `stencilizer.gui.session` and `stencilizer.gui.variable_session` loads no PySide6.

## Current code (from planning exploration)

Line numbers are at HEAD 770240f. Stage `cff2-static` has since edited `gui/session.py` (`unsupported_reason`) and `tests/gui/test_session.py` (`test_open_rejects_unsupported_fonts`), so locate code by the symbol and test names given here; the numbers are hints.

**`gui/session.py` (315 lines; class `FontSession` 230 of 300 lines).**
- `unsupported_reason` (lines 61-69) still rejects `fvar`.
- `FontSession.open(path, processor)` (lines 113-149) reads the file once inside `with FontReader(...)` (the reader is closed after line 120's block) and caches default `Glyph`s through `processor.classify_glyphs(reader)`.
- `preview(name, bridge, geometry, directions=None) -> PreviewResult` (lines 235-241) calls `process_glyph` on the cached glyph (`_preview_island`, lines 172-205); `_preview_composite` (line 220) previews each source through `_preview_island`.
- `unbridged(...)` (lines 252-268) surveys all island glyphs.
- `save(...)` (lines 280-308) calls `processor.process(..., classification=...)`, which dispatches variable fonts after W1.

**`gui/controller.py` (256 lines).**
- `set_parameters(bridge, max_workers)` (lines 129-134) refreshes the preview and schedules the survey.
- `_refresh_preview` (lines 165-179) runs `session.preview` synchronously on the GUI thread and catches only `StencilizerError` (line 176).

**`gui/controls.py` (160 lines).** `ControlPanel._build_layout` (lines 74-84) holds two cards, BRIDGES and PROCESSING, each built from `_section_title(...)` plus a `QFrame` with `setProperty("role", "card")` (lines 88-89), then `addStretch()`. The sidebar is 240-360 px wide (lines 40-41) and has no scroll area.

**`gui/main_window.py` (264 lines).** Window minimum 960×600 (line 46). `_update_parameters` (lines 180-182) calls `controller.set_parameters`; `_on_font_loaded` (lines 199-215) fills the grid and header and selects the first glyph.

**`gui/theme.py`.** Line 80 styles only `QFrame[role="card"]`; lines 160-180 style only `QSpinBox` (`QDoubleSpinBox` is not a subclass, so it gets no rule). `tests/gui/test_theme.py:42-69` only checks that its listed selectors exist, so adding selectors is safe.

## Steps

1. **`variable_session.py`** holds the variable logic so `session.py` stays ≤400 lines and `FontSession` ≤300.
   - `AxisInfo` frozen dataclass: `tag`, `name`, `minimum`, `default`, `maximum`, from fvar plus the name table; `name` falls back to the tag when the axis name record is missing (the `variable_font_path` fixture has no nameID 256).
   - `read_axes(font) -> tuple[AxisInfo, ...]` and `read_avar(font) -> dict[str, dict[float, float]]` (`dict(font["avar"].segments)` when present, else `{}`). Both run inside `FontSession.open`'s reader block: the font is closed afterwards.
   - `normalize(location_user, axes, avar_segments) -> dict[str, float]`: `fontTools.varLib.models.normalizeLocation(location_user, {a.tag: (a.minimum, a.default, a.maximum)})`, then `fontTools.varLib.models.piecewiseLinearMap(v, avar_segments[tag])` per tag that has segments. All three fixtures have avar: Inter wght 700 is 0.6 by fvar alone and about 0.54 after avar.
   - A `VariableOutcomeCache` class owned by the `FontSession` (a new session on every open, so reopening drops it). Key: `(name, bridge.model_dump_json(), geometry.model_dump_json(), direction)`, every input of `transform_variable_glyph` except `upm`, which is fixed per session (`BridgeConfig` is not hashable). Outcomes live in a `collections.OrderedDict` capped at `max_entries` (constructor parameter, 64 in `FontSession`), least recently used evicted first. A request whose bridge or geometry JSON differs from the cached entries' clears the cache, so it holds one parameter set. The survey (`FontSession.unbridged`, run on a pool thread by `gui/controller.py` `_run_survey`) stores only `bridge_count` per key in a second dict and never outlines, so surveying a large font cannot fill memory with outlines; it reuses a cached outcome when one exists. One `threading.Lock` guards both dicts, held only for lookups and inserts, never while `transform_variable_glyph` runs (two threads may each compute a key once; the second insert wins and both results are equal).
   - Measure the cold preview (cache miss) time of every fixture island glyph once and record the worst in loom memory. If any exceeds 100 ms, record it with `loom memory note` and report it to the orchestrator: moving previews off the GUI thread is a separate change.
   - `preview_variable(vg, bridge, geometry, upm, location_norm) -> (original: Glyph, stenciled: Glyph | None, bridges: int)`: run `transform_variable_glyph` once per cache key, then return `vg.instance(loc)` and `outcome.glyph.instance(loc)`, or `None` as stenciled when there are 0 bridges. A slider move then costs only `instance()` (deltas are cached in the model).
2. **`session.py`:**
   - `unsupported_reason` stops rejecting `fvar`;
   - for a variable font, `open` calls `classify_variable_glyphs(processor, reader)` (W1) inside the reader block to get both the classification and the `VariableGlyph` dict, and stores `axes`, the avar segments and that dict; static fonts keep `processor.classify_glyphs(reader)`;
   - `FontSession.axes: tuple[AxisInfo, ...]` (empty for static fonts);
   - `preview(name, bridge, geometry, directions=None, location=None)`, where `location` is user-space `{tag: value}` and `None` means fvar defaults; `directions` stays the 4th positional parameter (`tests/gui/test_session_directions.py:101` passes it positionally);
   - for variable sessions, island previews go through `preview_variable`; `_preview_composite` takes each source's stenciled outline from `preview_variable` at `{}` (the default master) and composes with the default component offsets (the seam `gui/composites.py` `compose` built by the bridge-direction plan, unchanged for static fonts), so a composite on Inter `A` shows the bridge the saved font gets at the default location. The static `process_glyph` path finds no counter in overlap-built glyphs. Composites ignore `location`. That is a product rule the window states (steps 3 and 7) and the frozen contract `test_composite_preview_at_default` pins; following composite gvar offsets per location is out of scope;
   - `FontSession.is_composite(name) -> bool` (True for names in the composite index);
   - glyphs in `classification.unsupported_islands` are display glyphs too: their preview returns the original outline (read at open with `fonttools_glyph_to_domain` on the default glyph set), no stenciled outline, and the error `"unsupported variation data"`; the survey reports them as unbridged;
   - the survey (`unbridged`) uses the outcomes' `bridge_count`.
3. **`axis_controls.py`:** `AxisPanel(QFrame)` with `setProperty("role", "card")` in the constructor (a `QWidget` with that role gets no style or background), and signal `location_changed = Signal(dict)`.
   - `set_axes(axes: tuple[AxisInfo, ...])` builds one row per axis inside a `QScrollArea` (fonts with many axes would otherwise push the sidebar past the 600 px window minimum): a label with the axis name, a `QSlider` with object name `axis-slider-<tag>`, and a `QDoubleSpinBox`.
   - Sliders run over integer steps from minimum to maximum, scaled ×10 when the range is under 50 (Ubuntu wdth 75..100).
   - `location()` returns `{tag: value}`.
   - An empty tuple hides the panel.
   - `set_location_applies(applies: bool)`: False disables every slider and spin box and shows a word-wrapped `QLabel` with object name `axis-composite-note` and text "Composite glyphs preview at the default axis location."; True re-enables them and hides the label. Style the label with existing theme tokens only.
4. **`theme.py`:** extend the spin-box rules at lines 160-180 to `QSpinBox, QDoubleSpinBox` (and their sub-controls), using existing tokens only.
5. **`controls.py`:** add an "AXES" `_section_title` and an `AxisPanel` between the PROCESSING card and `addStretch()`. Expose it as `ControlPanel.axes_panel`. The title is hidden when there are no axes.
6. **`controller.py`:** `set_location(location: dict[str, float])` stores it, then calls `_refresh_preview()`. `_refresh_preview` passes `location=self._location`. `_on_font_loaded` resets the location to the session's axis defaults. Location changes do not reschedule the survey (bridge counts do not depend on location).
7. **`main_window.py`:**
   - `_on_font_loaded` calls `controls.axes_panel.set_axes(session.axes)`;
   - connect `controls.axes_panel.location_changed` to `self.controller.set_location`, directly as `main_window.py:123` does for `select_glyph`, or through a slot that calls it. The stage wiring check greps `controller\.set_location\b` in main_window.py, which matches either;
   - `_on_glyph_selected` calls `self.controls.axes_panel.set_location_applies(not session.is_composite(name))` for the current session (the stage wiring check greps `set_location_applies\(` in main_window.py);
   - the open dialog filter stays "Fonts (*.ttf *.otf)".
8. **Tests:**
   - `tests/gui/test_session.py`: `test_open_rejects_unsupported_fonts` still asserts that `variable_font_path` (Roboto plus an fvar with no gvar) is rejected. Rewrite that assertion, keeping the `classify_glyphs` stub at :75, into a successful open whose `session.axes` has one `wght` axis named `wght`. Never delete a test or assertion: the stage's test-integrity check fails on fewer declarations or assertions.
   - `tests/gui/test_axis_controls.py`:
     - `AxisPanel` builds one slider per axis for Ubuntu-VF-subset (`wdth`, `wght`);
     - moving a slider emits `location_changed` with the user value;
     - an empty axes tuple hides the panel;
     - `normalize({"wght": 700}, inter_axes, inter_avar)` is about 0.54 (not 0.6);
     - `set_location_applies(False)` disables the sliders and shows `axis-composite-note`; `True` reverses it.
   - `tests/gui/test_variable_session.py` (Qt-free session tests):
     - a `VariableOutcomeCache(max_entries=4)` given outcomes for 6 keys holds the 4 most recently used; after `unbridged(...)` over Inter-VF-subset the session's survey dict holds only integers, never a `Glyph` or `VariableGlyph`;
     - two previews of `o` differing only in `GeometryConfig` miss the cache (they compute twice: count `transform_variable_glyph` calls with a wrapper);
     - a preview and a survey running on two threads at once (a `ThreadPoolExecutor` with 2 workers) return equal bridge counts and raise nothing.

Helpers `load_session(controller, qtbot, path)` (conftest.py:125) and the `processor` fixture are in `tests/gui/conftest.py`; the `window` fixture is defined per test file, not in conftest. Set `QT_QPA_PLATFORM=offscreen` on every Qt command line: the host desktop exports `QT_QPA_PLATFORM=wayland;xcb`, which beats a conftest `setdefault` and opens real windows (stage `cff2-static` makes conftest assign it, but never rely on that alone).

## Proof

```bash
QT_QPA_PLATFORM=offscreen uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_session.py tests/gui/test_axis_controls.py tests/gui/test_variable_session.py tests/gui/test_controls.py tests/gui/test_controller.py tests/gui/test_variable_gui_contracts.py
```

Size limits: files ≤400 lines, functions ≤50 effective lines, classes ≤300 (`tests/regression/test_code_structure.py`). The verifier runs the full gate.
