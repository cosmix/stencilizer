# W8: main_window.py (wave 3, codex gpt-5.6-terra)

Read `doc/plans/briefs/stencilizer-gui/gui-app/_shared.md` first; its contract for
`main_window.py` is binding. Every other gui module exists already: read them with
`cat src/stencilizer/gui/*.py` (the source graph does not show them).

## Files owned

W8 (wave 3) writes `src/stencilizer/gui/main_window.py` from Steps. W8T (wave 4) writes
`tests/gui/test_main_window.py` from Tests, reading the finished module with `cat`.

Read-only anchors: `ControlPanel` (controls.py), `GlyphGrid` (glyph_grid.py), `ComparisonView`
(glyph_view.py), `GuiController` (controller.py), `FontSession`/`PreviewResult` (session.py),
`FontWriter.get_stenciled_path` in `src/stencilizer/io/writer.py` (default save name
`<stem>-Stenciled<ext>`).

## Steps

1. Layout: window title "Stencilizer"; central widget a horizontal `QSplitter` holding
   `controls`, `grid`, `comparison` in that order; a status bar. Store the controller as the
   public attribute `controller`.
2. Wiring (all connections made in `__init__`):
   - `controls.open_requested` -> `open_font_dialog`; `controls.save_requested` ->
     `save_font_dialog`.
   - `controls.parameters_changed` and `controls.workers_spin.valueChanged` -> a method calling
     `controller.set_parameters(controls.bridge_config(), controls.max_workers())`; call it once
     at the end of `__init__` so the controller starts with the panel's values.
   - `grid.glyph_selected` -> `controller.select_glyph`.
   - `controller.font_loaded` -> fill the grid with `session.island_glyphs` (and the session's
     ascender/descender), set the font info text (`f"{path.name}\n{font_format}, {units_per_em}
     UPM\n{glyph_count} glyphs, {len(island_glyphs)} with islands"`),
     `controls.set_font_loaded(True)`, `comparison.clear()`, select the first island glyph when
     there is one, and show `f"Loaded {path.name}"` in the status bar.
   - `controller.preview_ready` -> `comparison.show_preview(result, session.ascender,
     session.descender)`.
   - `controller.save_progress` -> `controls.set_progress`; `controller.busy_changed` ->
     `controls.set_busy`.
   - `controller.save_finished` -> `controls.reset_progress()` and status
     `f"Saved {output.name}: {stats.processed_count} glyphs stencilized, {stats.error_count}
     errors"`, where `output` is the path remembered by `save_font`.
   - `controller.error` -> `controls.reset_progress()` and `QMessageBox.warning(self,
     "Stencilizer", message)`.
3. Actions: `load_font(path)` -> `controller.open_font(path)`; `save_font(path)` remembers the
   path, then `controller.save(path)`. `open_font_dialog()` uses
   `QFileDialog.getOpenFileName(self, "Open Font", "", "Fonts (*.ttf *.otf)")` and loads the
   chosen path (an empty string means cancelled). `save_font_dialog()` returns when no session is
   loaded; otherwise it offers `FontWriter.get_stenciled_path(session.path)` in
   `QFileDialog.getSaveFileName(self, "Save Stenciled Font", str(default), "Fonts (*.ttf *.otf)")`
   and saves to the chosen path. `closeEvent`: while `controller.is_busy`, `event.ignore()` and
   `statusBar().showMessage("Wait for the current operation to finish")`; otherwise
   `controller.shutdown()`, then `event.accept()` (`shutdown` waits on the pool with no
   deadline; mid-save it would freeze the window).

## Tests (`tests/gui/test_main_window.py`, `qtbot`)

Fixture: `controller = GuiController(tmp_path / "gui.log")`, `window = MainWindow(controller)`,
`qtbot.addWidget(window)`; teardown `controller.shutdown()`. Tests call `load_font`/`save_font`
directly; dialogs are exercised by monkeypatching `QFileDialog.getOpenFileName` /
`getSaveFileName` in the `stencilizer.gui.main_window` namespace. Monkeypatch
`QMessageBox.warning` there too (it would block), recording its calls.

- `load_font(roboto_path)` inside `qtbot.waitSignal(controller.font_loaded, timeout=30000)`:
  `grid.count() == 562`; the save button is enabled; the first island glyph is selected, so
  `comparison.after_canvas.glyph` is not None (use `qtbot.waitUntil` if needed).
- After selecting `O` (`grid.select_glyph("O")`), moving `controls.width_slider` to 30 and then
  110 changes `comparison.after_canvas.glyph.to_dict()`.
- `save_font(tmp_path / "out.ttf")` inside `qtbot.waitSignal(controller.save_finished,
  timeout=120000)`: `statusBar().currentMessage()` starts with `"Saved out.ttf: 562 glyphs
  stencilized, 0 errors"`.
- `test_saved_font_matches_preview`: after the load, `grid.select_glyph("O")`,
  `controls.workers_spin.setValue(1)`, move `controls.width_slider` to 30, and keep
  `expected = comparison.after_canvas.glyph` (the preview on screen). `save_font(tmp_path /
  "w30.ttf")` inside `waitSignal(save_finished, timeout=120000)`; the stats carry
  `error_count == 0`; `with FontReader(w30) as reader:` `outlines_match(reader.get_glyph("O"),
  expected)` is True. A window that saves default settings, or a save whose glyph writes
  failed, fails this.
- `load_font` on a `b"not a font"` file: the recorded `QMessageBox.warning` message starts with
  `"Failed to load font"`.
- `test_unsupported_font_is_rejected`: `load_font(cff2_font_path)`, then `qtbot.waitUntil` the
  recorded warning: its message starts with `"Failed to load font"` and contains `"CFF2"`;
  `controller.session` is None, `grid.count() == 0` and `controls.save_button.isEnabled()` is
  False (the rejected font never becomes a session, so save stays disabled).
- `open_font_dialog` with `getOpenFileName` patched to return `(str(roboto_path), "")` loads the
  font; patched to return `("", "")` loads nothing (no `busy_changed` emission).
- `window.close()` returns without hanging after a load has finished.
- Busy reflection: synchronously after `save_font(tmp_path / "out.ttf")` (before waiting),
  `controls.save_button.isEnabled()` and `controls.open_button.isEnabled()` are False; once
  `save_finished` has arrived both are True and `controls.progress_bar.isVisibleTo(controls)`
  is False.
- `test_close_while_busy_is_refused`: `window.show()`; after `save_font(...)`, `window.close()`
  returns False and the status bar shows `"Wait for the current operation to finish"`; after
  `save_finished`, `window.close()` returns True.
- Workers reach the controller: `monkeypatch.setattr(controller, "set_parameters", recorder)`,
  then `controls.workers_spin.setValue(1)`: the last recorded call has `max_workers == 1`.

## Proof command

W8 (module unit):

```bash
.venv/bin/mypy src/stencilizer/gui/main_window.py && .venv/bin/ruff check src/stencilizer/gui/main_window.py
```

W8T (test unit):

```bash
.venv/bin/mypy src/stencilizer/gui/main_window.py tests/gui/test_main_window.py && .venv/bin/ruff check tests/gui/test_main_window.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_main_window.py
```
