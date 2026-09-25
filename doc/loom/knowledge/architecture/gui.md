# Gui

> GUI package layout, threading model, save safety

## GUI package layout

`stencilizer-gui = "stencilizer.gui.app:main"` (pyproject.toml:20) launches `src/stencilizer/gui/`. The CLI is untouched: nothing under cli/core/io/config imports `stencilizer.gui`, the package `__init__` imports nothing, and PySide6 lives only in the optional `gui` extra.

| Module | Role |
| --- | --- |
| `app.py` | `build_parser`, `default_log_file`, `create_window`, `main` (sets the `spawn` start method, then `apply_theme(application)`, app.py:57-60) |
| `theme.py` | `ThemeColors` tokens for LIGHT and DARK, `palette_for`, `stylesheet_for`, `apply_theme`; owns every colour and QSS rule (see "Theme and styling") |
| `header.py` | `HeaderBar`: app title, font name and details, "Open Font…" (`role=secondary`) and "Stencilize & Save…" (`role=primary`) buttons |
| `session.py` | `FontSession`: Qt-free open, `display_glyphs` (island glyphs plus bridged composites), `direction_sources`, `preview`, `unbridged`, `save`; `unsupported_reason` rejects `fvar` and CFF2 with `FontLoadError` |
| `composites.py` | Qt-free composite discovery and composition over the fontTools glyph set: `find_bridged_composites`, `load_component_outlines`, `compose` |
| `controller.py` | `GuiController`: one `FontProcessor`, one `QThreadPool`, busy guards, per-glyph directions, debounced survey, signals |
| `tasks.py` | `BackgroundTask` (`QRunnable`, `setAutoDelete(False)`) and `TaskSignals` finished/failed/progress |
| `main_window.py` | `MainWindow`: `HeaderBar` over a horizontal `QSplitter` of the sidebar (`ControlPanel`), a `QStackedWidget` (`empty_state` label, then `GlyphGrid`) and the preview pane (`ComparisonView` plus a picker card); save progress lives in the status bar (`progress_label` and `progress_bar`, `set_progress`, `reset_progress`) |
| `direction_picker.py` | `DirectionPicker`: Auto / Vertical / Horizontal combo; for a composite it is disabled and reads "Follows <base>" (direction_picker.py:41) |
| `controls.py`, `glyph_grid.py`, `glyph_view.py`, `outline.py` | Parameters-only control panel (width slider and spin, spanning check, `workers_slider` 0..`os.cpu_count()` with "Auto" at 0); thumbnail `QListWidget` (badge and red label via `UNBRIDGED_ROLE`, direction marker); Original and Stencilized `GlyphCanvas` sharing one union frame; glyph to `QPainterPath` via `QtPen(None, path=path)` |

## Composites in the grid

The domain reader drops components (`fonttools_glyph_to_domain` records outline segments only), so `classify_glyphs` files a composite as an empty glyph and `session.island_glyphs` (= `classification.glyphs_to_process`) never lists one, yet the saved font bridges it through the glyphs it references. `FontSession.display_glyphs` therefore adds every composite that draws an island glyph, found by `find_bridged_composites` over the fontTools glyph set (Roboto: 562 island glyphs + 465 composites = 1027; Lato 447 + 370; CommitMono 467 + 0). A composite is composed from leaf outlines with fontTools `Transform` (child first, then parent), matching `DecomposingRecordingPen` output for all Roboto composites. `direction_sources(name)` returns the island glyphs a name follows: `(name,)` for an island glyph, the referenced islands for a composite, `()` otherwise. A composite has no direction of its own: `preview` runs each source under that source's direction and composes the result. `_component_parts` bounds depth and part count and raises `ValueError` on a component cycle, which `FontSession.open` wraps as `FontLoadError`; without the bound a doubling component DAG hung open.

## Threading model

Open and save run as `BackgroundTask`s on the controller's `QThreadPool`. `_start` connects `TaskSignals` to bound controller methods with `Qt.ConnectionType.QueuedConnection` (controller.py:79-86), so handlers run on the GUI thread. The controller keeps `self._task` until its handler runs, then emits `busy_changed(False)`. `open_font` and `save` refuse a second request while busy (controller.py:120, 183). Previews run synchronously on the GUI thread (one glyph takes at most 8.3 ms). `shutdown()` stops the survey timer and waits on the pool; `MainWindow.closeEvent` refuses while busy.

**Directions.** The controller holds per-glyph choices in `_directions` (`direction_for`, `set_direction`, controller.py:136-158); `AUTO` removes the entry. `set_direction` refuses a composite or non-island name with an `error` signal. Every change refreshes the preview and schedules a survey; `save` passes a copy of the dict through `functools.partial(session.save, ..., directions=dict(self._directions))`.

**Unbridged survey.** `_schedule_survey` bumps `_survey_generation` and restarts a `SURVEY_DELAY_MS = 250` timer (controller.py:27); `_run_survey` runs `session.unbridged(bridge, geometry, directions)` for every displayed glyph as its own `BackgroundTask` kept in `_survey_task`, never touching `_task` or `busy_changed`, so a save is not blocked by it. Only one survey runs at a time; a request during a run sets `_survey_pending`. `_on_survey_finished` emits `unbridged_changed` only when the result's generation equals the current one, then starts the pending survey. `_on_survey_failed` mirrors that: it clears the task, reports only a failure of the current generation, and drains `_survey_pending`. A failure handler must repeat the success path's pending and generation handling.

The save's `ProcessPoolExecutor` must not fork from the multi-threaded Qt process: `app.main` sets `multiprocessing.set_start_method("spawn")` when none is set, and `tests/gui/conftest.py` patches `stencilizer.core.processor.ProcessPoolExecutor` with a spawn context (tests never call `set_start_method`, it is process-global).

## Save safety

`FontSession.save` (session.py:280-315) refuses the input path (resolve/samefile), a missing output folder, and a source whose digest changed since open (`source_digest`, checked again after the write; `_assert_source_unchanged`). `FontWriter` reopens its target by path after the whole glyph run, and `TTFont.save` follows symlinks, so the font is written into a private 0700 `tempfile.TemporaryDirectory` (session.py:299) and published by `_publish`: an `O_CREAT|O_EXCL` sibling of the output written through its descriptor, then rename (session.py:38-51). An output that is a symlink is replaced by a regular file, never written through. `default_log_file` uses `tempfile.mkstemp(prefix="stencilizer-gui-", suffix=".log")` (app.py:40) because `FileHandler` appends through links and a predictable shared-tmp name can be pre-planted. `save(..., directions=None)` forwards the per-glyph directions to `FontProcessor.process`; nothing else in it depends on them.

Accepted residual risks: a non-`OSError` save failure passes `str(error)` to the user; the staging dir follows `TMPDIR`; the digest check cannot stop an input swap-and-revert between its two checks (needs write access to the input).

## Deliberate non-handling

`GuiController._refresh_preview` catches only `StencilizerError`: `process_glyph` already converts transform exceptions into an error dict.

## Theme and styling

`theme.apply_theme(app, scheme=None)` sets the Fusion style, then the palette and stylesheet for LIGHT or DARK (`colors_for`, `palette_for`, `stylesheet_for`). `app.main` passes no scheme, so a `_SchemeFollower` (a `QObject` child of the app) connects `styleHints().colorSchemeChanged` and re-applies the matching theme on a system light/dark switch. On Linux, `portal_theme.PortalTheme` reads the XDG Settings portal with a 250 ms timeout and follows `SettingChanged`; the follower uses that preference whenever Qt reports `Unknown`. An explicit scheme leaves an existing follower connected and installs none. Tokens (LIGHT / DARK): window `#f4f5f7` / `#16181d`, surface `#ffffff` / `#1e2127`, base `#ffffff` / `#121418`, accent `#2563eb` with white `accent_text` in both. `_derived_colors` blends the rest (`border_strong`, hover and pressed fills, selection, focus ring) with `_blend`. Layout modules only set styling hooks (conventions.md "Qt and GUI code"); a contract test pins WCAG contrast: text at least 7.0 on base, surface and window, muted text and accent text at least 4.5.

`GlyphGrid` handles `QEvent.Type.PaletteChange` (glyph_grid.py:137): it re-renders thumbnails and re-colours the unbridged mark, `#c62828` on a light base (lightness 128 or more) and `#ff8a80` on a dark one. Unbridged glyphs also carry a badge (`_badged`, a rounded rect plus dot rather than `drawText`) so the cue does not depend on colour and survives selection.

QSS behaviours found by offscreen renders (Qt 6.11):

- A styled `QSpinBox::up-button` / `QComboBox::drop-down` (any background or border) draws no arrow unless `::up-arrow` / `::down-arrow` has a drawable; Fusion is not a fallback. A border-triangle arrow (width and height 0, coloured top or bottom edge) needs opaque side edges in the field colour, or `qDrawBorder` skips the mitre and draws a flat bar (`QSpinBox::up-arrow`, `QComboBox::down-arrow`).
- Selected `QListWidget` thumbnails get a 30% Highlight wash from the application palette, which widget QSS cannot suppress; theme.py paints the selected cell with `blend(base, accent, 0.3)` so tile and wash match.
- A `QProgressBar` label styled by QSS is drawn in one colour, and no single colour reaches 4.5:1 over both the groove and the accent chunk. The bar hides its text (`setTextVisible(False)`) and a status label beside it shows the percentage.
- `:focus` works on sub-controls (`QSlider::handle:horizontal:focus`, `QCheckBox::indicator:focus`, `QListWidget::item:focus`). A sub-control's width and height exclude its border, so a 2px focus border needs the indicator narrowed (18 to 16) or a 2px resting border on the slider handle, or the layout jumps.
- Focus rings: accent-filled controls (primary button, checked box, slider handle) use the text colour; neutral fills use `_blend(accent, text, 0.3)`. A focused primary button or handle keeps its accent fill on hover and press (press is a 1px content shift): in the light theme no ring colour reaches 3:1 against both the white header and a darkened accent, and lightening the fill drops white text under 4.5:1.
- Once a stylesheet is set, `app.style()` is the `QStyleSheetStyle` proxy and `name()` is `''`: assert on `palette()` and `styleSheet()`, never on the style name.
- QSS properties Qt does not know print "Unknown property" on stderr, which fails the launch check.

## Linux desktop theme detection

Decision: use the XDG Settings portal when Qt reports an unknown colour scheme on Linux, including live SettingChanged notifications. Confirmed on COSMIC: the portal returns uint32 1 (dark), while the native Wayland QApplication reports ColorScheme.Unknown with QT_QPA_PLATFORMTHEME=qt5ct. Keep explicit Qt schemes authoritative and bound portal reads so an unavailable service cannot stall startup.

## PySide D-Bus slot signatures

**What happened:** Passing a bytes slot signature to QDBusConnection.connect raised ValueError during the COSMIC theme check. **Why:** The PySide stub advertises bytes, but the runtime requires the Qt SLOT signature helper. **Prevention:** Use SLOT and verify against a real session bus. **Fix:** Register the portal callback using SLOT rather than a raw bytes signature.

## Constructing D-Bus test replies

**What happened:** The portal read test returned Unknown despite a dark fixture. **Why:** PySide overload selection for createReply with a list wraps the list as a single argument. **Prevention:** Build the reply and set its arguments explicitly. **Fix:** Use createReply followed by setArguments in the fake bus. PySide also incorrectly annotates connect slot signatures as bytes; the SLOT string requires a narrow call-overload suppression.
