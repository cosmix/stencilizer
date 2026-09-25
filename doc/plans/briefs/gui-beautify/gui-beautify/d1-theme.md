# D1: Theme and visual pass (loom-senior-software-engineer with model fable, wave 3)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first. Waves 1-2 rebuilt the layout
and set every styling hook in its table; the window still renders with Qt's default Fusion look.
You own how it looks: palettes, the stylesheet, and the visual review.

**Owns:** `src/stencilizer/gui/theme.py` (new), `src/stencilizer/gui/app.py`,
`tests/gui/test_theme.py` (new), `tests/gui/test_app.py`.
**Reads:** every module under `src/stencilizer/gui/`, and the frozen contracts in
`tests/gui/test_beautify_contracts.py` (never edit them): `test_theme_colors_are_legible`,
`test_theme_follows_system_scheme_changes`, `test_main_applies_theme_before_showing_window`,
`test_grid_thumbnails_rerender_on_palette_change` fail until your work lands.

## Direction

A calm, modern desktop tool: neutral greys, one blue accent, soft 1px borders, 8-10px corner radii,
generous padding, a clear type hierarchy (bold app title, semibold font name, muted details and
section headings), a primary Save button and a quiet secondary Open button. The glyph thumbnails
and the two preview canvases are the content; the chrome recedes. Starting tokens (tune them; the
contracts check WCAG contrast):

| Token | LIGHT | DARK |
| --- | --- | --- |
| window | `#f4f5f7` | `#16181d` |
| surface | `#ffffff` | `#1e2127` |
| base | `#ffffff` | `#121418` |
| border | `#dde1e6` | `#2c3038` |
| text | `#1c1f24` | `#e6e8eb` |
| muted_text | `#5f6670` | `#9aa1ab` |
| accent | `#2563eb` | `#2563eb` |
| accent_text | `#ffffff` | `#ffffff` |

Contract thresholds (WCAG 2 relative luminance): text on base, surface and window at least 7:1;
muted_text on surface and window at least 4.5:1; accent_text on accent at least 4.5:1. DARK.window
lightness below 128, LIGHT.window 128 or above, and DARK.base differs from LIGHT.base. Keep
DARK.base relative luminance at or below 0.05: the grid's dark unbridged mark `#ff8a80` needs it for
4.5:1.

## Steps

1. `src/stencilizer/gui/theme.py`, the surface in `_shared.md`. `palette_for` maps, for the Active
   and Inactive groups: Window←window, WindowText←text, Base←base, AlternateBase←surface,
   Text←text, Button←surface, ButtonText←text, Highlight←accent, HighlightedText←accent_text,
   PlaceholderText←muted_text, Mid←border, ToolTipBase←surface, ToolTipText←text, Link←accent;
   the Disabled group's Text, ButtonText and WindowText get muted_text. `GlyphCanvas` paints with
   Base, Text and Mid, and `GlyphGrid` rasterizes thumbnails with Text on Base, so these roles are
   the canvas and thumbnail colours. `stylesheet_for` returns QSS built from the tokens, with a rule
   for every hook in `_shared.md`'s table plus `QSplitter::handle`, `QStatusBar`, `QSlider` (groove,
   sub-page, handle), `QCheckBox::indicator`, `QSpinBox`, `QComboBox`, `QToolTip`,
   `QScrollBar:vertical`, and `QListWidget#glyphGrid::item` (`:hover`, `:selected`). Give
   `#glyphGrid` the same background as `base`: a QSS background changes the widget palette, and the
   grid re-renders its thumbnails from it. Use only properties Qt's stylesheet engine knows: an
   unknown one prints `Unknown property <name>` and fails the stage's launch check
   (`letter-spacing`, `text-transform`, `box-shadow`, `transition` and `opacity` on widgets are
   unknown). `apply_theme(app, scheme=None)`: `app.setStyle("Fusion")`, then apply `palette_for` and
   `stylesheet_for` for `colors_for(scheme)`, or for `colors_for(app.styleHints().colorScheme())`
   when `scheme` is None; with `scheme` None it also follows `app.styleHints().colorSchemeChanged`,
   connected once per application however often `apply_theme` runs (a `QObject` follower parented
   to the app and found again with `app.findChild`, or an equivalent guard).
2. `src/stencilizer/gui/app.py`: `from stencilizer.gui.theme import apply_theme` and, directly
   after `application = QApplication(sys.argv[:1])`, the line `apply_theme(application)`.
   `tests/gui/test_app.py`: in `_stub_startup` add a documented no-op
   `def skip_theme(_application: object) -> None:` and `monkeypatch.setattr(app, "apply_theme", skip_theme)`,
   so the stubbed startups keep `events == ["application", "show"]`. Change nothing else there.
   `tests/gui/test_theme.py`: a fixture that saves `QApplication.instance()`'s palette and
   stylesheet and restores both afterwards (other tests compare pixels with the default palette);
   tests for `colors_for` (Unknown and Light give LIGHT, Dark gives DARK), the `palette_for` role
   map for both themes, `stylesheet_for` naming every hook selector of `_shared.md`, and
   `apply_theme` called twice then one emitted `colorSchemeChanged(Qt.ColorScheme.Dark)` applying
   DARK once.
3. Visual pass. Write a throwaway script in the session scratchpad directory named in your system
   prompt (the Read tool cannot open images elsewhere): all code in `def main()`, the only top-level
   code `if __name__ == "__main__": multiprocessing.set_start_method("spawn"); main()`. It creates a
   `QApplication`, calls `apply_theme(app, Qt.ColorScheme.Light)`, builds the window with
   `app.create_window(Path("tests/fixtures/Roboto-Regular.ttf"), <scratchpad>/gui.log)`, resizes it
   to 1280x800, shows it, waits for `controller.font_loaded` with a `QEventLoop` and a 30 s
   `QTimer` guard, selects `O`, calls `window.set_progress(3, 10)`, and saves `window.grab()` as PNG;
   then the same after `apply_theme(app, Qt.ColorScheme.Dark)`; then a second window without a font
   (empty state) in both schemes. Run it with `QT_QPA_PLATFORM=offscreen uv run python <script>`,
   open every PNG, and refine `theme.py` until the window reads as one designed product in both
   schemes: aligned edges, even spacing, compact buttons, legible text, thumbnails that match their
   background, a visible selection. At most three render rounds. If a fix needs a layout change
   outside your files (a margin, a spacing, a widget order), do not make it: list it in your report
   as `file: widget: change: reason`, and the orchestrator applies it.

Renders are your design loop. Your one verification check, run once at the end:
`uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_theme.py tests/gui/test_app.py tests/gui/test_beautify_contracts.py tests/regression/test_code_structure.py`.
Report its result, the final token table, and the paths of the final PNGs.
