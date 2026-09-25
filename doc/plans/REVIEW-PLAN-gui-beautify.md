# Code Review: GUI beautification

**Plan:** PLAN-gui-beautify | **Generated:** 2026-09-25 11:25 UTC

## Summary

## Overview

## Changes by Stage

### Header bar, sidebar cards, workers slider and system-following theme (gui-beautify)

**Status:** completed  
**Purpose:** Restyle the stencilizer GUI: file actions move to a top bar with compact buttons, the left
column becomes a settings sidebar of cards, the worker spin box becomes a slider, the grid
gets an empty state, save progress moves to the status bar, and a light and a dark theme
follow the system setting.
Use parallel subagents and skills to maximize performance.

CONTRACT: doc/plans/briefs/gui-beautify/gui-beautify/_shared.md pins the widget tree, the
styling-hook table, the public surface and the repo rules. Read it before spawning; do not
re-derive it.

PUBLIC SURFACE (the contract session writes tests against exactly this):
  stencilizer.gui.header.HeaderBar(parent: QWidget | None = None), a QFrame with signals
    open_requested, save_requested; attributes title_label, font_name_label,
    font_details_label (QLabel), open_button, save_button (QPushButton); methods
    set_font_info(name: str, details: str), set_font_loaded(loaded: bool),
    set_busy(busy: bool). Both font labels use Qt.TextFormat.PlainText.
  stencilizer.gui.controls.ControlPanel gains workers_slider (QSlider, range
    0..(os.cpu_count() or 1), value 0) and workers_value_label; max_workers() returns None
    at 0, else the slider value. workers_spin is gone.
  stencilizer.gui.main_window.MainWindow(controller) gains header (HeaderBar, above the
    splitter), grid_stack (QStackedWidget), empty_state (QLabel), progress_bar
    (QProgressBar in the status bar), set_progress(completed, total), reset_progress().
    controls, grid, comparison, direction_picker and controller stay.
  stencilizer.gui.glyph_grid.GlyphGrid re-renders its thumbnails (text on base colour of
    its own palette) when its palette changes; THUMBNAIL_SIZE stays 64.
  stencilizer.gui.theme: frozen dataclass ThemeColors(window, surface, base, border, text,
    muted_text, accent, accent_text), every field a "#rrggbb" string; module constants
    LIGHT and DARK; colors_for(scheme: Qt.ColorScheme) -> ThemeColors (DARK for
    Qt.ColorScheme.Dark, LIGHT otherwise); palette_for(colors) -> QPalette with
    Window=window, Base=base, Text=text, Highlight=accent, HighlightedText=accent_text;
    stylesheet_for(colors) -> str; apply_theme(app: QApplication, scheme:
    Qt.ColorScheme | None = None) -> None, which sets Fusion, the palette and the
    stylesheet, and with scheme None follows app.styleHints().colorSchemeChanged.
  stencilizer.gui.app imports apply_theme at module level (from stencilizer.gui.theme
    import apply_theme) and main calls apply_theme(application) right after creating the
    QApplication and before create_window.
CONTRACT NOTES: all contracts live in tests/gui/test_beautify_contracts.py. Import every
new name (theme, header) inside the test function that uses it, so each contract fails on
its own. The file defines its own window fixture (GuiController(tmp_path / "gui.log"),
MainWindow, qtbot.addWidget, controller.shutdown() at teardown) and uses the roboto_path
fixture from tests/gui/conftest.py. Contrast is WCAG 2: channel c/255, linear =
c/12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, L = 0.2126 R + 0.7152 G +
0.0722 B, ratio = (L_hi + 0.05) / (L_lo + 0.05). A test that calls apply_theme restores
QApplication.instance()'s palette and stylesheet afterwards (other tests compare pixels
with the default palette). The offscreen color scheme reads Qt.ColorScheme.Unknown, and
emitting app.styleHints().colorSchemeChanged from Python reaches its connections
(both measured with PySide6 6.11.2).

FOUNDATION (you, before any spawn): uv run pytest --no-cov -q -p no:cacheprovider
tests/gui/test_controls.py tests/regression/test_code_structure.py (green at HEAD; also
creates the worktree .venv the codex units call through .venv/bin/). Read stderr: a blocked
PyPI download is a blocker to report.

WAVES (each starts after the previous wave's files exist):
  wave 1: U1, U2, U3, U4
  wave 2: U5, T1
  wave 3: D1
Territories are DISJOINT. Workers NEVER spawn subagents. Codex rows: spawn every codex unit
of a wave BY AGENT TYPE, ALL in ONE message: loom-codex-forwarder, in the FOREGROUND, with
--model <Tier column> --effort xhigh, --unit-id <worker id>, an explicit Bash timeout of
600000 ms, and the prompt "Your brief: <Brief path>. Read it and
doc/plans/briefs/gui-beautify/gui-beautify/_shared.md in full before anything else. Do not
run git." After EACH codex run, check git status --short yourself and confirm only the
unit's owned files changed. Codex units cannot run uv or record loom memory: their one
check is the brief's static proof through .venv/bin/; you run the real tests and record
their reported assumptions. T1: spawn one loom-software-engineer (sonnet) in the same
message as U5. D1: spawn one loom-senior-software-engineer with the model override fable
(visual design is fable-tier work); its prompt is the fixed prompt minus the git line.

| Worker | Role | Tier | Files owned | Shared context | Brief path |
| ------ | ---- | ---- | ----------- | -------------- | ---------- |
| U1 | HeaderBar and its tests | gpt-5.6-terra | src/stencilizer/gui/header.py, tests/gui/test_header.py | _shared.md (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/u1-header.md |
| U2 | Sidebar cards, workers slider, tests | gpt-5.6-terra | src/stencilizer/gui/controls.py, tests/gui/test_controls.py | _shared.md (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/u2-controls.md |
| U3 | Grid cells, palette-aware thumbnails and marks, tests | gpt-5.6-terra | src/stencilizer/gui/glyph_grid.py, tests/gui/test_glyph_grid.py | gui/outline.py (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/u3-glyph-grid.md |
| U4 | Comparison view cards | gpt-6-luna | src/stencilizer/gui/glyph_view.py | _shared.md (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/u4-glyph-view.md |
| U5 | Main window layout and wiring | gpt-5.6-terra | src/stencilizer/gui/main_window.py | gui/header.py, gui/controls.py (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/u5-main-window.md |
| T1 | Window tests for the new layout | sonnet | tests/gui/test_main_window.py, tests/gui/test_main_window_directions.py, tests/gui/test_main_window_layout.py | all gui modules (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/t1-tests.md |
| D1 | Theme, app wiring, visual pass | fable | src/stencilizer/gui/theme.py, src/stencilizer/gui/app.py, tests/gui/test_theme.py, tests/gui/test_app.py | all gui modules (read-only) | doc/plans/briefs/gui-beautify/gui-beautify/d1-theme.md |

AFTER EACH WAVE (you): uv run ruff format <wave files> && uv run ruff check <wave files>,
then:
  wave 1: uv run mypy src/stencilizer/gui/header.py src/stencilizer/gui/controls.py
          src/stencilizer/gui/glyph_grid.py src/stencilizer/gui/glyph_view.py, and
          uv run pytest --no-cov -q -p no:cacheprovider tests/gui/test_header.py
          tests/gui/test_controls.py tests/gui/test_glyph_grid.py
          tests/gui/test_grid_marks.py tests/gui/test_glyph_view.py
          tests/regression/test_code_structure.py. main_window.py and its tests are
          expected to fail until wave 2.
  wave 2: uv run mypy, then uv run pytest --no-cov -q -p no:cacheprovider tests/gui
          tests/regression/test_code_structure.py. Expected red until wave 3: exactly the
          four theme contracts (theme-colors-are-legible, theme-follows-system-scheme-
          changes, main-applies-theme-before-showing-window, grid-thumbnails-rerender-on-
          palette-change); the three others must pass here.
  wave 3: every acceptance command below. Then look at D1's final PNGs yourself. D1 lists
          any layout change it wants outside its files as file: widget: change: reason;
          apply them yourself when they fit the small-change rule (at most 20 lines in at
          most 2 files), else spawn one codex gpt-5.6-terra unit per file with the list.
Check wc -l on every test file you touched: tests/regression/test_code_structure.py
covers src/ only, and the limit is 400 lines.

INTEGRITY: moving the file actions edits assertion lines in tests/gui/test_controls.py,
tests/gui/test_main_window.py and tests/gui/test_main_window_directions.py. U1, U2 and T1
report every moved assertion as file:old line -> file:new line. After the final review
round, run loom stage review integrity gui-beautify and file ONE loom stage
dispute-integrity gui-beautify with a --event flag per TI-edit event and the moved-assertion
list as the reason. A TI-assert or TI-decl event means an assertion or a test was dropped:
restore it instead of disputing.

FAILURES: a unit exiting 124 timed_out is re-split against the partial tree, never
re-forwarded as is: U1, U2, U3 as module then tests; U5 as _build_panes/_build_status_bar
then the connections and handlers. A unit whose proof or wave tests still fail after one
fix attempt (a fresh codex unit briefed with the failure output) moves one tier up to a
Claude subagent with the brief, the failed diff and the error output:
loom-software-engineer (sonnet) for luna and terra work, loom-senior-software-engineer
(opus) for a Qt event or palette problem in U3. The same worker failing twice gets a
loom-advisor diagnosis before any further attempt. If the codex CLI is unavailable at run
time, the codex rows run on loom-software-engineer.

ERROR HANDLING: no new exception types and no new error paths; the controller's error
signal and QMessageBox.warning stay as they are.

DO NOT TOUCH: everything outside src/stencilizer/gui/ and tests/gui/;
src/stencilizer/gui/session.py, composites.py, controller.py, tasks.py, outline.py,
direction_picker.py; tests/regression/; the frozen tests/gui/test_beautify_contracts.py.

MEMORY: record mistakes, decisions and surprises with loom memory immediately (subagents
report theirs to you; you record them, codex units cannot). NEVER loom knowledge in this
stage; NEVER Claude Code auto-memory. A knowledge file contradicted by the tree gets
loom memory note "stale-knowledge: <file>#<heading> claims X; the tree does Y". Record for
knowledge-distill: the styling-hook convention (objectName for unique widgets, a role
property for classes, theme.py owns every colour and QSS rule), the grid's palette
re-render and dark unbridged colour, the final theme token table from D1. Before
loom stage complete, record loom memory note "wiring: stencilizer.gui modules are imported
by dotted path (from stencilizer.gui.<module> import ...), which loom's unwired-file scan
does not match; header.py is imported by main_window.py, theme.py by app.py; tests/gui is
collected by pytest". Without a memory mentioning wiring, the completion's unwired-file
check fails.


#### Files Changed

- src/stencilizer/gui/main_window.py - initial splitter sizes 260/560/460 -> 300/520/460 so the 'Spanning bridges for stacked islands' checkbox is not clipped (D1 render)

#### Key Decisions

- Contract test_main_applies_theme_before_showing_window also asserts apply_theme gets no explicit scheme (scheme is None), since app.main must follow system scheme changes; test_theme_follows_system_scheme_changes restores the app style name as well as palette and stylesheet, because apply_theme sets Fusion.
- Contract file tests/gui/test_beautify_contracts.py imports theme/header inside test functions; until the stage lands, mypy reports import-not-found for stencilizer.gui.theme/header and attr-defined for MainWindow.header and ControlPanel.workers_slider. No type: ignore added, since those would become unused-ignore errors under mypy --strict once the surface exists.
- D1 (theme, visual pass) spawned on loom-senior-software-engineer with model override fable *(The stage signal assigns D1 to fable: visual/UI design is fable-tier work per Rule 7 point 3; not an escalation after failure)*
- apply_theme with an explicit scheme leaves an existing scheme follower connected and installs none; only scheme=None follows colorSchemeChanged *(the brief only specifies following for scheme None; the render script uses explicit schemes on a fresh app, tests restore palette/stylesheet and never emit after an explicit call)*

#### Notes

- found: the review-harvest hook records rounds at sha256:3c1320e7 (it runs outside the Bash sandbox, where the dotfile mount points do not exist), while loom stage complete run from the sandboxed Bash tool computes sha256:72c10cf8 including them. A new review round cannot close that gap; completion has to run outside the sandbox. Also: round 4 was malformed again although the brief said to end the final text message with the block — the reviewer still ended on its SubagentHandback call.
- gotcha: loom stage complete's review gate failed after a clean round 3 because the Bash sandbox mounts /dev/null over denied dotfiles in the worktree root (.bashrc, .gitconfig, .mcp.json, .idea, .vscode, ...), and those untracked char-device mount points enter the change fingerprint. Prevention: expect them in 'changed since'; they are not stage files and are never committed; a fresh review round at the current fingerprint clears the gate.
- mistake: two loom-code-reviewer rounds recorded malformed. Why: the reviewer delivers its report through a SubagentHandback tool call, then ends with a plain-text 'Review complete and handed back.'; loom hook review-harvest reads only that last text message, so the loom-review block inside the hand-back is never seen. Prevention: brief reviewers that their LAST plain-text assistant message (after any hand-back call) must itself end with the loom-review block
- gotcha: a loom-code-reviewer round was recorded malformed ('no loom-review block') although the reviewer said the block was in its final message; its hand-back summary replaced the final message. Prevention: state the block format and 'nothing follows it' explicitly in every reviewer brief
- gotcha: D1 wanted a placeholder in direction_picker.py source_label for the empty state (picker card is blank before a font loads); not applied because direction_picker.py is on this stage's do-not-touch list — candidate for a later stage
- wiring: stencilizer.gui modules are imported by dotted path (from stencilizer.gui.<module> import ...), which loom's unwired-file scan does not match; header.py is imported by main_window.py, theme.py by app.py; tests/gui is collected by pytest
- decision record: final theme tokens LIGHT window #f4f5f7 surface #ffffff base #ffffff border #dde1e6 text #1c1f24 muted_text #5f6670 accent #2563eb accent_text #ffffff; DARK window #16181d surface #1e2127 base #121418 border #2c3038 text #e6e8eb muted_text #9aa1ab accent #2563eb accent_text #ffffff. Derived in stylesheet_for: border_strong, hover/pressed, accent_hover/pressed, selection via _blend
- found: GlyphGrid re-renders thumbnails and re-colours unbridged marks on QEvent.Type.PaletteChange; unbridged colour is #c62828 on a light base (lightness >= 128) and #ff8a80 on a dark one
- convention: gui styling hooks — objectName for unique widgets (headerBar, appTitle, fontName, fontDetails, sidebar, glyphGrid, emptyState, previewPane, saveProgress), a dynamic 'role' property for classes (card, sectionTitle, hint, value, status, primary, secondary); theme.py owns every colour and QSS rule, layout modules set only hooks
- mistake: asserted QApplication.style().name() == 'fusion' after apply_theme because a probe without a stylesheet printed 'fusion'. Failed because once a stylesheet is set app.style() is the QStyleSheetStyle proxy whose name() is ''. Prevention: never assert on style().name() when a stylesheet is active; assert on palette and styleSheet() instead. Fix: dropped the assertion in tests/gui/test_theme.py

### Integration Verification (integration-verify)

**Status:** completed  
**Purpose:** Final verification of the restyled GUI. Verify FUNCTIONAL INTEGRATION and the visual
result, not only green tests. NEVER Claude Code auto-memory.
CONTEXT: read the plan (doc/plans/), doc/plans/briefs/gui-beautify/gui-beautify/_shared.md,
loom memory show --all, and doc/loom/knowledge/architecture/gui.md.
BUILD & TEST (zero tolerance, fix every warning and failure): the acceptance commands
below. The full suite took 225 s at HEAD against the 300 s cap; record the measured time
with loom memory. If it exceeds 300 s, run tests/gui and the other directories as two
pytest processes, record both results, and dispute the full-suite criterion with those
numbers (loom stage dispute-criteria).
CODE REVIEW: spawn three parallel loom-code-reviewer subagents: (1) Qt behaviour: the
theme follower connects once per application and survives repeated apply_theme calls,
GlyphGrid.changeEvent re-renders only when text or base changed and re-colours marks, no
widget or method orphaned by the move out of ControlPanel, HeaderBar busy/loaded state
matches the old ControlPanel semantics, every label showing font-controlled text is
PlainText, functions within 50 lines and files within 400; (2) visual and accessibility:
give it the PNGs from the FUNCTIONAL step and theme.py; it checks contrast of every token
pair in use, button sizing, alignment and spacing consistency, selection visibility,
thumbnails matching their background in both schemes, and the empty state; (3) test
migration: every assertion removed from tests/gui/test_controls.py,
tests/gui/test_main_window.py and tests/gui/test_main_window_directions.py reappears with
the same strength against HeaderBar or MainWindow, and the new tests exercise the new
behaviour. Fix every finding with an engineer subagent (the reviewers are read-only);
dispute any you judge wrong; never defer one.
SUGGESTIONS: weigh every pending reviewer suggestion the signal lists; resolve each one
implemented with loom memory resolve <id> --outcome implemented --reason <what changed>.
FUNCTIONAL: write a throwaway script in the session scratchpad directory named in your
system prompt (the Read tool cannot open images elsewhere): all code in def main(); the
only top-level code is if __name__ == "__main__": multiprocessing.set_start_method("spawn");
main(). It creates a QApplication, applies the theme with Qt.ColorScheme.Light, builds the
window with app.create_window for tests/fixtures/Roboto-Regular.ttf at 1280x800, waits for
font_loaded and then unbridged_changed, and grabs PNGs of: the loaded window with O
selected; Aacute selected (picker disabled, reading "Follows A"); set_progress(3, 10)
showing; then the same three after apply_theme with Qt.ColorScheme.Dark; and a window
without a font in both schemes. Open every PNG: compact buttons on the right of the top
bar, sidebar cards, even grid cells, legible text, dark thumbnails on a dark background,
red unbridged marks readable in both schemes. Then save the font through
window.save_font into the scratchpad, wait for save_finished, and reload the output:
error_count 0 and O split into 4 contours (as tests/gui/test_main_window.py
_assert_roboto_save checks). Repeat the load and one save for
CommitMono-Cosmix-700-Regular.otf.
If this stage adds files, record the same wiring memory note as gui-beautify before
completing. Record discoveries with loom memory for knowledge-distill, including any
knowledge file contradicted by the tree: loom memory note "stale-knowledge: ...".


#### Files Changed

No changes recorded.

#### Key Decisions

- IV fixes for review round 4: progress text moves out of the bar (setTextVisible False + a status label showing the bar's own text) because QStyleSheetStyle draws a QSS-styled QProgressBar label in ONE colour and no single colour reaches 4.5:1 over both the white groove and the #2563eb chunk; unbridged glyphs get a badge on the thumbnail (colour-independent cue that survives selection) besides the red label; border_strong and focus rings derived to reach 3:1 *(F-4-1..F-4-5 from the visual/accessibility reviewer; DirectionPicker source_label gets PlainText (Qt reviewer finding lost to a malformed round) despite the plan's DirectionPicker non-goal because the PlainText label rule is a repository rule)*
- focus ring on accent-filled controls (primary button, checked box, slider handle) uses the text colour; neutral fills use _blend(accent, text, 0.3) *(light theme: a ring needs luminance <= 0.018 to reach 3:1 on both #2563eb and white, so only near-black works; plain accent ring fails 3:1 on dark hover (2.67) and selection (2.64))*
- unbridged badge '!' drawn as a rounded rect plus dot, not drawText *(thumbnail pixels stay independent of installed fonts under the offscreen platform used by tests)*
- F-6-1 fix: a focused primary button / slider handle keeps its accent fill on hover and press (press shown by a 1px content shift), because in the light theme no ring colour reaches 3:1 against both the white header and an accent darkened toward text, and lightening the fill drops white button text under 4.5:1 *(round 6 finding F-6-1: text ring vs accent_hover 2.66, vs accent_pressed 2.21 (light), 2.68 (dark pressed))*

#### Notes

- blocker (same as gui-beautify): IV work is committed (a5e3166..cc18664) and every check passes (acceptance 8/8, goal-backward, reachable); review rounds 8 and 10 are clean at sha256:042f5acd (computed by the review hook outside the Bash sandbox), while loom stage complete run from the sandboxed Bash tool computes sha256:9736e4e2, which includes the sandbox's /dev/null bind mounts (.bashrc, .gitconfig, .idea, .vscode, ...). Completion must run outside the sandbox: loom stage complete integration-verify from the worktree root
- found: in the IV session the review fingerprint changed from sha256:042f5acd (round 8, clean, recorded before committing) to sha256:9736e4e2 after the four fix commits, with nothing else changed and the dotfile mount points' metadata stable (mtime 2026-09-23, inode 5 across calls); the signal's 'a commit does not change it' did not hold. loom stage complete then failed only on the review gate. Prevention: commit first, then run the final review round, then complete
- suggestions weighed and left for knowledge-distill: move the duplicated window/_load_font fixtures into tests/gui/conftest.py (5 files incl. the frozen contract file, so not in IV); a test for apply_theme's setStyle('Fusion') (style().name() is '' once a stylesheet is set, see the earlier mistake note); tests for header tooltips/cursor (cosmetic); a comment on card borders using border vs border_strong; accent_pressed and focus share one blend expression (distinct roles, kept separate)
- measured on the final IV tree: 366 passed, no warnings, 240.1 s pytest / 241.0 s wall (an earlier run of 364 tests took 265.9 s wall: run-to-run spread is ~25 s against the 300 s cap); ruff/format/mypy clean on 108 files; launch smoke exit 124 with empty stderr
- found/gotcha: under QT_QPA_PLATFORM=offscreen, QWidget.hasFocus() stays False after setFocus() until window.activateWindow() is called and events are pumped; a render script that samples :focus pseudo-state pixels without that step silently renders the non-focused (:hover/:pressed) rule instead and produces misleading contrast/cascade failures. Prevention: always activateWindow()+processEvents() before any focus-state pixel sample in an offscreen Qt render check.
- measured after IV fixes: full suite 364 passed, no warnings, 264.8 s pytest / 265.9 s wall against the 300 s cap (up from 241 s: 15 new tests); the margin is now ~34 s, so the next plan adding GUI tests should expect to split tests/gui or raise the cap
- stale-knowledge: concerns.md#Glyph names rendered as rich text in the direction picker claims DirectionPicker.source_label uses AutoText; the tree now sets Qt.TextFormat.PlainText (src/stencilizer/gui/direction_picker.py:18, test_picker_shows_glyph_names_as_plain_text). Correction: remove the concern
- gotcha: Qt 6.11 QSS honours :focus on sub-controls (QSlider::handle:horizontal:focus, QCheckBox::indicator:focus, QListWidget::item:focus) - confirmed by offscreen render diff. A sub-control's width/height excludes its border (PM_IndicatorWidth/PM_SliderLength add border), so a 2px focus border needs width reduced by 2 (indicator 18->16) or a 2px resting border (slider handle width 10 + 2px accent border) to avoid a layout jump
- mistake: two of three IV loom-code-reviewer rounds (Qt behaviour, visual) recorded malformed or split: the reviewer still delivered via SubagentHandback although the brief said not to call any hand-back tool; round 2 (test migration) recorded fine with the same brief. Why: the hand-back tool is injected by the harness and the model prefers it. Prevention: harvest findings from the hand-back message too and fix them even when the round is malformed; the final re-review brief must repeat the no-hand-back rule
- functional: offscreen render of the real window (create_window, 1280x800) in Light and Dark: Roboto O/Aacute/progress and empty state; Aacute picker disabled reading 'Follows A'; Roboto save 562 processed, 0 errors, reloaded O has 4 contours; CommitMono-Cosmix-700 (CFF) save 467 processed, 0 errors, O 4 contours; stderr empty (no QSS parse warnings)

## Open Questions

No open questions.

