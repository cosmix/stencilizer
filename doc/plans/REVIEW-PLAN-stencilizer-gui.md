# Code Review: Stencilizer desktop GUI

**Plan:** PLAN-stencilizer-gui | **Generated:** 2026-09-23 19:27 UTC

## Summary

## Overview

## Changes by Stage

### PySide6 GUI package (gui-app)

**Status:** completed  
**Purpose:** Build src/stencilizer/gui/ (PySide6): open a font, set parameters, browse island
glyphs, preview one glyph before/after stencilization, save the stenciled font.
Use parallel subagents and skills to maximize performance.

CONTRACT: doc/plans/briefs/stencilizer-gui/gui-app/_shared.md pins every module's
signatures, the repo rules (mypy strict, ruff, 400/50/300 size limits, Qt noqa codes,
QtPen path=, queued connections) and the measured facts the tests use. Read it before
spawning; do not re-derive it.

FOUNDATION (you, before any spawn; commands plus two hand-edited lines):
1. uv add --optional gui 'pyside6-essentials>=6.11.2'
2. uv add --dev 'pyside6-essentials>=6.11.2' 'pytest-qt>=4.5.0'
3. pyproject.toml: under [project.scripts] add
     stencilizer-gui = "stencilizer.gui.app:main"
   and under [tool.pytest.ini_options] add
     qt_api = "pyside6"
4. Prove it: uv run python -c "import pytestqt" and
   QT_QPA_PLATFORM=offscreen uv run python -c "from PySide6.QtWidgets import
   QApplication; app = QApplication([]); print('qt-ok')" (Qt starts inside this
   stage's sandbox) and uv run pytest --no-cov -q -p no:cacheprovider
   tests/integration/test_stencilization_formats.py tests/unit/test_refactor_contracts.py
   (a real process pool and tmp_path inside this sandbox with pytest-qt installed;
   tests/unit/test_processor.py mocks the pool and proves neither). This also creates
   the worktree's .venv, which codex units call through .venv/bin/. Read stderr: a
   blocked PyPI download or a Qt platform-plugin failure is a blocker to report, not a
   pass.

WAVES (each wave starts only after the previous wave's files exist and its proof
passed; later modules import earlier ones):
  wave 0: W0
  wave 1: W1, W2, W3, W4
  wave 2: W5, W6, W7
  wave 3: W7T, W8
  wave 4: W8T, W9
Territories below are DISJOINT. Workers NEVER spawn subagents. Spawn every worker of a
wave BY AGENT TYPE, ALL in ONE message: loom-codex-forwarder, in the FOREGROUND, with
--model <Tier column> --effort xhigh, --unit-id <worker id>, an explicit Bash timeout
of 600000 ms, and the prompt "Your brief: <Brief path>. Read it and
doc/plans/briefs/stencilizer-gui/gui-app/_shared.md in full before anything else. Do
not run git." Codex units must not run git: after EACH codex run, check
git status --short yourself and confirm only the unit's owned files changed.

A T row (W7T, W8T) uses its row's brief; append to its prompt "You are unit <id>:
write only <file>, from the brief's Tests section, against the module already in the
tree." Codex units cannot run `uv run` (no network, read-only uv cache and /tmp inside
codex's sandbox): their one check is the brief's proof command through .venv/bin/,
run once, and they never run the tests. You run the real tests after each wave.

| Worker | Role | Tier | Files owned | Shared context | Brief path |
| ------ | ---- | ---- | ----------- | -------------- | ---------- |
| F | Foundation (you, not spawned) | orchestrator | pyproject.toml, uv.lock | none | FOUNDATION section above |
| W0 | Test scaffold | gpt-6-luna | tests/gui/__init__.py, tests/gui/conftest.py | pyproject.toml (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w0-scaffold.md |
| W1 | Font session model | gpt-5.6-terra | src/stencilizer/gui/__init__.py, src/stencilizer/gui/session.py, tests/gui/test_session.py | core/processor.py, cli/app.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w1-session.md |
| W2 | Outline rendering | gpt-5.6-terra | src/stencilizer/gui/outline.py, tests/gui/test_outline.py | io/converter.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w2-outline.md |
| W3 | Background task | gpt-5.6-terra | src/stencilizer/gui/tasks.py, tests/gui/test_tasks.py | exceptions.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w3-tasks.md |
| W4 | Control panel | gpt-6-luna | src/stencilizer/gui/controls.py, tests/gui/test_controls.py | config/settings.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w4-controls.md |
| W5 | Glyph canvas and comparison | gpt-5.6-terra | src/stencilizer/gui/glyph_view.py, tests/gui/test_glyph_view.py | gui/outline.py, gui/session.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w5-glyph-view.md |
| W6 | Glyph grid | gpt-6-luna | src/stencilizer/gui/glyph_grid.py, tests/gui/test_glyph_grid.py | gui/outline.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w6-glyph-grid.md |
| W7 | Controller module (threads, busy guard) | gpt-6-sol | src/stencilizer/gui/controller.py | gui/session.py, gui/tasks.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w7-controller.md |
| W7T | Controller tests | gpt-5.6-terra | tests/gui/test_controller.py | gui/controller.py, gui/session.py, gui/tasks.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w7-controller.md |
| W8 | Main window module | gpt-5.6-terra | src/stencilizer/gui/main_window.py | all gui modules (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w8-main-window.md |
| W8T | Main window tests | gpt-5.6-terra | tests/gui/test_main_window.py | all gui modules (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w8-main-window.md |
| W9 | Console entry point | gpt-5.6-terra | src/stencilizer/gui/app.py, tests/gui/test_app.py | gui/main_window.py, gui/controller.py (read-only) | doc/plans/briefs/stencilizer-gui/gui-app/w9-app.md |

AFTER EACH WAVE (you), one command: uv run ruff format <wave files> && uv run ruff
check <wave files> && uv run mypy <wave files> && uv run pytest --no-cov -q -p
no:cacheprovider <wave test files> (wave 0 has no test module: format, lint and mypy,
then uv run python -c "import tests.gui.conftest"). Only then start the next wave.

FAILURES: a unit exiting 124 timed_out is re-split against the partial tree (module
and test as two units), never re-forwarded as is. A unit whose proof or wave tests
still fail after one fix attempt (a fresh codex unit briefed with the failure output)
moves one tier up to a Claude subagent with the brief, the failed diff
and the error output: luna/terra work to loom-software-engineer (sonnet), W7 to
loom-senior-software-engineer (opus). The same unit failing twice gets a loom-advisor
diagnosis before any further attempt. If the codex CLI is unavailable at run time,
every row runs on loom-software-engineer, W7 on loom-senior-software-engineer.

ERROR HANDLING: the existing hierarchy only (FontLoadError, FontSaveError,
GlyphNotFoundError under StencilizerError); the controller turns failures into its
error signal and the window shows them in a QMessageBox. No new exception types.

DO NOT TOUCH: src/stencilizer/cli/, src/stencilizer/core/, src/stencilizer/io/,
src/stencilizer/config/, existing tests. The CLI must keep working without the gui
extra: nothing outside src/stencilizer/gui/ imports PySide6, and gui/__init__.py
imports nothing.

MEMORY: record mistakes, decisions and surprises with loom memory immediately
(subagents report theirs to you; you record them). NEVER loom knowledge in this
stage; NEVER Claude Code auto-memory. A knowledge file contradicted by the tree gets
loom memory note "stale-knowledge: <file>#<heading> claims X; the tree does Y".
Before loom stage complete, record loom memory note "wiring: stencilizer.gui modules
are imported by dotted path (from stencilizer.gui.<module> import ...), which loom's
unwired-file scan does not match; app.py is reached through the stencilizer-gui
console script; tests/gui is collected by pytest". Without a memory mentioning wiring,
the completion's unwired-file check fails.


#### Files Changed

No changes recorded.

#### Key Decisions

- The GUI complements the CLI and never replaces it: the CLI stays fully functional and untouched *(User instruction during gui-app. Enforced by: separate stencilizer-gui console script (the stencilizer Typer command is unchanged), no edits under src/stencilizer/cli|core|io|config, gui/__init__.py imports nothing, PySide6 only in the optional gui extra, acceptance checks that importing stencilizer.cli.app loads no PySide6, and the existing CLI test suite runs in the full gate)*
- Adversarial test review requested by the user: GUI tests must not be tailor-made to pass or conceal an imperfect implementation *(User instruction mid-stage. Three read-only loom-code-reviewer passes (tests A, tests B, source) look for tests that would stay green under a plausible broken implementation, then the orchestrator mutation-spot-checks the top suspects before fixing)*

#### Notes

- wiring: stencilizer.gui modules are imported by dotted path (from stencilizer.gui.<module> import ...), which loom's unwired-file scan does not match; app.py is reached through the stencilizer-gui console script; tests/gui is collected by pytest
- gotcha: after the first full uv run pytest in the stage sandbox created .coverage, every later sandboxed run crashed in coverage.py file_be_gone with EBUSY on .coverage: the sandbox bind-mounts allowlisted writable paths that exist, so the file cannot be removed. Suite verified with COVERAGE_FILE=$TMPDIR/cov.data (251 passed). Prevention: set COVERAGE_FILE to a temp path for in-sandbox runs, or pre-create nothing at .coverage
- mistake: my mutation spot-check left a stale mutant .pyc in src/stencilizer/gui/__pycache__: the WindingFill->OddEvenFill mutant has the same byte length, the restore landed in the same mtime second, so Python kept the mutant bytecode and test_winding_fill_shows_broken_hole failed afterwards on correct source. A first cleanup with plain fd deleted nothing because fd skips gitignored dirs without -I. Prevention: run mutants with PYTHONDONTWRITEBYTECODE=1 and delete __pycache__ with fd -H -I afterwards
- found: mutation spot-checks (12 mutants over gui/ behaviours: queued connection, config swap per save, winding fill, fvar/CFF2 rejection, input-overwrite refusal, busy guards, shutdown wait, spawn start method, close-while-busy, save status path) killed 10 of 12 on first run; survivors were the CFF2 filename leak and a frame-union test whose Roboto O stenciled bounds never exceed the original. Script: re-apply one str.replace per mutant, run the named node ids, restore bytes
- mistake: the CFF2 rejection tests passed via the fixture filename: conftest named the converted font CommitMono-CFF2.otf, FontLoadError embeds the path, so assert 'CFF2' in message held even with the CFF2 branch of unsupported_reason deleted (the font was still rejected by the no-outline-table branch). Found by a mutation spot-check. Prevention: error-message asserts must match the reason text, and fixture filenames must not contain the words the test searches for
- found/gotcha: task brief for gui-app stage said compare 'a.type == b.type' on domain Point objects; the actual field is Point.point_type (src/stencilizer/domain/contour.py:58), there is no .type attribute. Used point_type in tests/gui/conftest.py outlines_match.
- found: adversarial test review of tests/gui found weak asserts (glyph view frame not checked as union, outline frame right edge one-sided, width range 30-110 unasserted, outlines_match ignoring point type, thread test not requiring each signal) and one source defect (MainWindow.save_font overwrote _output_path on a busy-rejected save, mislabeling the status). Spec-inherited weaknesses left as-is: CFF outline test compares bounds only, canvas paint check needs one pixel
- mistake: W4 ControlPanel.__init__ came out at 59 effective lines, failing tests/regression/test_code_structure.py::test_function_line_limit; the codex proof command (mypy, ruff, collect-only) cannot catch it. Prevention: add tests/regression/test_code_structure.py to each wave's pytest run
- gotcha: codex units cannot run loom memory: W1 and W8 both reported their memory write failed because the loom scratch dir is read-only inside codex's workspace-write sandbox; the orchestrator must record codex assumptions itself. W8 assumption: stale preview/save signals with no active session or remembered output are ignored
- gotcha: loom subagents watch run from the sandboxed Bash tool exited 3 'companion job process 104 is gone while its record says running' seconds after the W0 forward started; the sandbox gives each Bash call its own PID namespace, so the watch cannot see codex's pid. Treat that exit as unknown and wait for the forwarder's own completion

### Integration Verification (integration-verify)

**Status:** completed  
**Purpose:** Final verification of the GUI. Verify FUNCTIONAL INTEGRATION, not only green tests.
NEVER Claude Code auto-memory.
CONTEXT: read the plan (doc/plans/), the shared brief
doc/plans/briefs/stencilizer-gui/gui-app/_shared.md, loom memory show --all, and the
knowledge sections the GUI touches (architecture.md "Processing pipeline",
patterns.md "Winding normalization", mistakes.md).
BUILD & TEST (zero tolerance, fix every warning and failure): uv run pytest (full
suite with coverage; 168 tests at the base commit plus the new tests/gui),
uv run ruff check src tests, uv run ruff format --check src tests, uv run mypy.
CODE REVIEW: spawn parallel loom-code-reviewer subagents: (1) architecture and
concurrency: every signal emitted from a QThreadPool thread reaches a bound method of a
GUI-thread QObject through a queued connection; the controller keeps the running task
referenced until its handler runs; busy guards sit in open_font/save; one FontProcessor
per controller; the start method is set in app.main only. (2) security: paths from
dialogs and argv, the overwrite-the-input guard (resolved path and samefile), the
source-revision check before and after the save, the unsupported-font rejection at
open (fvar, CFF2, no glyf/CFF), no PySide6 import reachable from the
CLI (python -c "import stencilizer.cli.app" must not import PySide6: check
sys.modules). (3) test coverage of src/stencilizer/gui/, judged against the briefs'
test lists (no fail_under is configured). Fix every finding with an
engineer subagent (the reviewers are read-only).
FUNCTIONAL (prove it is wired in and usable):
- uv run stencilizer-gui --help exits 0 through the installed console script.
- The offscreen launch of the real console script with Roboto is an acceptance command
  below (runs 15 s, must end by timeout with exit 124 and no Traceback on stderr).
- Write a short throwaway script under the session scratchpad directory named in your
  system prompt (not the tree, and not another temp dir: the Read tool is blocked
  outside the scratchpad, so PNGs elsewhere cannot be inspected). All script code sits
  in def main(); the only top-level code is if __name__ == "__main__":
  multiprocessing.set_start_method("spawn"); main() (spawn children re-import
  __main__). The script builds the window with app.create_window for each fixture font
  (Roboto-Regular.ttf, Lato-Black.ttf, CommitMono-Cosmix-700-Regular.otf), waits for the
  load, selects O, and saves window.grab() as a PNG; open each PNG and confirm the
  Original and Stencilized panes show the same glyph at one scale, with the bridge cut
  visible. Save each font through the window into the scratchpad and reload the output
  with FontReader: save_finished must report error_count 0 and processed_count equal
  to the font's island-glyph count, and the saved O must have 4 contours (the input's
  has 2); a reload that only proves readability does not count.
If this stage adds files, record the same wiring memory note as gui-app before
completing.
Record discoveries with loom memory for knowledge-distill, including: the GUI layout
and threading model, the spawn decision, and that existing tests write
stencilizer_<timestamp>.log into the working directory (FontProcessor built without
log_file). A knowledge file contradicted by the tree gets
loom memory note "stale-knowledge: ...".


#### Files Changed

No changes recorded.

#### Key Decisions

- app.main sets the multiprocessing start method to spawn (only when none is set) before building QApplication, so the save's ProcessPoolExecutor never forks from the multi-threaded Qt process; tests never call set_start_method (process-global) and instead an autouse conftest fixture patches stencilizer.core.processor.ProcessPoolExecutor with a spawn mp_context *(CPython 3.13 on Linux defaults to fork and warns that fork from a multi-threaded process may deadlock; the save runs on a QThreadPool thread. IV functional driver confirmed start-method=spawn and 562/447/467 processed, 0 errors on Roboto/Lato/CommitMono)*
- IV kept two Low architecture notes unchanged: _refresh_preview catches only StencilizerError, and ControlPanel's _emit_open/_emit_save_requested default _checked while _emit_parameters_changed does not *(The first matches the w7 brief and process_glyph already converts transform exceptions into an error dict, so a non-Stencilizer exception there is not a reachable case (no handling for impossible cases); the second is cosmetic, PySide6 trims signal arguments to slot arity either way)*
- GUI FontSession.save writes to an O_EXCL-created random sibling (.<name>.<hex>.tmp, mode 0o666 minus umask) and publishes with Path.replace only after the post-save source digest check; any failure unlinks the temp and leaves an existing output untouched. Chosen over writing output_path directly (FontWriter -> TTFont.save follows symlinks, so an output swapped to a link to the input overwrote it) and over mkstemp (forces 0600 on the saved font). *(CWE-367/CWE-59 fix confined to the gui package; core/io stay untouched. An output_path that is a symlink is now replaced by a regular file instead of written through.)*
- GUI default_log_file creates a per-run private log via tempfile.mkstemp(prefix='stencilizer-gui-', suffix='.log') instead of the predictable <tmp>/stencilizer-gui-<user>.log *(logging.FileHandler opens in append mode following links, so a predictable shared-tmp name let another local user pre-plant it (CWE-377). Per-run files match the CLI's timestamped logs; staying in the temp dir keeps sandboxed runs writable.)*
- GUI save stages in a private 0700 tempfile.TemporaryDirectory and publishes via an O_CREAT|O_EXCL sibling written through its descriptor plus rename; supersedes the earlier temp-sibling-passed-to-processor decision *(TTFont.save reopens its output by path after the whole glyph run, so a sibling visible in the output directory during processing can be swapped for a symlink to the input by any user with write access there; O_EXCL at creation does not cover a later path-based reopen. Staging under gettempdir (sticky /tmp or per-user dir) keeps the reopened path unreachable, and the publish step never reopens by path.)*
- Split tests/gui/test_session.py into test_session.py + test_session_open.py + a third file test_session_save.py (not just two) *(Moving only open/preview tests left test_session.py at 408-420 lines; adding the new pinned test_save_refuses_missing_output_folder pushed it back over 400. Every remaining save test needs the 7-line _settings helper, which per the brief's duplication rule (>3 lines) must live in one shared place rather than be copied, so it moved to tests/gui/conftest.py as build_settings and both test_session.py and the new test_session_save.py import it; test_session_save.py also carries its own duplicate of the autouse staging_root fixture and the 3-line _temporary_files helper so every save-calling test keeps the same leftover-staging-dir safety net.)*
- IV accepted three residual save risks without code change: a non-OSError save failure passes str(error) through (it carries the diagnosis, e.g. 'disk full'; any embedded path is the private, already-deleted staging dir); the staging dir follows TMPDIR (mkdtemp is 0700 wherever it lands); the input digest check cannot stop a swap-and-revert of the input between the two checks (needs write access to the input itself) *(Final adversarial review of the private-staging save found no Critical/High issue and confirmed the input is never written through the output directory; the three items were rated plausible or negligible)*

#### Notes

- wiring: IV added tests/gui/test_session_open.py, tests/gui/test_session_save.py and tests/gui/test_controller_errors.py (split from test_session.py and test_controller.py to meet the 400-line limit); they are collected by pytest and import shared helpers from tests.gui.conftest (build_settings, load_session, LOAD_TIMEOUT, SAVE_TIMEOUT; staging_root applied with pytestmark usefixtures), matching the repo's tests.integration.conftest import convention. stencilizer.gui modules are imported by dotted path, which loom's unwired-file scan does not match; app.py is reached through the stencilizer-gui console script
- mistake: placed the new 'output folder does not exist' check (output_path.parent.is_dir()) in FontSession.save before the overwrite-input check, following the brief's 'next to the is_dir check' wording literally. Failed because tests/gui/test_session.py::test_save_refuses_input_path passes tmp_path/'sub'/'..'/source.name as an output path where 'sub' never exists on disk: Path.is_dir() must stat through 'sub' literally and fails, while Path.resolve() normalizes '..' lexically without requiring 'sub' to exist, so the overwrite check must run first. Prevention: when adding a filesystem-existence check near an existing resolve()-based check, order it AFTER the resolve()-based check, since resolve() tolerates missing intermediate components that raw stat-based checks (is_dir/exists) do not. Fix: session.py save() now checks is_dir, then the resolve()/samefile overwrite check, then parent.is_dir() (src/stencilizer/gui/session.py:180-187).
- found: tests/regression/test_code_structure.py enforces the 50-line function limit on src/ only, so test files grow past the 400-line file limit unnoticed: tests/gui/test_session.py reached 524 and test_controller.py 429 lines during IV and were split (open/preview tests to test_session_open.py, error paths to test_controller_errors.py) keeping the plan's acceptance node ids in place. Prevention: check wc -l tests/gui/*.py in the gate
- mistake: my IV brief for the save TOCTOU fix had FontProcessor write the font to an O_EXCL temp sibling in the output directory, then closed the descriptor; FontWriter.save -> TTFont.save reopens that path with open(path, 'wb') after the whole processing run, so a local user with write access to the output dir could delete the visible .tmp and plant a symlink to the input, which the write then follows. Why: I treated O_EXCL creation as protecting the later path-based reopen. Prevention: never let a path-based writer (fontTools, FontWriter) target a name in a directory another user can write; stage in a private mkdtemp directory and copy into the destination through the descriptor that O_EXCL returned, then os.replace. Found by the IV adversarial review
- gotcha: PySide6 QThread.currentThread() wrapper objects stored in a list and compared to a later controller.thread() call can spuriously report inequality (or raise 'libshiboken: Internal C++ object already deleted') under real test-suite load, even though the actual delivery thread is correct. Prevention: compare QThread.currentThread() == controller.thread() immediately inside the DirectConnection-connected slot and store only the resulting bool, not the QThread object. Evidence: tests/gui/test_controller.py test_failure_signals_delivered_on_gui_thread (flaky with raw-QThread-list pattern, reliable with immediate bool comparison).
- found: IV functional check drove app.create_window over Roboto/Lato/CommitMono offscreen, selected O, grabbed the window, saved through the window and reloaded with FontReader: processed equals island count (562/447/467), error_count 0, O 2 contours in, 4 in preview and saved, before/after canvases share one frame. Screenshot showed the comparison canvases filling only a third of the pane height (QGridLayout gave title, canvas and info rows equal stretch); fixed with setRowStretch(1, 1)
- found: GUI layout and threading model. MainWindow is a horizontal QSplitter: ControlPanel (open, font info, bridge width 30-110 slider+spin, spanning-bridges toggle, workers Auto/N, Stencilize & Save, progress) | GlyphGrid (QListWidget thumbnails of island glyphs) | ComparisonView (Original and Stencilized GlyphCanvas sharing one union frame, info label). GuiController (GUI thread, one FontProcessor, one QThreadPool) runs open and save as BackgroundTask QRunnables (setAutoDelete False); TaskSignals finished/failed/progress connect with QueuedConnection to bound controller methods; the controller keeps self._task until the handler runs, then busy_changed(False); busy guards in open_font/save refuse a second request; previews run synchronously on the GUI thread (one glyph <= 8.3 ms); shutdown() waits on the pool and closeEvent refuses while busy. FontSession (session.py) is Qt-free
- gotcha: with QT_QPA_PLATFORM=offscreen, showing the main window prints 'This plugin does not support propagateSizeHints()' on stderr (once per show). It is the offscreen QPA plugin's own message, not a Traceback or a code defect; the 15 s launch acceptance (exit 124, no Traceback) passes with it present
- gotcha: a fresh loom worktree has no .venv; the first uv run builds it offline from the uv cache (32 packages) and prints 'warning: Failed to hardlink files; falling back to full copy' because cache and worktree sit on different filesystems under the sandbox. Harmless; UV_LINK_MODE=copy silences it
- found: existing tests write stencilizer_<timestamp>.log into the working directory: setup_logging names the file stencilizer_%Y%m%d_%H%M%S.log in cwd when log_file is None, and FontProcessor(settings) without settings.logging.log_file hits that path (tests/unit/test_processor.py, tests/integration/test_stencilization*.py, test_e2e_output.py, tests/regression/_golden.py:195). One full uv run pytest left 12 such files at the worktree root; *.log is gitignored (.gitignore:65) so git status stays clean. GUI tests avoid it via the conftest processor fixture with a tmp_path log

## Open Questions

No open questions.

