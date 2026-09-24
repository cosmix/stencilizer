# Shared contract: per-glyph bridge direction and complete glyph grid (stage bridge-direction)

Every worker reads this file in full before its own brief. The signatures below are binding: a
worker implements its own module exactly as written and imports the others exactly as written.
Where a body is described in prose, the prose is the specification.

## What the stage delivers

1. A per-glyph bridge direction (Auto / Vertical / Horizontal) chosen in the GUI, used by the
   preview and by the saved font.
2. The glyph grid lists every glyph that comes out bridged, including composite glyphs (Aacute,
   Aring, eacute, ...) that draw a bridged island glyph through a component reference.
3. Glyphs where no bridge can be placed are reported truthfully (0 islands bridged) and marked in
   the grid.

## Rules for every worker

- Never run `git`. Write only the files your brief lists under "Files owned".
- Modules written by earlier waves exist in the worktree but NOT in the source graph: `loom map`
  answers from the base commit and will not show them. Read them with `cat`. The FOUNDATION edits
  (below) are in the worktree before any worker starts.
- Python >= 3.11, `X | None` unions. mypy runs in strict mode and ruff with the repo config.
  Inside codex's sandbox `uv run` fails (no network, read-only uv cache and `/tmp`): call the
  tools through `.venv/bin/` (`.venv/bin/ruff check --fix <files>`, `.venv/bin/mypy <files>`).
  The orchestrator formats (`ruff format`) and runs the tests after each wave.
- `tests/regression/test_code_structure.py` enforces on everything under `src/`: file <= 400
  lines, function <= 50 lines (docstring excluded), class <= 300 lines, no `x._font` access
  outside `io/reader.py` (use the public `FontReader.font`), and no axis-mirrored copies of
  bridge code. Test files are not checked by it: keep every test file under 400 lines yourself.
- `tests/regression/` and `tests/unit/test_refactor_contracts.py` are frozen: never edit them.
  `tests/regression/test_behavior_golden.py` pins the default (Auto) output bit for bit, so Auto
  must behave exactly as the code does today.
- Qt override methods need `# noqa: N802` (plus `ARG002` when the event is unused). No `@Slot`
  decorators. Signals emitted from a `QThreadPool` thread connect ONLY to bound methods of a
  GUI-thread QObject with `Qt.ConnectionType.QueuedConnection`, never to a lambda.
- fontTools ships no type hints: import with `# type: ignore[import-untyped]` as
  `src/stencilizer/io/converter.py` does, and wrap values read from it in `int()`/`float()`
  before returning them (mypy `warn_return_any`).
- `src/stencilizer/gui/session.py` and `src/stencilizer/gui/composites.py` stay Qt-free: they
  never import PySide6 (an acceptance command imports `stencilizer.gui.session` and fails if
  PySide6 lands in `sys.modules`).
- Every module, public class and public function gets a docstring (the style of `src/`).
- Errors use the existing hierarchy in `src/stencilizer/exceptions.py` (`StencilizerError`,
  `FontLoadError`, `FontSaveError`, `GlyphNotFoundError`). Add no new exception types.
- Import `BridgeDirection` from `stencilizer.config.settings` (it is not re-exported from
  `stencilizer.config`).

## FOUNDATION (written by the orchestrator before wave 1)

`src/stencilizer/config/settings.py`:

```python
from enum import StrEnum


class BridgeDirection(StrEnum):
    """Which way a glyph's bridges cut its strokes."""

    AUTO = "auto"  # the analyzer's choice, as before this change
    VERTICAL = "vertical"  # bridge line at a fixed x: an O loses its top and bottom strokes
    HORIZONTAL = "horizontal"  # bridge line at a fixed y: an O loses its left and right strokes


class BridgeConfig(BaseModel):
    # existing fields unchanged, then:
    direction: BridgeDirection = Field(
        default=BridgeDirection.AUTO,
        description="Bridge direction for every island of the glyph (auto keeps the analyzer's choice)",
    )
```

`src/stencilizer/core/surgery_context.py`: `SurgeryContext` gains the field
`direction: BridgeDirection = BridgeDirection.AUTO`, declared directly after `use_spanning`.
`SurgeryContext.merge` applies it only when the caller forced nothing:

```python
if not force_horizontal and not force_vertical:
    force_horizontal = self.direction is BridgeDirection.HORIZONTAL
    force_vertical = self.direction is BridgeDirection.VERTICAL
```

`ContourMerger.merge_contours_with_bridges` (`core/merger.py`) already honours the two flags:
`MergeDispatch.forced_horizontal` / `forced_vertical` (`core/merger_dispatch.py`) build the
requested orientation and fall back to the other one when it cannot be built. With AUTO both
flags stay False and `MergeDispatch.preferred` runs, exactly as today.

## Direction semantics (settled; implement exactly this)

| Where | Auto | Explicit direction D |
| --- | --- | --- |
| Single island (`arrangement` returns "single") | `preferred()` | `SurgeryContext.merge` forces D, falls back to the other axis |
| Island group whose `arrangement` equals D | spanning only if `ctx.use_spanning`, else sequential | spanning ALWAYS tried (ignores the spanning checkbox); if it fails, sequential (which cuts the other way) |
| Island group whose `arrangement` differs from D | spanning if `ctx.use_spanning`, else sequential | sequential only (it already cuts along D) |
| Nested children and inverted islands (`_merge_child`, `_process_inverted` in `core/surgery_nested.py`) | `preferred()` | `SurgeryContext.merge` forces D |
| `_split_child` (continues an existing vertical gap of the parent through a nested outer) | unchanged | unchanged |

`arrangement` strings are "vertical" (islands stacked, B, 8), "horizontal" (side by side),
"single". `_sequential` forces the axis perpendicular to the arrangement; spanning cuts along it.
Consequences the tests rely on: B with HORIZONTAL produces exactly the Auto output with
`use_spanning_bridges=False`; B with VERTICAL and `use_spanning_bridges=False` produces exactly
the Auto output with `use_spanning_bridges=True`.

A glyph whose explicit direction cannot be built on either axis stays unbridged; three Roboto
glyphs that Auto bridges (its bottom-bar fallback in `MergeDispatch.preferred`) come out unbridged
under a forced direction. The grid marking makes that visible; do not add fallbacks.

## Module contract

### core/processor.py (worker U2)

```python
def process_glyph(glyph_dict, config_dict, upm, reference_stroke_width=None, geometry_dict=None) -> dict[str, Any]
    # unchanged signature; "bridges_added" becomes the number of the analyzer's islands that no
    # longer appear verbatim in the output (0 when no bridge could be placed)

class FontProcessor:
    def process(
        self,
        font_path: Path,
        output_path: Path | None = None,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
        classification: GlyphClassification | None = None,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> ProcessingStats: ...
    # directions maps glyph name -> explicit direction; a glyph absent from it uses
    # self.config.bridge.direction. It reaches process_glyph through that glyph's config_dict.
```

### gui/composites.py (worker U3, new, Qt-free)

```python
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from stencilizer.domain import Glyph, GlyphMetadata
from stencilizer.io import FontReader

Affine = tuple[float, float, float, float, float, float]  # fontTools order: xx, xy, yx, yy, dx, dy


@dataclass(frozen=True)
class ComponentPart:
    """One outline glyph drawn into a composite, with its composed transform."""

    base: str
    transform: Affine


@dataclass(frozen=True)
class CompositeGlyph:
    """A composite glyph that draws at least one island glyph."""

    metadata: GlyphMetadata
    parts: tuple[ComponentPart, ...]
    sources: tuple[str, ...]  # island-glyph bases among parts: first-appearance order, no repeats

    @property
    def name(self) -> str: ...


def component_parts(glyph_set: Any, name: str) -> tuple[ComponentPart, ...]: ...
def find_bridged_composites(
    reader: FontReader, island_names: Collection[str]
) -> tuple[CompositeGlyph, ...]: ...
def load_component_outlines(
    reader: FontReader, composites: Collection[CompositeGlyph]
) -> dict[str, Glyph]: ...
def compose(composite: CompositeGlyph, outlines: Mapping[str, Glyph]) -> Glyph: ...
```

### gui/session.py (worker U6)

```python
@dataclass(frozen=True)
class PreviewResult:  # unchanged fields
    glyph_name: str
    original: Glyph
    stenciled: Glyph | None
    bridges_added: int  # islands actually bridged; 0 with stenciled set = no bridge could be placed
    error: str | None
    duration_ms: float


@dataclass
class FontSession:
    # existing fields, then (all set by open()):
    composites: tuple[CompositeGlyph, ...]
    component_outlines: dict[str, Glyph]
    display_names: tuple[str, ...]  # island glyphs and composites, in the font's glyph order

    @property
    def island_glyphs(self) -> list[Glyph]: ...  # unchanged: classification.glyphs_to_process
    @property
    def display_glyphs(self) -> list[Glyph]: ...  # one Glyph per display_names entry
    def glyph(self, name: str) -> Glyph | None: ...  # island glyph, or a composite's composed outline
    def direction_sources(self, name: str) -> tuple[str, ...]: ...
    def preview(
        self,
        name: str,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> PreviewResult: ...
    def unbridged(
        self,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> frozenset[str]: ...
    def save(
        self,
        output_path: Path,
        settings: StencilizerSettings,
        progress: ProgressCallback | None = None,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> ProcessingStats: ...
```

### gui/direction_picker.py (worker U5, new)

```python
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from stencilizer.config.settings import BridgeDirection


class DirectionPicker(QWidget):
    """Choose the bridge direction of the selected glyph."""

    direction_chosen = Signal(str)  # BridgeDirection value; emitted only on a user change

    def __init__(self, parent: QWidget | None = None) -> None: ...
    # public attributes: combo: QComboBox, source_label: QLabel
    def show_for(self, name: str, sources: tuple[str, ...], direction: BridgeDirection) -> None: ...
    def clear(self) -> None: ...
```

### gui/glyph_grid.py additions (worker U4)

```python
UNBRIDGED_ROLE = Qt.ItemDataRole.UserRole + 1  # bool on every item after set_unbridged
BASE_TOOLTIP_ROLE = Qt.ItemDataRole.UserRole + 2  # the tooltip set_glyphs wrote

class GlyphGrid(QListWidget):
    def set_direction_marker(self, name: str, direction: BridgeDirection) -> None: ...
    def set_unbridged(self, names: Collection[str]) -> None: ...
```

### gui/controller.py additions (worker U7)

```python
SURVEY_DELAY_MS = 250

class GuiController(QObject):
    direction_changed = Signal(str, str)  # glyph name, BridgeDirection value
    unbridged_changed = Signal(object)  # frozenset[str]

    def direction_for(self, name: str) -> BridgeDirection: ...
    def set_direction(self, name: str, direction: BridgeDirection) -> None: ...
```

### gui/main_window.py additions (worker U8)

Public attribute `direction_picker: DirectionPicker`, placed under `comparison` in the right-hand
pane of the splitter.

## Tests

- pytest + pytest-qt (`qtbot`) under `tests/gui/`; `tests/gui/conftest.py` provides `processor`
  (logs into `tmp_path`), `roboto_path`, `commit_mono_path`, `outlines_match(saved, expected)`,
  and an autouse fixture that makes every save start its pool with the spawn context. Never build
  a `FontProcessor` without a `tmp_path` `log_file`.
- No `skip`/`xfail`/`importorskip` in `tests/gui/`, `tests/integration/test_bridge_direction.py`
  or `tests/integration/test_processor_directions.py` (an acceptance command rejects them).
- Codex units write each module together with its own test file (the brief's step 3) and run
  their proof command once, at the end, reporting its output. They never run the tests (codex
  cannot: `tmp_path` needs a writable temp dir); the orchestrator does after each wave. The
  session, controller and main-window tests are one sonnet worker (TG) in the last wave.
- Every test that saves a font asserts `stats.error_count == 0` and reads the saved outlines back.
- These test names are fixed; the stage acceptance runs them by node id:
  `tests/integration/test_bridge_direction.py::test_explicit_direction_splits_o_along_axis`,
  `tests/integration/test_bridge_direction.py::test_stacked_islands_follow_direction`,
  `tests/integration/test_processor_directions.py::test_unbridgeable_glyph_reports_zero_bridges`,
  `tests/integration/test_processor_directions.py::test_process_applies_per_glyph_directions`,
  `tests/gui/test_composites.py::test_composed_outlines_match_fonttools_decomposition`,
  `tests/gui/test_session_directions.py::test_saved_font_changes_only_displayed_glyphs`,
  `tests/gui/test_session_directions.py::test_composite_preview_follows_base_direction`,
  `tests/gui/test_controller_directions.py::test_survey_reports_unbridged_glyphs`,
  `tests/gui/test_controller_directions.py::test_stale_survey_result_is_dropped`,
  `tests/gui/test_main_window_directions.py::test_choosing_direction_updates_preview_and_marker`,
  `tests/gui/test_main_window_directions.py::test_saved_font_uses_chosen_direction`.

## Measured facts tests may rely on (measured at commit 111d115, 2026-09-24)

- Roboto (`tests/fixtures/Roboto-Regular.ttf`): 3387 glyphs, 562 island glyphs, 465 composites
  that draw an island glyph, so 1027 display glyphs. Lato (`tests/fixtures/Lato-Black.ttf`): 447
  island glyphs + 370 composites = 817. CommitMono: 467 island glyphs, 0 composites.
- Composites are read with no contours (`fonttools_glyph_to_domain` records only outline
  segments), so `classify_glyphs` files them under "empty glyph" and the grid never showed them.
  Their bridges come from the referenced island glyph, so the saved font bridges them anyway.
- Roboto `Aacute` = `A` (identity transform) + `acute` (dx 447, dy 310): sources `("A",)`.
  `Aring` sources `("A", "ring")`; `Aringacute` sources `("A", "ringacute")`. No fixture
  composite nests another composite, and none uses a mirroring transform (determinant < 0).
- Roboto with default `BridgeConfig()`: the glyphs with no bridge placed are exactly
  `AE, AEacute, AEmacron, four, four.lnum, four.onum, four.smcp, four.tnum, uni04D4, uniA72C,
  uniA72D, uniA72E, uniA72F, uniA736` (9 island glyphs + 5 composites of them). `O`, `A`, `B`,
  `eight` and `Aacute` are bridged. Lato has 32 such glyphs, CommitMono 0.
- Invariants that hold on all three fixtures with default settings: every glyph whose decomposed
  outline (fontTools `DecomposingRecordingPen` on the glyph set) differs between the input and
  the saved font is a display glyph; every display glyph whose outline is unchanged is in
  `unbridged()`. (Some Lato glyphs are rewritten by the writer without a bridge, so "unbridged"
  does not imply "byte-identical output"; never assert that.)
- Roboto `O` input bbox is (118, -20, 1289, 1476): centre x 703.5, centre y 728. Auto and
  VERTICAL give 4 contours, none of whose bboxes spans x = 703.5 (left/right halves). HORIZONTAL
  gives 4 contours, none of whose bboxes spans y = 728 (top/bottom halves).
- No island glyph of Roboto, Lato or CommitMono raises or returns an error under VERTICAL or
  HORIZONTAL (simulated at planning time); Roboto has 12 unbridged island glyphs under either
  explicit direction versus 9 under Auto.
- A full `unbridged()` pass over one fixture font takes about 0.3 s, which is why the controller
  runs it on the pool behind a 250 ms debounce instead of on every slider tick.
- `.notdef` is the first display glyph of all three fixtures, as before.
