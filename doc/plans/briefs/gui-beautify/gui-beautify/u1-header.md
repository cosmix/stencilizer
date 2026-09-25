# U1: HeaderBar and its tests (codex gpt-5.6-terra, wave 1)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first.

**Owns:** `src/stencilizer/gui/header.py` (new), `tests/gui/test_header.py` (new).
**Reads:** nothing else. U2 deletes the two tests you port from `tests/gui/test_controls.py` in
parallel, so port them from the base text quoted here:

```python
def test_busy_state_restores_save_availability(panel: ControlPanel) -> None:
    """Busy state disables actions and restores save based on font availability."""
    panel.set_font_loaded(True)
    panel.set_busy(True)

    assert not panel.open_button.isEnabled()
    assert not panel.save_button.isEnabled()

    panel.set_busy(False)

    assert panel.open_button.isEnabled()
    assert panel.save_button.isEnabled()

    panel.set_font_loaded(False)
    panel.set_busy(False)

    assert not panel.save_button.isEnabled()


def test_open_and_save_buttons_emit_requests(qtbot: QtBot, panel: ControlPanel) -> None:
    """Clicking the available action buttons emits their request signals."""
    open_calls: list[None] = []
    save_calls: list[None] = []
    panel.open_requested.connect(lambda: open_calls.append(None))
    panel.save_requested.connect(lambda: save_calls.append(None))
    panel.set_font_loaded(True)
    panel.show()

    qtbot.mouseClick(panel.open_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    qtbot.mouseClick(panel.save_button, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]

    assert len(open_calls) == 1
    assert len(save_calls) == 1
```

## Steps

1. Create `src/stencilizer/gui/header.py` with module docstring
   `"""Top bar with the loaded font's identity and the open and save actions."""` and
   `class HeaderBar(QFrame)` with a class docstring and the signals `open_requested = Signal()` and
   `save_requested = Signal()`. `__init__(self, parent: QWidget | None = None) -> None` calls
   `super().__init__(parent)`, `self.setObjectName("headerBar")`, sets `self._font_loaded = False`
   and `self._busy = False`, calls `self._build_widgets()` and `self._build_layout()`, and connects
   `self.open_button.clicked` to `self._emit_open_requested` and `self.save_button.clicked` to
   `self._emit_save_requested`.
   `_build_widgets(self) -> None` creates, all parented to `self`:
   - `self.title_label = QLabel("Stencilizer", self)`, objectName `appTitle`.
   - `self.font_name_label = QLabel("No font loaded", self)`, objectName `fontName`.
   - `self.font_details_label = QLabel("Open a TrueType or OpenType font (.ttf, .otf)", self)`,
     objectName `fontDetails`.
   - Both font labels: `setTextFormat(Qt.TextFormat.PlainText)` (font file names are untrusted text)
     and `setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)` (a long file name
     is clipped instead of pushing the buttons off the bar).
   - `self.open_button = QPushButton("Open Font…", self)` (U+2026 ellipsis),
     `setProperty("role", "secondary")`, tooltip `"Open a TTF or OTF font"`.
   - `self.save_button = QPushButton("Stencilize && Save…", self)`, `setProperty("role", "primary")`,
     tooltip `"Stencilize every glyph and save a new font file"`, `setEnabled(False)`.
   - Both buttons: `setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)` and
     `setCursor(Qt.CursorShape.PointingHandCursor)`.
   `_build_layout(self) -> None`: `layout = QHBoxLayout(self)`, `setContentsMargins(20, 12, 20, 12)`,
   `setSpacing(16)`; `layout.addWidget(self.title_label)`; `info = QVBoxLayout()`,
   `info.setSpacing(2)`, add `font_name_label` then `font_details_label`; `layout.addLayout(info, 1)`;
   `layout.addWidget(self.open_button)`; `layout.addWidget(self.save_button)`. No `addStretch`: the
   info column's stretch factor 1 takes the free width and the buttons stay at their size hint.
   Imports: `from PySide6.QtCore import Qt, Signal` and
   `from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget`.
2. Methods, moved from the base `ControlPanel` with the same semantics, each with a one-line
   docstring:
   - `_emit_open_requested(self, _checked: bool = False) -> None` and
     `_emit_save_requested(self, _checked: bool = False) -> None` emit the matching signal.
   - `set_font_info(self, name: str, details: str) -> None`: `font_name_label.setText(name)`,
     `font_name_label.setToolTip(name)`, `font_details_label.setText(details)`.
   - `set_font_loaded(self, loaded: bool) -> None`: store it;
     `save_button.setEnabled(loaded and not self._busy)`.
   - `set_busy(self, busy: bool) -> None`: store it; `open_button.setEnabled(not busy)`;
     `save_button.setEnabled(self._font_loaded and not busy)`.
3. `tests/gui/test_header.py` (module docstring `"""Tests for the window's top bar."""`), with a
   fixture `header(qtbot: QtBot) -> HeaderBar` that builds `HeaderBar()` and `qtbot.addWidget`s it:
   - `test_header_defaults`: save disabled, open enabled, `font_name_label.text() == "No font loaded"`.
   - `test_set_font_info_updates_labels`: `set_font_info("Roboto-Regular.ttf", "TrueType · 2048 UPM")`
     sets both label texts exactly.
   - `test_busy_state_restores_save_availability` and `test_open_and_save_buttons_emit_requests`:
     the quoted tests with `panel: ControlPanel` replaced by `header: HeaderBar`; every assert kept
     word for word apart from that rename.
   - `test_action_buttons_do_not_expand`: both buttons' `sizePolicy().horizontalPolicy()` is
     `QSizePolicy.Policy.Maximum`.
   Every test and fixture gets a docstring; lambdas connected to signals take no parameters.

## Done

Your one check:
`.venv/bin/mypy src/stencilizer/gui/header.py tests/gui/test_header.py && .venv/bin/ruff check src/stencilizer/gui/header.py tests/gui/test_header.py && .venv/bin/ruff format --check src/stencilizer/gui/header.py tests/gui/test_header.py && .venv/bin/python -m pytest --no-cov -q -p no:cacheprovider --collect-only tests/gui/test_header.py`
exits 0. Every function is at most 50 lines. Do not edit any other file; U5 mounts the bar in
wave 2.
