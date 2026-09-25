# U4: Comparison view cards (codex gpt-6-luna, wave 1)

Read `doc/plans/briefs/gui-beautify/gui-beautify/_shared.md` first.

**Owns:** `src/stencilizer/gui/glyph_view.py`.
**Start from:** `ComparisonView.__init__` in `src/stencilizer/gui/glyph_view.py` (lines 57-74 at
base). Leave `GlyphCanvas` (all of it, including `paintEvent`), `show_preview`, `clear` and
`_preview_text` unchanged: `tests/gui/test_glyph_view.py` compares canvas pixels with
`canvas.palette().base()` and `.text()`, and asserts the info text.

## Steps

1. Rewrite `ComparisonView.__init__`: create `self.before_canvas = GlyphCanvas(self)` and
   `self.after_canvas = GlyphCanvas(self)`; `self.info_label = QLabel(self)` with
   `setAlignment(Qt.AlignmentFlag.AlignCenter)`, `setTextFormat(Qt.TextFormat.PlainText)` (glyph
   names come from the font file) and `setProperty("role", "status")`. `layout = QGridLayout(self)`,
   `setContentsMargins(0, 0, 0, 0)`, `setHorizontalSpacing(12)`, `setVerticalSpacing(10)`;
   `layout.addWidget(self._card("ORIGINAL", self.before_canvas), 0, 0)`;
   `layout.addWidget(self._card("STENCILIZED", self.after_canvas), 0, 1)`;
   `layout.addWidget(self.info_label, 1, 0, 1, 2)`; `layout.setRowStretch(0, 1)`.
2. New method `_card(self, title: str, canvas: GlyphCanvas) -> QFrame` with a one-line docstring:
   `card = QFrame(self)`, `card.setProperty("role", "card")`; `label = QLabel(title, card)`,
   `label.setProperty("role", "sectionTitle")`, `label.setAlignment(Qt.AlignmentFlag.AlignCenter)`;
   `box = QVBoxLayout(card)`, `setContentsMargins(12, 10, 12, 12)`, `setSpacing(6)`;
   `box.addWidget(label)`; `box.addWidget(canvas, 1)`; return `card`.
3. Imports: add `QFrame` and `QVBoxLayout` to the `PySide6.QtWidgets` import.

## Done

`.venv/bin/mypy src/stencilizer/gui/glyph_view.py && .venv/bin/ruff check src/stencilizer/gui/glyph_view.py && .venv/bin/ruff format --check src/stencilizer/gui/glyph_view.py`
exits 0.
