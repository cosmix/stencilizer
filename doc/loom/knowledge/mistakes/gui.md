# Gui

> fontTools TTFont iteration, Qt widget lifetime in tests

## Iterating a TTFont breaks every font open

**What happened:** The font info builder listed tables with `for tag in font`. Every open in the GUI failed with "Failed to load font ...: '0'".
**Why:** `TTFont` has `keys()` and `__getitem__` but no `__iter__`, so Python falls back to `font[0]`, `font[1]`, ... and the sfnt reader raises `KeyError('0')`. ruff SIM118 rewrote a correct `font.keys()` loop into the broken form.
**Prevention:** Iterate `font.keys()` with `# noqa: SIM118` and a comment saying why. Any code that reads a `TTFont` needs a test that opens real fixtures through `FontSession.open`, not only synthetic fonts.
**Fix:** `font_info.py` iterates `font.keys()`; `test_real_fonts_open_with_info` opens Roboto, Lato-Black, CommitMono, Ubuntu and Inter.

## Calling methods on a temporary parentless widget in tests

**What happened:** `_panel(qtbot).findChildren(...)` raised "C++ object already deleted" mid-iteration.
**Why:** No Python reference held the parentless widget, so it was garbage-collected along with its C++ object.
**Prevention:** Bind a parentless widget under test to a local variable (or `qtbot.addWidget`) before using it.
