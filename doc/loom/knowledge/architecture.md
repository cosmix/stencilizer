# Architecture

> High-level component relationships, data flow, and module dependencies.
> Keep current: correct or delete entries the code no longer supports.

(Add architecture diagrams and component relationships as you discover them)

## Processing pipeline

`cli/app.py:stencilize()` validates flags, builds `StencilizerSettings`, classifies glyphs once via `FontProcessor.classify_glyphs(reader)` for the island list, and passes that `GlyphClassification` to `FontProcessor.process(..., classification=...)`.

`FontProcessor.process()` classifies (or reuses the classification), dispatches selected glyphs to worker processes with `BridgeConfig` and `GeometryConfig` dictionaries, then writes through `FontWriter` using `FontReader.font`. It does not measure a font-wide stroke width.

Per glyph: `GlyphAnalyzer` → `GlyphTransformer.transform()` → `ContourMerger` → axis-generic contour builders. Bridge width is `width_percent / 100 * (upm * 0.1)`. Algorithm detail: [patterns/bridge-algorithm](patterns/bridge-algorithm.md).

## Process-pool IPC

`process_glyph()` is module-level so `ProcessPoolExecutor` can pickle it. Production workers receive `Glyph.to_dict()`, `BridgeConfig.model_dump()`, UPM, and `GeometryConfig.model_dump()` as a keyword argument. The fourth positional argument remains accepted for frozen callers but is ignored. The worker rebuilds the glyph and configs, constructs an analyzer and transformer, and returns `{"glyph", "bridges_added", "duration_ms"}` or `{"error", "traceback"}`. KeyboardInterrupt cancels pending futures. `Glyph`, `Contour`, `Point`, and `GlyphMetadata` serialize for worker IPC.

## Font format I/O

Read: `fonttools_glyph_to_domain()` records any outline with `RecordingPen`; contours are point-reversed to TrueType winding only when `"CFF " in font` (src/stencilizer/io/converter.py:30-35), so CFF2 is not normalized. Write: `domain_glyph_to_fonttools()` (converter.py:51) checks `"glyf"` first → `_update_truetype_glyph` (converter.py:161), then `"CFF "` → `_update_cff_glyph` (converter.py:232, reverses back), anything else raises `NotImplementedError` (converter.py:65-72). The `glyf`-first check means a variable TrueType glyph written through this path would get a new glyf entry with its gvar left stale. There is no CFF2 write path; `FontReader.format` labels CFF2 "OpenType" (src/stencilizer/io/reader.py:76).

## GUI (summary)

`stencilizer-gui` is a PySide6 window over the unchanged core: `FontSession` (Qt-free) opens, previews and saves; the grid lists island glyphs plus composites that draw one (`gui/composites.py`); `GuiController` runs open and save as `QRunnable`s with queued signals, keeps per-glyph bridge directions, and runs a debounced unbridged-glyph survey on the pool; saves stage in a private temp dir and never write through the output directory. `MainWindow` is a header bar over a splitter of sidebar, glyph grid and preview pane, with save progress in the status bar; `gui/theme.py` styles it in light or dark and follows the system scheme. Layout, theme, threading and save-safety detail: [architecture/gui](architecture/gui.md).

## Font format and error boundary

`FontReader.load` and `FontWriter` reject variable fonts (fvar) and CFF2 before conversion with `FontFormatError`, and glyph conversion errors are raised instead of silently omitting glyphs. Unsupported inputs must fail explicitly without publishing an output font.

## Processing outcomes

Operational glyph worker, conversion, and write failures abort font publication. Geometrically unbridgeable islands are explicit incomplete outcomes: count only confirmed hole connections as bridges, retain an unbridged count, and warn in the CLI rather than converting an expected geometric limitation into a worker crash.

## Preserve no-op font glyphs

A worker outcome with zero confirmed bridges must not enqueue a glyph rewrite. Keeping the original font glyph avoids dropping TrueType hint instructions or changing encoding merely because analysis found an unbridgeable island. Its unbridged count remains visible in processing statistics.
