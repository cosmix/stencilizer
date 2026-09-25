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

Read: `fonttools_glyph_to_domain()` records any outline with `RecordingPen`; CFF contours are point-reversed to TrueType winding (src/stencilizer/io/converter.py:48-54). Write: `domain_glyph_to_fonttools()` branches on `"glyf"` → `_update_truetype_glyph`, `"CFF "` → `_update_cff_glyph` (reverses back, converter.py:243-265), anything else raises `NotImplementedError` (converter.py:89-95). CFF2 is only named in format detection (src/stencilizer/io/reader.py:63); there is no CFF2 write path.

## GUI (summary)

`stencilizer-gui` is a PySide6 window over the unchanged core: `FontSession` (Qt-free) opens, previews and saves; the grid lists island glyphs plus composites that draw one (`gui/composites.py`); `GuiController` runs open and save as `QRunnable`s with queued signals, keeps per-glyph bridge directions, and runs a debounced unbridged-glyph survey on the pool; saves stage in a private temp dir and never write through the output directory. `MainWindow` is a header bar over a splitter of sidebar, glyph grid and preview pane, with save progress in the status bar; `gui/theme.py` styles it in light or dark and follows the system scheme. Layout, theme, threading and save-safety detail: [architecture/gui](architecture/gui.md).

## Font format and error boundary

Review fixes reject variable fonts (fvar) and CFF2 before conversion and expose glyph conversion errors instead of silently omitting glyphs. Unsupported inputs must fail explicitly without publishing an output font.

## Processing outcomes

Operational glyph worker, conversion, and write failures abort font publication. Geometrically unbridgeable islands are explicit incomplete outcomes: count only confirmed hole connections as bridges, retain an unbridged count, and warn in the CLI rather than converting an expected geometric limitation into a worker crash.

## Preserve no-op font glyphs

A worker outcome with zero confirmed bridges must not enqueue a glyph rewrite. Keeping the original font glyph avoids dropping TrueType hint instructions or changing encoding merely because analysis found an unbridgeable island. Its unbridged count remains visible in processing statistics.
