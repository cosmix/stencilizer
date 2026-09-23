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

`stencilizer-gui` is a PySide6 window over the unchanged core: `FontSession` (Qt-free) opens, previews and saves; `GuiController` runs open and save as `QRunnable`s with queued signals; saves stage in a private temp dir and never write through the output directory. Layout, threading and save-safety detail: [architecture/gui](architecture/gui.md).
