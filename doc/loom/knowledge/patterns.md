# Architectural Patterns

> Discovered patterns in the codebase that help agents understand how things work.
> Keep current: correct or delete entries the code no longer supports.

(Add patterns as you discover them)

## Bridge algorithm (summary)

Analyzer finds islands (holes whose on-curve points sit inside an outer contour, src/stencilizer/core/analyzer.py:202); `ContourMerger` cuts notches into the outer contour rather than adding contours. Horizontal and vertical bridges share one implementation parameterized by `Axis` (core/axis.py); horizontal_bridge.py, vertical_bridge.py, multi_island.py and horizontal_multi_island.py are thin wrappers. Font-unit thresholds scale with UPM through `GeometryConfig`. Full detail, including the Θ and ⑧ multi-island cases: [patterns/bridge-algorithm](patterns/bridge-algorithm.md).

## Winding normalization

Internally all contours use TrueType winding: clockwise = outer (negative signed area), counter-clockwise = hole (`GlyphAnalyzer.analyze`, src/stencilizer/core/analyzer.py:94; `signed_area`, src/stencilizer/core/geometry_polygon.py:8). CFF uses the opposite, so the converter reverses CFF point order on read and again on write (src/stencilizer/io/converter.py). The root CLAUDE.md states the reverse ("TrueType: CCW=outer, CW=inner"); that claim is wrong, the code above is authoritative.

## Exception hierarchy

`StencilizerError` roots `FontError` (`FontLoadError`, `FontSaveError`, `FontFormatError`), `GlyphError`, `GeometryError`, and `ProcessingCancelledError` (src/stencilizer/exceptions.py). The CLI catches `FontLoadError`, `FontSaveError`, then `StencilizerError`, then a catch-all. Worker errors return as `{"error", "traceback"}` dictionaries rather than raised exceptions. The unused bridge placement and generation exception subclasses were removed with the candidate-generation path.

## Settings and logging

`StencilizerSettings` composes pydantic `BridgeConfig`, `GeometryConfig`, `ProcessingConfig`, `LoggingConfig` with `Field(ge=, le=)` bounds (src/stencilizer/config/settings.py:20-158). `configure_logging()` sets up structlog with an always-on file log, auto-named `stencilizer_<timestamp>.log` when `--log-file` is absent (src/stencilizer/utils/logging.py:51-80); `ProcessingStats` tracks counts and per-glyph timings (logging.py:11-48).
