"""Variable-font classification and processing driven by ``FontProcessor``."""

from collections.abc import Mapping
from pathlib import Path

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config.settings import BridgeDirection
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.processor import FontProcessor, GlyphClassification, ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.exceptions import FontProcessingError, VariationDataError
from stencilizer.io import FontReader
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.utils import ProcessingStats
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.reader import read_variable_glyph
from stencilizer.variable.transform import process_variable_glyph

__all__ = [
    "classify_variable_glyphs",
    "flatten_compatible",
    "process_variable_font",
    "read_variable_glyph",
    "variable_island_counts",
]

UNSUPPORTED_REASON = "unsupported variation data"

# (variable glyph or None, skip reason or None, island count)
_Inspection = tuple[VariableGlyph | None, str | None, int]


def _islands(analyzer: GlyphAnalyzer, glyph: Glyph, upm: int) -> int:
    return len(analyzer.analyze(glyph, upm).get_islands())


def _counted_default(vg: VariableGlyph, upm: int) -> Glyph:
    """Return the default the worker counts islands on: merged, else flattened, else raw."""
    try:
        flat = flatten_compatible(vg, upm)
    except VariationDataError:
        return vg.default
    try:
        merged = remove_overlaps_compatible(flat)
    except VariationDataError:
        return flat.default
    return (merged if merged is not None else flat).default


def _inspect(font: TTFont, analyzer: GlyphAnalyzer, name: str, upm: int) -> _Inspection:
    """Read one glyph and decide whether it is processed, with its default-master island count."""
    try:
        vg = read_variable_glyph(font, name)
    except VariationDataError:
        static = fonttools_glyph_to_domain(name, font.getGlyphSet()[name], font)
        return None, UNSUPPORTED_REASON, _islands(analyzer, static, upm)
    if vg is None:
        composite = "glyf" in font and font["glyf"][name].isComposite()
        return None, "composite glyph" if composite else "empty glyph", 0
    islands = _islands(analyzer, _counted_default(vg, upm), upm)
    if islands == 0:
        return None, "no islands", 0
    return vg, None, islands


def classify_variable_glyphs(
    processor: FontProcessor, reader: FontReader
) -> tuple[GlyphClassification, dict[str, VariableGlyph]]:
    """Select glyphs whose overlap-merged default has islands, with their variable data."""
    font, upm = reader.font, reader.units_per_em
    result = GlyphClassification()
    variable: dict[str, VariableGlyph] = {}
    for name in font.getGlyphOrder():
        vg, reason, islands = _inspect(font, processor.analyzer, name, upm)
        if vg is not None:
            result.glyphs_to_process.append(vg.default)
            variable[name] = vg
            continue
        skipped = reason or "no islands"
        result.skipped_reasons[name] = skipped
        processor.processing_logger.log_glyph_skipped(name, skipped)
        if skipped == UNSUPPORTED_REASON and islands:
            result.unsupported_islands[name] = islands
    processor.logger.info(
        "Filtered glyphs",
        total=len(font.getGlyphOrder()),
        to_process=len(result.glyphs_to_process),
        skipped=result.skipped_count,
    )
    return result, variable


def variable_island_counts(reader: FontReader) -> list[tuple[str, int]]:
    """Return ``(name, island count)`` for glyphs with islands after overlap removal."""
    font, upm = reader.font, reader.units_per_em
    analyzer = GlyphAnalyzer()
    counts = []
    for name in font.getGlyphOrder():
        _, _, islands = _inspect(font, analyzer, name, upm)
        if islands:
            counts.append((name, islands))
    return counts


def _selected_glyphs(
    processor: FontProcessor, reader: FontReader, classification: GlyphClassification | None
) -> tuple[GlyphClassification, dict[str, VariableGlyph]]:
    """Return the classification and the variable glyphs to process, reading no glyph twice."""
    if classification is None:
        return classify_variable_glyphs(processor, reader)
    variable: dict[str, VariableGlyph] = {}
    for glyph in classification.glyphs_to_process:
        try:
            vg = read_variable_glyph(reader.font, glyph.name)
        except VariationDataError as error:
            processor.logger.warning("Glyph left unchanged", glyph=glyph.name, reason=str(error))
            continue
        if vg is not None:
            variable[glyph.name] = vg
    return classification, variable


def process_variable_font(
    processor: FontProcessor,
    reader: FontReader,
    output_path: Path,
    max_workers: int | None,
    stats: ProcessingStats,
    progress_callback: ProgressCallback | None,
    classification: GlyphClassification | None,
    directions: Mapping[str, BridgeDirection] | None,
) -> None:
    """Stencil a variable font, filling ``stats`` in place and saving to ``output_path``."""
    selected, variable = _selected_glyphs(processor, reader, classification)
    stats.skipped_count = selected.skipped_count
    stats.unbridged_count += sum(selected.unsupported_islands.values())
    processed: dict[str, VariableGlyph] = {}
    if variable:
        processed = processor._process_glyphs_parallel(
            list(variable.values()),
            reader.units_per_em,
            max_workers,
            stats,
            progress_callback,
            directions,
            worker=process_variable_glyph,
            rebuild=VariableGlyph.from_dict,
        )
    else:
        processor.logger.info("No glyphs to process")
    if stats.error_count:
        raise FontProcessingError(stats.errors)
    processor._save_font(reader, output_path, processed, variable=True)
