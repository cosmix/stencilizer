"""Variable-font classification and processing driven by ``FontProcessor``."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

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


class _Inspection(NamedTuple):
    """The verdict on one glyph: its variable data or a skip reason, and its island count."""

    glyph: VariableGlyph | None
    reason: str | None
    islands: int
    detail: str = ""  # why the variation data was unusable, for the skip log


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


def _inspect(reader: FontReader, analyzer: GlyphAnalyzer, name: str) -> _Inspection:
    """Read one glyph and decide whether it is processed, with its default-master island count."""
    font, upm, unicode_by_name = reader.font, reader.units_per_em, reader.unicode_by_name
    try:
        vg = read_variable_glyph(font, name, unicode_by_name)
    except VariationDataError as error:
        glyph = font.getGlyphSet()[name]
        static = fonttools_glyph_to_domain(name, glyph, font, unicode_by_name)
        return _Inspection(None, UNSUPPORTED_REASON, _islands(analyzer, static, upm), error.reason)
    if vg is None:
        composite = "glyf" in font and font["glyf"][name].isComposite()
        return _Inspection(None, "composite glyph" if composite else "empty glyph", 0)
    islands = _islands(analyzer, _counted_default(vg, upm), upm)
    if islands == 0:
        return _Inspection(None, "no islands", 0)
    return _Inspection(vg, None, islands)


def classify_variable_glyphs(
    processor: FontProcessor, reader: FontReader
) -> tuple[GlyphClassification, dict[str, VariableGlyph]]:
    """Select glyphs whose overlap-merged default has islands, with their variable data."""
    glyph_order = reader.font.getGlyphOrder()
    result = GlyphClassification()
    variable: dict[str, VariableGlyph] = {}
    for name in glyph_order:
        inspection = _inspect(reader, processor.analyzer, name)
        if inspection.glyph is not None:
            result.glyphs_to_process.append(inspection.glyph.default)
            variable[name] = inspection.glyph
            continue
        skipped = inspection.reason or "no islands"
        result.skipped_reasons[name] = skipped
        logged = f"{skipped}: {inspection.detail}" if inspection.detail else skipped
        processor.processing_logger.log_glyph_skipped(name, logged)
        if skipped == UNSUPPORTED_REASON and inspection.islands:
            result.unsupported_islands[name] = inspection.islands
    processor.logger.info(
        "Filtered glyphs",
        total=len(glyph_order),
        to_process=len(result.glyphs_to_process),
        skipped=result.skipped_count,
    )
    return result, variable


def variable_island_counts(reader: FontReader) -> list[tuple[str, int]]:
    """Return ``(name, island count)`` for glyphs with islands after overlap removal."""
    analyzer = GlyphAnalyzer()
    counts = []
    for name in reader.font.getGlyphOrder():
        islands = _inspect(reader, analyzer, name).islands
        if islands:
            counts.append((name, islands))
    return counts


def _selected_glyphs(
    processor: FontProcessor, reader: FontReader, classification: GlyphClassification | None
) -> tuple[GlyphClassification, dict[str, VariableGlyph]]:
    """Return the classification and the variable glyphs to process, reading no glyph twice.

    A given classification may name a glyph whose variation data no longer reads. Such a glyph
    is left unchanged, counted as skipped, and its default islands count as unbridged. The
    returned classification is a copy; the caller's is not modified.
    """
    if classification is None:
        return classify_variable_glyphs(processor, reader)
    variable: dict[str, VariableGlyph] = {}
    kept: list[Glyph] = []
    skipped = dict(classification.skipped_reasons)
    unsupported = dict(classification.unsupported_islands)
    for glyph in classification.glyphs_to_process:
        try:
            vg = read_variable_glyph(reader.font, glyph.name, reader.unicode_by_name)
        except VariationDataError as error:
            processor.logger.warning("Glyph left unchanged", glyph=glyph.name, reason=str(error))
            skipped[glyph.name] = UNSUPPORTED_REASON
            islands = _islands(processor.analyzer, glyph, reader.units_per_em)
            if islands:
                unsupported[glyph.name] = islands
            continue
        kept.append(glyph)
        if vg is not None:
            variable[glyph.name] = vg
    selected = dataclasses.replace(
        classification,
        glyphs_to_process=kept,
        skipped_reasons=skipped,
        unsupported_islands=unsupported,
    )
    return selected, variable


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
