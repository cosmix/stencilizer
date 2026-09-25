"""Parallel processing orchestration for stencilization."""

import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.config.settings import BridgeDirection, GeometryConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain import Glyph
from stencilizer.exceptions import FontFormatError, FontProcessingError, FontSaveError
from stencilizer.io import FontReader, FontWriter
from stencilizer.utils import ProcessingLogger, ProcessingStats, configure_logging

ProgressCallback = Callable[[int, int, str, bool], None]


@dataclass
class GlyphClassification:
    """Glyphs selected for processing and reasons for skipped glyphs."""

    glyphs_to_process: list[Glyph] = field(default_factory=list)
    skipped_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def skipped_count(self) -> int:
        """Return the number of skipped glyphs."""
        return len(self.skipped_reasons)


def _islands_bridged(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """Count entries in ``before`` with no matching entry left in ``after``.

    An unbridged island is appended verbatim to the transformed contours, so it
    consumes one matching entry from ``after`` per occurrence; contour dicts are
    unhashable, so matching uses list membership (multiset semantics) rather than
    a ``Counter``.
    """
    remaining = list(after)
    unmatched = 0
    for island in before:
        if island in remaining:
            remaining.remove(island)
        else:
            unmatched += 1
    return unmatched


def _transform_glyph(
    glyph_dict: dict[str, Any],
    config_dict: dict[str, Any],
    upm: int,
    geometry_dict: dict[str, Any] | None,
) -> tuple[Glyph, int, int]:
    glyph = Glyph.from_dict(glyph_dict)
    bridge_config = BridgeConfig(**config_dict)
    geometry_config = (
        GeometryConfig(**geometry_dict) if geometry_dict is not None else GeometryConfig()
    )
    analyzer = GlyphAnalyzer()
    transformer = GlyphTransformer(
        analyzer=analyzer,
        bridge_config=bridge_config,
        geometry_config=geometry_config,
    )
    outcome = transformer.transform_with_outcome(glyph, upm=upm)
    return outcome.glyph, outcome.bridge_count, outcome.unbridged_count


def process_glyph(
    glyph_dict: dict[str, Any],
    config_dict: dict[str, Any],
    upm: int,
    reference_stroke_width: float | None = None,  # noqa: ARG001
    geometry_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a serialized glyph in a worker and return its result.

    The fourth argument is accepted for compatibility and ignored.
    """
    start_time = time.time()
    try:
        transformed_glyph, bridges_added, unbridged_count = _transform_glyph(
            glyph_dict, config_dict, upm, geometry_dict
        )
        return {
            "glyph": transformed_glyph.to_dict(),
            "bridges_added": bridges_added,
            "unbridged_count": unbridged_count,
            "duration_ms": (time.time() - start_time) * 1000,
        }
    except Exception as error:
        return {
            "error": str(error),
            "glyph_name": glyph_dict.get("metadata", {}).get("name", "unknown"),
            "traceback": traceback.format_exc(),
            "duration_ms": (time.time() - start_time) * 1000,
        }


def _config_for_glyph(
    config_dict: dict[str, Any],
    directions: Mapping[str, BridgeDirection] | None,
    name: str,
) -> dict[str, Any]:
    """Return the shared config or a glyph-specific direction override."""
    if directions is None or name not in directions:
        return config_dict
    return {**config_dict, "direction": directions[name]}


class FontProcessor:
    """Orchestrate font stencilization and glyph processing."""

    def __init__(self, config: StencilizerSettings, quiet: bool = False) -> None:
        """Initialize processing services."""
        self.config = config
        self.logger = configure_logging(
            log_file=config.logging.log_file,
            console_level=config.logging.log_level,
            file_level=config.logging.file_log_level,
            quiet=quiet,
        )
        self.processing_logger = ProcessingLogger(self.logger)
        self.analyzer = GlyphAnalyzer()

    def classify_glyphs(self, reader: FontReader) -> GlyphClassification:
        """Classify loaded glyphs once for processing."""
        result = GlyphClassification()
        total = 0
        for glyph in reader.iter_glyphs():
            total += 1
            reason = None
            if glyph.is_empty():
                reason = "empty glyph"
            elif self.config.processing.skip_composite and glyph.is_composite():
                reason = "composite glyph"
            else:
                hierarchy = self.analyzer.analyze(glyph)
                if hierarchy.has_islands():
                    result.glyphs_to_process.append(glyph)
                else:
                    reason = "no islands"
            if reason is not None:
                result.skipped_reasons[glyph.name] = reason
                self.processing_logger.log_glyph_skipped(glyph.name, reason)
        self.logger.info(
            "Filtered glyphs",
            total=total,
            to_process=len(result.glyphs_to_process),
            skipped=result.skipped_count,
        )
        return result

    def process(
        self,
        font_path: Path,
        output_path: Path | None = None,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
        classification: GlyphClassification | None = None,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> ProcessingStats:
        """Process a font, optionally reusing a prior glyph classification."""
        stats = ProcessingStats()
        stats.start_time = time.time()
        if max_workers is None:
            max_workers = self.config.processing.max_workers
        if output_path is None:
            output_path = FontWriter.get_stenciled_path(font_path)
        self.logger.info(
            "Starting font processing",
            input=str(font_path),
            output=str(output_path),
            max_workers=max_workers,
        )
        reader = FontReader(font_path)
        try:
            reader.load()
            self._process_loaded_font(
                reader,
                output_path,
                max_workers,
                stats,
                progress_callback,
                classification,
                directions,
            )
        finally:
            reader.close()
        stats.end_time = time.time()
        self.logger.info(
            "Processing complete",
            processed=stats.processed_count,
            skipped=stats.skipped_count,
            errors=stats.error_count,
            bridges_added=stats.bridges_added,
            unbridged=stats.unbridged_count,
            duration_seconds=round(stats.duration_seconds, 2),
        )
        return stats

    def _process_loaded_font(
        self,
        reader: FontReader,
        output_path: Path,
        max_workers: int | None,
        stats: ProcessingStats,
        progress_callback: ProgressCallback | None,
        classification: GlyphClassification | None,
        directions: Mapping[str, BridgeDirection] | None,
    ) -> None:
        upm = reader.units_per_em
        self.logger.info(
            "Font loaded",
            format=reader.format,
            upm=upm,
            glyph_count=reader.glyph_count,
        )
        selected = classification if classification is not None else self.classify_glyphs(reader)
        stats.skipped_count = selected.skipped_count
        if selected.glyphs_to_process:
            processed = self._process_glyphs_parallel(
                glyphs=selected.glyphs_to_process,
                upm=upm,
                max_workers=max_workers,
                stats=stats,
                progress_callback=progress_callback,
                directions=directions,
            )
        else:
            self.logger.info("No glyphs to process")
            processed = {}
        if stats.error_count:
            raise FontProcessingError(stats.errors)
        self._save_font(reader, output_path, processed)

    def _process_glyphs_parallel(
        self,
        glyphs: list[Glyph],
        upm: int,
        max_workers: int | None,
        stats: ProcessingStats,
        progress_callback: ProgressCallback | None = None,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> dict[str, Glyph]:
        """Dispatch glyphs to workers and collect their results."""
        processed: dict[str, Glyph] = {}
        config_dict = self.config.bridge.model_dump()
        geometry_dict = self.config.geometry.model_dump()
        tasks = {glyph.name: glyph.to_dict() for glyph in glyphs}
        self.logger.info(
            "Starting parallel processing", glyph_count=len(tasks), max_workers=max_workers
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Any, str] = {}
            for name, glyph_dict in tasks.items():
                future = executor.submit(
                    process_glyph,
                    glyph_dict,
                    _config_for_glyph(config_dict, directions, name),
                    upm,
                    geometry_dict=geometry_dict,
                )
                pending[future] = name
            try:
                for completed, future in enumerate(as_completed(pending), 1):
                    name = pending.pop(future)
                    success = self._collect_result(future, name, stats, processed)
                    if progress_callback is not None:
                        progress_callback(completed, len(tasks), name, success)
            except KeyboardInterrupt:
                self.logger.info("Cancellation requested by user")
                for future in pending:
                    future.cancel()
                stats.was_cancelled = True
                stats.cancelled_count = len(pending)
                executor.shutdown(wait=True, cancel_futures=True)
                raise
        return processed

    def _collect_result(
        self,
        future: Any,
        name: str,
        stats: ProcessingStats,
        processed: dict[str, Glyph],
    ) -> bool:
        try:
            result = future.result()
            if "error" in result:
                error_message = result["error"]
                self.processing_logger.log_glyph_error(
                    glyph_name=name,
                    error=Exception(error_message),
                    traceback=result.get("traceback"),
                )
                stats.error_count += 1
                stats.errors.append((name, error_message))
                return False
            bridges_added = result["bridges_added"]
            if bridges_added:
                processed[name] = Glyph.from_dict(result["glyph"])
            stats.processed_count += 1
            stats.bridges_added += bridges_added
            unbridged_count = result.get("unbridged_count", 0)
            stats.unbridged_count += unbridged_count
            if unbridged_count:
                self.logger.warning(
                    "Glyph has unbridged islands", glyph=name, count=unbridged_count
                )
            duration_ms = result.get("duration_ms", 0.0)
            self.processing_logger.log_glyph_complete(
                glyph_name=name, bridges_added=bridges_added, duration_ms=duration_ms
            )
            stats.glyph_timings_ms.append(duration_ms)
            return True
        except Exception as error:
            self.processing_logger.log_glyph_error(
                glyph_name=name, error=error, traceback=traceback.format_exc()
            )
            stats.error_count += 1
            stats.errors.append((name, str(error)))
            return False

    def _save_font(
        self,
        reader: FontReader,
        output_path: Path,
        processed_glyphs: dict[str, Glyph],
    ) -> None:
        """Write transformed glyphs into the loaded font and save it."""
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}-",
                suffix=output_path.suffix,
                dir=output_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_name = temporary_file.name
            writer = FontWriter(reader.font, Path(temporary_name))
            for glyph_name, glyph in processed_glyphs.items():
                try:
                    writer.update_glyph(glyph)
                except Exception as error:
                    raise FontSaveError(
                        str(output_path), f"failed to update glyph '{glyph_name}': {error}"
                    ) from error
            try:
                writer.save()
            except FontSaveError as error:
                raise FontSaveError(str(output_path), error.reason) from error
            except FontFormatError:
                raise
            except Exception as error:
                raise FontSaveError(str(output_path), str(error)) from error
            Path(temporary_name).replace(output_path)
        except OSError as error:
            raise FontSaveError(str(output_path), str(error)) from error
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        self.logger.info(
            "Font saved", output=str(output_path), updated_glyphs=len(processed_glyphs)
        )
