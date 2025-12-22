"""Parallel processing orchestration for stencilization pipeline.

This module coordinates the full stencilization workflow with parallel processing
of individual glyphs using ProcessPoolExecutor.

Key components:
- process_glyph: Top-level picklable function for parallel execution
- FontProcessor: Main orchestrator class for font processing
"""

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain import Glyph
from stencilizer.io import FontReader, FontWriter
from stencilizer.utils import ProcessingLogger, ProcessingStats, configure_logging


def process_glyph(
    glyph_dict: dict[str, Any],
    config_dict: dict[str, Any],
    upm: int,
    reference_stroke_width: float | None = None,
) -> dict[str, Any]:
    """Process a single glyph with stencilization.

    Top-level function designed to be picklable for use with ProcessPoolExecutor.
    Deserializes glyph, applies transformation, and returns result.

    Args:
        glyph_dict: Serialized glyph (from Glyph.to_dict())
        config_dict: Serialized bridge configuration
        upm: Font units per em
        reference_stroke_width: Reference stroke width for consistent bridge sizing

    Returns:
        Dictionary containing either:
        - Success: {"glyph": glyph_dict, "bridges_added": int, "duration_ms": float}
        - Error: {"error": str, "glyph_name": str, "traceback": str, "duration_ms": float}
    """
    start_time = time.time()

    try:
        # Deserialize glyph
        glyph = Glyph.from_dict(glyph_dict)

        # Create configuration
        bridge_config = BridgeConfig(**config_dict)

        # Initialize transformer components
        analyzer = GlyphAnalyzer()
        placer = BridgePlacer(config=bridge_config)
        generator = BridgeGenerator(
            config=bridge_config,
            reference_stroke_width=reference_stroke_width,
        )
        transformer = GlyphTransformer(
            analyzer=analyzer,
            placer=placer,
            generator=generator,
        )

        # Count original islands using analyzer
        original_hierarchy = analyzer.analyze(glyph)
        original_island_count = len(original_hierarchy.get_islands())

        # Transform glyph (cuts notches into contours for bridge gaps)
        transformed_glyph = transformer.transform(glyph, upm=upm)

        # With the notch-cutting approach, islands remain but have bridge gaps
        # Count bridges as the number of islands we processed
        bridges_added = original_island_count

        duration_ms = (time.time() - start_time) * 1000
        return {
            "glyph": transformed_glyph.to_dict(),
            "bridges_added": bridges_added,
            "duration_ms": duration_ms,
        }

    except Exception as e:
        # Capture full traceback for debugging
        duration_ms = (time.time() - start_time) * 1000
        tb = traceback.format_exc()
        return {
            "error": str(e),
            "glyph_name": glyph_dict.get("metadata", {}).get("name", "unknown"),
            "traceback": tb,
            "duration_ms": duration_ms,
        }


class FontProcessor:
    """Orchestrates parallel font stencilization processing.

    Manages the complete workflow:
    1. Load font file
    2. Filter glyphs requiring processing (those with islands)
    3. Process glyphs in parallel using worker processes
    4. Collect results and update statistics
    5. Save modified font

    Example:
        settings = StencilizerSettings()
        processor = FontProcessor(settings)
        stats = processor.process(
            font_path=Path("font.ttf"),
            output_path=Path("font-stenciled.ttf"),
            max_workers=4
        )
    """

    # Reference glyphs for calculating stroke width (in priority order)
    REFERENCE_GLYPHS: ClassVar[list[str]] = ["H", "I", "l", "O", "o", "zero"]

    def __init__(self, config: StencilizerSettings) -> None:
        """Initialize font processor with configuration.

        Args:
            config: Stencilizer settings containing bridge and processing config
        """
        self.config = config
        self.logger = configure_logging(
            log_file=config.logging.log_file,
            console_level=config.logging.log_level,
            file_level=config.logging.file_log_level,
            quiet=False,
        )
        self.processing_logger = ProcessingLogger(self.logger)
        self.analyzer = GlyphAnalyzer()

    def _calculate_reference_stroke_width(self, glyphs: list[Glyph], upm: int) -> float:
        """Calculate reference stroke width from reference glyphs.

        Uses glyphs like H, I, O to measure typical stroke width.
        Falls back to average stroke width across all glyphs with islands.

        Args:
            glyphs: List of all glyphs in font
            upm: Font units per em

        Returns:
            Reference stroke width in font units
        """
        from stencilizer.core.geometry import nearest_point_on_contour

        # Build name lookup
        glyph_by_name = {g.name: g for g in glyphs}

        stroke_widths: list[float] = []

        # Try reference glyphs first
        for ref_name in self.REFERENCE_GLYPHS:
            if ref_name in glyph_by_name:
                glyph = glyph_by_name[ref_name]
                hierarchy = self.analyzer.analyze(glyph)

                # If glyph has islands, measure stroke width
                if hierarchy.has_islands():
                    for island_idx in hierarchy.islands:
                        parent_idx = hierarchy.containment.get(island_idx)
                        if parent_idx is not None:
                            island = glyph.contours[island_idx]
                            outer = glyph.contours[parent_idx]

                            # Sample a few points and measure distances
                            samples = island.sample_points(8)
                            for pt in samples:
                                _, dist = nearest_point_on_contour(pt, outer)
                                if dist > 0:
                                    stroke_widths.append(dist)

        # If we got measurements from reference glyphs, use their average
        if stroke_widths:
            avg = sum(stroke_widths) / len(stroke_widths)
            self.logger.info(
                "Calculated reference stroke width from reference glyphs",
                stroke_width=round(avg, 1),
                samples=len(stroke_widths),
            )
            return avg

        # Fallback: measure from any glyphs with islands
        for glyph in glyphs:
            hierarchy = self.analyzer.analyze(glyph)
            if hierarchy.has_islands():
                for island_idx in hierarchy.islands:
                    parent_idx = hierarchy.containment.get(island_idx)
                    if parent_idx is not None:
                        island = glyph.contours[island_idx]
                        outer = glyph.contours[parent_idx]

                        samples = island.sample_points(4)
                        for pt in samples:
                            _, dist = nearest_point_on_contour(pt, outer)
                            if dist > 0:
                                stroke_widths.append(dist)

                # Stop after getting enough samples
                if len(stroke_widths) >= 20:
                    break

        if stroke_widths:
            avg = sum(stroke_widths) / len(stroke_widths)
            self.logger.info(
                "Calculated reference stroke width from fallback glyphs",
                stroke_width=round(avg, 1),
                samples=len(stroke_widths),
            )
            return avg

        # Ultimate fallback: use 10% of UPM
        fallback = upm * 0.10
        self.logger.warning(
            "Could not calculate stroke width, using fallback",
            stroke_width=round(fallback, 1),
        )
        return fallback

    def process(
        self,
        font_path: Path,
        output_path: Path | None = None,
        max_workers: int | None = None,
        progress_callback: Callable[[int, int, str, bool], None] | None = None,
    ) -> ProcessingStats:
        """Process a font file with parallel glyph processing.

        Args:
            font_path: Path to input font file (TTF or OTF)
            output_path: Path for output font (auto-generated if None)
            max_workers: Maximum worker processes (None = auto-detect)
            progress_callback: Optional callback(completed, total, glyph_name, success)
                for progress updates

        Returns:
            ProcessingStats with counts, timing, and error details

        Raises:
            FileNotFoundError: If font file does not exist
            ValueError: If font format is not supported
            KeyboardInterrupt: If processing is cancelled by user
        """
        # Initialize stats
        stats = ProcessingStats()
        stats.start_time = time.time()

        # Use config default if max_workers not specified
        if max_workers is None:
            max_workers = self.config.processing.max_workers

        # Determine output path
        if output_path is None:
            output_path = FontWriter.get_stenciled_path(font_path)

        self.logger.info(
            "Starting font processing",
            input=str(font_path),
            output=str(output_path),
            max_workers=max_workers,
        )

        # Load font
        reader = FontReader(font_path)
        reader.load()

        try:
            upm = reader.units_per_em

            self.logger.info(
                "Font loaded",
                format=reader.format,
                upm=upm,
                glyph_count=reader.glyph_count,
            )

            # Filter glyphs with islands
            glyphs_to_process: list[Glyph] = []
            all_glyphs: list[Glyph] = []

            for glyph in reader.iter_glyphs():
                all_glyphs.append(glyph)

                # Skip empty glyphs
                if glyph.is_empty():
                    stats.skipped_count += 1
                    self.processing_logger.log_glyph_skipped(
                        glyph.name, "empty glyph"
                    )
                    continue

                # Skip composite glyphs if configured
                if self.config.processing.skip_composite and glyph.is_composite():
                    stats.skipped_count += 1
                    self.processing_logger.log_glyph_skipped(
                        glyph.name, "composite glyph"
                    )
                    continue

                # Check for islands using analyzer
                hierarchy = self.analyzer.analyze(glyph)
                if hierarchy.has_islands():
                    glyphs_to_process.append(glyph)
                else:
                    stats.skipped_count += 1
                    self.processing_logger.log_glyph_skipped(
                        glyph.name, "no islands"
                    )

            self.logger.info(
                "Filtered glyphs",
                total=len(all_glyphs),
                to_process=len(glyphs_to_process),
                skipped=stats.skipped_count,
            )

            # Calculate reference stroke width for consistent bridges
            reference_stroke_width = self._calculate_reference_stroke_width(all_glyphs, upm)

            # Process glyphs in parallel
            processed_glyphs: dict[str, Glyph] = {}
            if glyphs_to_process:
                processed_glyphs = self._process_glyphs_parallel(
                    glyphs=glyphs_to_process,
                    upm=upm,
                    max_workers=max_workers,
                    stats=stats,
                    reference_stroke_width=reference_stroke_width,
                    progress_callback=progress_callback,
                )
            else:
                self.logger.info("No glyphs to process")

            # Save modified font
            self._save_font(
                reader=reader,
                output_path=output_path,
                processed_glyphs=processed_glyphs,
            )

        finally:
            reader.close()

        # Finalize stats
        stats.end_time = time.time()

        self.logger.info(
            "Processing complete",
            processed=stats.processed_count,
            skipped=stats.skipped_count,
            errors=stats.error_count,
            bridges_added=stats.bridges_added,
            duration_seconds=round(stats.duration_seconds, 2),
        )

        return stats

    def _process_glyphs_parallel(
        self,
        glyphs: list[Glyph],
        upm: int,
        max_workers: int | None,
        stats: ProcessingStats,
        reference_stroke_width: float,
        progress_callback: Callable[[int, int, str, bool], None] | None = None,
    ) -> dict[str, Glyph]:
        """Process glyphs in parallel using ProcessPoolExecutor.

        Args:
            glyphs: List of glyphs to process
            upm: Font units per em
            max_workers: Maximum worker processes
            stats: Statistics object to update
            reference_stroke_width: Reference stroke width for consistent bridges
            progress_callback: Optional callback(completed, total, glyph_name, success)
                for progress updates

        Returns:
            Dictionary mapping glyph names to transformed glyphs
        """
        processed_glyphs: dict[str, Glyph] = {}

        # Serialize configuration for workers
        config_dict = self.config.bridge.model_dump()

        # Create tasks
        tasks = {
            glyph.name: glyph.to_dict()
            for glyph in glyphs
        }

        self.logger.info(
            "Starting parallel processing",
            glyph_count=len(tasks),
            max_workers=max_workers,
        )

        # Process in parallel
        total = len(tasks)
        completed = 0
        pending_futures: dict = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            for name, glyph_dict in tasks.items():
                future = executor.submit(
                    process_glyph,
                    glyph_dict,
                    config_dict,
                    upm,
                    reference_stroke_width,
                )
                pending_futures[future] = name

            try:
                # Collect results as they complete
                for future in as_completed(pending_futures):
                    glyph_name = pending_futures.pop(future)
                    success = False

                    try:
                        result = future.result()

                        if "error" in result:
                            # Processing error
                            self.processing_logger.log_glyph_error(
                                glyph_name=result["glyph_name"],
                                error=Exception(result["error"]),
                                traceback=result.get("traceback"),
                            )
                            stats.error_count += 1
                        else:
                            # Success
                            success = True
                            transformed_glyph = Glyph.from_dict(result["glyph"])
                            processed_glyphs[glyph_name] = transformed_glyph

                            bridges_added = result["bridges_added"]
                            stats.processed_count += 1
                            stats.bridges_added += bridges_added

                            duration_ms = result.get("duration_ms", 0.0)
                            self.processing_logger.log_glyph_complete(
                                glyph_name=glyph_name,
                                bridges_added=bridges_added,
                                duration_ms=duration_ms,
                            )
                            stats.glyph_timings_ms.append(duration_ms)

                    except Exception as e:
                        # Executor-level error
                        tb = traceback.format_exc()
                        self.processing_logger.log_glyph_error(
                            glyph_name=glyph_name,
                            error=e,
                            traceback=tb,
                        )
                        stats.error_count += 1

                    # Update progress
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total, glyph_name, success)

            except KeyboardInterrupt:
                # Cancel pending futures
                self.logger.info("Cancellation requested by user")
                cancelled_count = 0
                for f in pending_futures:
                    if f.cancel():
                        cancelled_count += 1

                stats.was_cancelled = True
                stats.cancelled_count = len(pending_futures)

                # Shutdown executor with cancel_futures=True (Python 3.9+)
                executor.shutdown(wait=True, cancel_futures=True)
                raise

        return processed_glyphs

    def _save_font(
        self,
        reader: FontReader,
        output_path: Path,
        processed_glyphs: dict[str, Glyph],
    ) -> None:
        """Save the modified font to output path.

        Args:
            reader: Font reader with loaded font
            output_path: Path to save modified font
            processed_glyphs: Dictionary of processed glyphs to update
        """
        # Get underlying TTFont object
        font = reader._font

        if font is None:
            raise RuntimeError("Font not loaded")

        # Create writer
        writer = FontWriter(font, output_path)

        # Update processed glyphs
        for glyph_name, glyph in processed_glyphs.items():
            try:
                writer.update_glyph(glyph)
            except Exception as e:
                self.logger.error(
                    "Failed to update glyph in font",
                    glyph=glyph_name,
                    error=str(e),
                )

        # Save font
        writer.save()

        self.logger.info(
            "Font saved",
            output=str(output_path),
            updated_glyphs=len(processed_glyphs),
        )
