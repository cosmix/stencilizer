"""CLI handlers for read-only font analysis commands."""

from pathlib import Path

import typer

from stencilizer.cli.output import (
    SYM_OK,
    console,
    format_file_size,
    print_error,
    print_font_info,
    print_glyph_islands,
    print_step,
    print_success,
    variable_axes,
)
from stencilizer.cli.pinning import is_cff2_font, is_variable_font, pin_stenciled, publish_pinned
from stencilizer.config import BridgeWidthScaling, StencilizerSettings
from stencilizer.core import FontProcessor, GlyphAnalyzer
from stencilizer.exceptions import FontProcessingError
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats
from stencilizer.variable.reader import is_variable

STATIC_WARNING = "Width scaling applies only to variable fonts; using fixed width."
CFF2_WARNING = (
    "Proportional width scaling with --instance is not supported for CFF2 fonts; "
    "pinning first with fixed width."
)


def fixed_width_settings(settings: StencilizerSettings) -> StencilizerSettings:
    """Copy of ``settings`` with the bridge width scaling switched to fixed."""
    bridge = settings.bridge.model_copy(update={"width_scaling": BridgeWidthScaling.FIXED})
    return settings.model_copy(update={"bridge": bridge})


def resolve_width_scaling(
    settings: StencilizerSettings, input_font: Path, instance: str | None, quiet: bool
) -> StencilizerSettings:
    """Fall back to fixed width, with a warning, where proportional scaling cannot apply.

    Reads ``input_font``, never the pinned temporary file, so a static ``--instance`` run still
    fails with the usual "--instance requires a variable font".
    """
    if settings.bridge.width_scaling is not BridgeWidthScaling.PROPORTIONAL:
        return settings
    if not is_variable_font(input_font):
        message = STATIC_WARNING
    elif instance and is_cff2_font(input_font):
        message = CFF2_WARNING
    else:
        return settings
    if not quiet:
        console.print(f"[yellow]{message}[/yellow]")
    return fixed_width_settings(settings)


def stencil_pinned(
    stenciled: Path,
    source: Path,
    instance: str,
    workdir: Path,
    output_path: Path,
    settings: StencilizerSettings,
    workers: int | None,
) -> ProcessingStats:
    """Pin a stenciled variable font at ``instance`` and stencil that static font."""
    try:
        pinned = pin_stenciled(stenciled, source, instance, workdir)
        processor = FontProcessor(fixed_width_settings(settings), quiet=True)
        with FontReader(pinned) as reader:
            classification = processor.classify_glyphs(reader)
        if not classification.glyphs_to_process:
            publish_pinned(pinned, output_path)
            return ProcessingStats()
        stats = processor.process(
            font_path=pinned,
            output_path=output_path,
            max_workers=workers,
            classification=classification,
        )
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    if stats.error_count:
        raise FontProcessingError(stats.errors)
    return stats


def _handle_list_islands(font_path: Path, display_path: Path, quiet: bool) -> None:
    """List each glyph with islands; ``display_path`` is the font as the user named it."""
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=str(display_path),
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
                    axes=variable_axes(reader.font),
                )
                print_step("Scanning for islands")
            island_glyphs = _scan_islands(reader)
        if not quiet:
            console.print(f"\n[bold]{len(island_glyphs)} glyphs with islands[/bold]\n")
        for glyph_name, island_count in island_glyphs:
            print_glyph_islands(glyph_name, island_count)
    except Exception as error:
        print_error(f"Could not analyze font: {error}")
        raise typer.Exit(code=1) from error


def _scan_islands(reader: FontReader) -> list[tuple[str, int]]:
    if is_variable(reader.font):
        from stencilizer.variable.processing import variable_island_counts

        return variable_island_counts(reader)
    analyzer = GlyphAnalyzer()
    island_glyphs: list[tuple[str, int]] = []
    for glyph in reader.iter_glyphs():
        if glyph.is_empty() or glyph.is_composite():
            continue
        hierarchy = analyzer.analyze(glyph)
        if hierarchy.has_islands():
            island_glyphs.append((glyph.name, len(hierarchy.get_islands())))
    return island_glyphs


def _handle_dry_run(
    font_path: Path,
    display_path: Path,
    settings: StencilizerSettings,
    quiet: bool,
    verbose: bool,
) -> None:
    """Analyze a font without changing it; ``display_path`` is the font as the user named it."""
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=str(display_path),
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
                    axes=variable_axes(reader.font),
                )
                print_step("Analyzing (dry run)")
            island_glyphs = _scan_islands(reader)
        if not quiet:
            _report_dry_run(island_glyphs, settings, verbose)
    except Exception as error:
        print_error(f"Could not analyze font: {error}")
        raise typer.Exit(code=1) from error


def _report_dry_run(
    island_glyphs: list[tuple[str, int]], settings: StencilizerSettings, verbose: bool
) -> None:
    total_islands = sum(count for _, count in island_glyphs)
    console.print("\n[bold]Analysis[/bold]\n")
    console.print(f"  Glyphs with islands   {len(island_glyphs)}")
    console.print(f"  Total islands         {total_islands}")
    console.print(f"  Estimated bridges     {total_islands}")
    console.print(
        f"  Bridge width          {settings.bridge.width_percent}% of a reference stroke "
        "of 10% of font UPM"
    )
    bridge = settings.bridge
    if bridge.width_scaling is BridgeWidthScaling.PROPORTIONAL:
        console.print(
            f"  Width scaling         proportional (strength {bridge.scaling_strength}%, "
            f"minimum {bridge.min_width_percent}% of a reference stroke)"
        )
    else:
        console.print("  Width scaling         fixed")
    if verbose and island_glyphs:
        console.print("\n[bold]Glyphs[/bold]")
        for glyph_name, island_count in island_glyphs[:20]:
            print_glyph_islands(glyph_name, island_count)
        if len(island_glyphs) > 20:
            console.print(f"  ... +{len(island_glyphs) - 20} more")
    console.print(f"\n[bold green]{SYM_OK} Dry run complete[/bold green] – no changes made")  # noqa: RUF001


def validate_input(input_font: Path, verbose: bool, quiet: bool) -> None:
    if verbose and quiet:
        print_error("Cannot use --verbose and --quiet together")
        raise typer.Exit(code=1)
    if not input_font.exists():
        print_error(
            f"Input file not found: {input_font}",
            details=f"The file '{input_font}' does not exist or is not accessible.",
        )
        raise typer.Exit(code=1)
    if not input_font.is_file():
        print_error(
            f"Input path is not a file: {input_font}",
            details="Please provide a path to a TTF or OTF font file.",
        )
        raise typer.Exit(code=1)


def finish_run(output_path: Path, stats: ProcessingStats, quiet: bool) -> None:
    if quiet and stats.unbridged_count:
        noun = "island" if stats.unbridged_count == 1 else "islands"
        console.print(
            f"[yellow]Warning: {stats.unbridged_count} {noun} remained unbridged[/yellow]"
        )
    if not quiet:
        _report_success(output_path, stats)


def _report_success(output_path: Path, stats: ProcessingStats) -> None:
    print_success(
        output_path=str(output_path),
        file_size=format_file_size(output_path),
        total_time_s=stats.duration_seconds,
        processed=stats.processed_count,
        bridges=stats.bridges_added,
        unbridged=stats.unbridged_count,
        errors=stats.error_count,
        avg_time_ms=stats.avg_glyph_time_ms,
        min_time_ms=stats.min_glyph_time_ms,
        max_time_ms=stats.max_glyph_time_ms,
    )
