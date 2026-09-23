"""CLI application entry point for stencilizer."""

import os
from pathlib import Path
from typing import Annotated

import typer

from stencilizer import __version__
from stencilizer.cli.output import (
    SYM_OK,
    console,
    create_progress,
    print_cancellation_notice,
    print_cancellation_summary,
    print_error,
    print_font_info,
    print_header,
    print_islands_found,
    print_processing_info,
    print_step,
    print_success,
)
from stencilizer.config import BridgeConfig, LoggingConfig, ProcessingConfig, StencilizerSettings
from stencilizer.core import FontProcessor, GlyphAnalyzer
from stencilizer.core.processor import GlyphClassification
from stencilizer.exceptions import FontLoadError, FontSaveError, StencilizerError
from stencilizer.io import FontReader, FontWriter
from stencilizer.utils import ProcessingStats

app = typer.Typer(
    name="stencilizer",
    help="Convert fonts to stencil-ready versions by adding bridges to enclosed contours.",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Stencilizer[/bold blue] v{__version__}")
        raise typer.Exit()


BridgeWidthOption = Annotated[
    float,
    typer.Option(
        "--bridge-width",
        "-w",
        help="Bridge width as percent of a reference stroke of 10% of font UPM (30-110)",
        min=30.0,
        max=110.0,
    ),
]
VersionOption = Annotated[
    bool | None,
    typer.Option(
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
]


@app.command()
def stencilize(
    input_font: Annotated[
        Path, typer.Argument(help="Path to input TTF/OTF font file", show_default=False)
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path (default: {name}-Stenciled.{ext})"),
    ] = None,
    bridge_width: BridgeWidthOption = 60.0,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-j", help="Number of parallel workers (default: auto)", min=1),
    ] = None,
    list_islands: Annotated[
        bool, typer.Option("--list-islands", help="List all glyphs with islands and exit")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Analyze and show what would be done without modifying the font"
        ),
    ] = False,
    log_file: Annotated[
        Path | None, typer.Option("--log-file", help="Write detailed logs to file")
    ] = None,
    log_level: Annotated[
        str, typer.Option("--log-level", help="Logging level (DEBUG|INFO|WARNING|ERROR)")
    ] = "WARNING",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose console output")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Minimal console output")] = False,
    _version: VersionOption = None,
) -> None:
    """Convert a font to a stencil-ready version by bridging enclosed contours."""
    _run_command(
        input_font,
        output,
        bridge_width,
        workers,
        list_islands,
        dry_run,
        log_file,
        log_level,
        verbose,
        quiet,
    )


def _run_command(
    input_font: Path,
    output: Path | None,
    bridge_width: float,
    workers: int | None,
    list_islands: bool,
    dry_run: bool,
    log_file: Path | None,
    log_level: str,
    verbose: bool,
    quiet: bool,
) -> None:
    _validate_input(input_font, verbose, quiet)
    if not quiet:
        print_header(__version__)
    settings = StencilizerSettings(
        bridge=BridgeConfig(width_percent=bridge_width),
        processing=ProcessingConfig(max_workers=workers),
        logging=LoggingConfig(log_file=log_file, log_level=log_level if not quiet else "WARNING"),
    )
    try:
        if list_islands:
            _handle_list_islands(input_font, quiet)
            raise typer.Exit(code=0)
        if dry_run:
            _handle_dry_run(input_font, settings, quiet, verbose)
            raise typer.Exit(code=0)
        _run_standard(input_font, output, settings, workers, quiet, verbose)
    except FontLoadError as error:
        print_error(f"Could not load font: {error.reason}")
        raise typer.Exit(code=1) from error
    except FontSaveError as error:
        print_error(f"Could not save font: {error.reason}")
        raise typer.Exit(code=1) from error
    except StencilizerError as error:
        print_error(str(error))
        raise typer.Exit(code=1) from error
    except typer.Exit:
        raise
    except Exception as error:
        print_error(f"Unexpected error: {error}")
        raise typer.Exit(code=1) from error


def _validate_input(input_font: Path, verbose: bool, quiet: bool) -> None:
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


def _classify_font(font_path: Path, processor: FontProcessor, quiet: bool) -> GlyphClassification:
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=str(font_path),
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
                )
                print_step("Analyzing glyphs")
            return processor.classify_glyphs(reader)
    except Exception as error:
        raise FontLoadError(str(font_path), str(error)) from error


def _run_standard(
    font_path: Path,
    output: Path | None,
    settings: StencilizerSettings,
    workers: int | None,
    quiet: bool,
    verbose: bool,
) -> None:
    processor = FontProcessor(settings)
    classification = _classify_font(font_path, processor, quiet)
    island_names = [glyph.name for glyph in classification.glyphs_to_process]
    if not quiet:
        print_islands_found(len(island_names), island_names, verbose)
    if not island_names:
        if not quiet:
            console.print("\nNo glyphs with islands found. Nothing to process.")
        raise typer.Exit(code=0)
    if not quiet:
        actual_workers = workers if workers else os.cpu_count() or 1
        print_step("Processing")
        print_processing_info(actual_workers, is_auto=(workers is None))
    output_path = output if output is not None else FontWriter.get_stenciled_path(font_path)
    stats = _process_font(processor, font_path, output_path, workers, quiet, classification)
    if not quiet:
        _report_success(output_path, stats)


def _process_font(
    processor: FontProcessor,
    font_path: Path,
    output_path: Path,
    workers: int | None,
    quiet: bool,
    classification: GlyphClassification,
) -> ProcessingStats:
    stats: ProcessingStats | None = None
    try:
        if quiet:
            return processor.process(
                font_path=font_path,
                output_path=output_path,
                max_workers=workers,
                classification=classification,
            )
        with create_progress() as progress:
            task_id = progress.add_task(
                f"Processing {len(classification.glyphs_to_process)} glyphs",
                total=len(classification.glyphs_to_process),
            )

            def update_progress(completed: int, *_: object) -> None:
                progress.update(task_id, completed=completed)

            stats = processor.process(
                font_path=font_path,
                output_path=output_path,
                max_workers=workers,
                progress_callback=update_progress,
                classification=classification,
            )
            return stats
    except KeyboardInterrupt:
        if not quiet:
            print_cancellation_notice()
            print_cancellation_summary(
                processed=stats.processed_count if stats else 0,
                cancelled=stats.cancelled_count if stats else 0,
            )
        raise typer.Exit(code=130) from None


def _report_success(output_path: Path, stats: ProcessingStats) -> None:
    print_success(
        output_path=str(output_path),
        file_size=_format_file_size(output_path),
        total_time_s=stats.duration_seconds,
        processed=stats.processed_count,
        bridges=stats.bridges_added,
        errors=stats.error_count,
        avg_time_ms=stats.avg_glyph_time_ms,
        min_time_ms=stats.min_glyph_time_ms,
        max_time_ms=stats.max_glyph_time_ms,
    )


def _handle_list_islands(font_path: Path, quiet: bool) -> None:
    """List each glyph with islands."""
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=str(font_path),
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
                )
                print_step("Scanning for islands")
            island_glyphs = _scan_islands(reader)
        if not quiet:
            console.print(f"\n[bold]{len(island_glyphs)} glyphs with islands[/bold]\n")
        for glyph_name, island_count in island_glyphs:
            plural = "island" if island_count == 1 else "islands"
            console.print(f"  {glyph_name}: {island_count} {plural}")
    except Exception as error:
        print_error(f"Could not analyze font: {error}")
        raise typer.Exit(code=1) from error


def _scan_islands(reader: FontReader) -> list[tuple[str, int]]:
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
    font_path: Path, settings: StencilizerSettings, quiet: bool, verbose: bool
) -> None:
    """Analyze a font without changing it."""
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=str(font_path),
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
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
    if verbose and island_glyphs:
        console.print("\n[bold]Glyphs[/bold]")
        for glyph_name, island_count in island_glyphs[:20]:
            plural = "island" if island_count == 1 else "islands"
            console.print(f"  {glyph_name}: {island_count} {plural}")
        if len(island_glyphs) > 20:
            console.print(f"  ... +{len(island_glyphs) - 20} more")
    console.print(f"\n[bold green]{SYM_OK} Dry run complete[/bold green] – no changes made")  # noqa: RUF001


def _format_file_size(path: Path) -> str:
    """Format file size in human-readable form."""
    try:
        size_bytes = path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.0f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        return "unknown"


def cli() -> None:
    """Entry point for the CLI application."""
    app()


def main() -> None:
    """Entry point for the CLI application (alias for cli)."""
    cli()


if __name__ == "__main__":
    cli()
