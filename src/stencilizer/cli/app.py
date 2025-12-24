"""CLI application entry point for stencilizer.

This module provides the main CLI interface using Typer.
"""

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
from stencilizer.config import (
    BridgeConfig,
    BridgePosition,
    LoggingConfig,
    ProcessingConfig,
    StencilizerSettings,
)
from stencilizer.core import FontProcessor, GlyphAnalyzer
from stencilizer.exceptions import FontLoadError, FontSaveError, StencilizerError
from stencilizer.io import FontReader

# Create the Typer app
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


@app.command()
def stencilize(
    input_font: Annotated[
        Path,
        typer.Argument(
            help="Path to input TTF/OTF font file",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: {name}-Stenciled.{ext})",
        ),
    ] = None,
    bridge_width: Annotated[
        float,
        typer.Option(
            "--bridge-width",
            "-w",
            help="Bridge width as percentage of stroke width (30-110)",
            min=30.0,
            max=110.0,
        ),
    ] = 60.0,
    min_bridges: Annotated[
        int,
        typer.Option(
            "--min-bridges",
            "-n",
            help="Minimum bridges per island",
            min=1,
            max=4,
        ),
    ] = 1,
    position: Annotated[
        str,
        typer.Option(
            "--position",
            "-p",
            help="Bridge position preference (auto|top|bottom|left|right|top_bottom)",
        ),
    ] = "auto",
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-j",
            help="Number of parallel workers (default: auto)",
            min=1,
        ),
    ] = None,
    list_islands: Annotated[
        bool,
        typer.Option(
            "--list-islands",
            help="List all glyphs with islands and exit",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Analyze and show what would be done without modifying the font",
        ),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            help="Write detailed logs to file",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG|INFO|WARNING|ERROR)",
        ),
    ] = "WARNING",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose console output",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Minimal console output",
        ),
    ] = False,
    _version: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Convert a font to a stencil-ready version by adding bridges to enclosed contours.

    Automatically detects enclosed contours (islands) in glyphs like O, A, B, D, P, R, Q,
    4, 6, 8, 9, @, etc. and adds bridges to connect them to outer contours.

    Example:
        stencilizer Roboto-Regular.ttf

    This will create Roboto-Regular-Stenciled.ttf with bridges added to all glyphs
    that have enclosed contours.
    """
    # Validate mutually exclusive options
    if verbose and quiet:
        print_error("Cannot use --verbose and --quiet together")
        raise typer.Exit(code=1)

    # Validate input file exists
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

    # Validate position argument
    try:
        position_pref = BridgePosition(position.lower())
    except ValueError:
        print_error(
            f"Invalid position: {position}",
            details="Valid values: auto, top, bottom, left, right, top_bottom",
        )
        raise typer.Exit(code=1)

    # Print header
    if not quiet:
        print_header(__version__)

    # Create settings from CLI arguments
    settings = StencilizerSettings(
        bridge=BridgeConfig(
            width_percent=bridge_width,
            min_bridges=min_bridges,
            position_preference=position_pref,
        ),
        processing=ProcessingConfig(
            max_workers=workers,
        ),
        logging=LoggingConfig(
            log_file=log_file,
            log_level=log_level if not quiet else "WARNING",
        ),
    )

    try:
        # Handle --list-islands mode
        if list_islands:
            _handle_list_islands(input_font, quiet)
            raise typer.Exit(code=0)

        # Handle --dry-run mode
        if dry_run:
            _handle_dry_run(input_font, settings, quiet, verbose)
            raise typer.Exit(code=0)

        # Load font info
        if not quiet:
            print_step("Loading font")

        try:
            reader = FontReader(input_font)
            reader.load()
            font_type = reader.format
            glyph_count = reader.glyph_count
            upm = reader.units_per_em
            reader.close()
        except Exception as e:
            raise FontLoadError(str(input_font), str(e)) from e

        if not quiet:
            print_font_info(
                font_path=str(input_font),
                font_type=font_type,
                glyph_count=glyph_count,
                upm=upm,
            )

        # Analyze glyphs
        if not quiet:
            print_step("Analyzing glyphs")

        island_glyphs = []
        try:
            reader = FontReader(input_font)
            reader.load()
            analyzer = GlyphAnalyzer()

            for glyph in reader.iter_glyphs():
                if glyph.is_empty() or glyph.is_composite():
                    continue
                hierarchy = analyzer.analyze(glyph)
                if hierarchy.has_islands():
                    island_glyphs.append(glyph.name)

            reader.close()
        except Exception as e:
            raise FontLoadError(str(input_font), str(e)) from e

        if not quiet:
            print_islands_found(
                count=len(island_glyphs),
                glyph_names=island_glyphs,
                verbose=verbose,
            )

        if len(island_glyphs) == 0:
            if not quiet:
                console.print("\nNo glyphs with islands found. Nothing to process.")
            raise typer.Exit(code=0)

        # Process font
        if not quiet:
            import os

            actual_workers = workers if workers else os.cpu_count() or 1
            print_step("Processing")
            print_processing_info(actual_workers, is_auto=(workers is None))

        # Determine output path early for cancellation handling
        if output is None:
            from stencilizer.io import FontWriter

            actual_output_path = FontWriter.get_stenciled_path(input_font)
        else:
            actual_output_path = output

        processor = FontProcessor(settings)
        stats = None

        try:
            if not quiet:
                with create_progress() as progress:
                    task_id = progress.add_task(
                        f"Processing {len(island_glyphs)} glyphs",
                        total=len(island_glyphs),
                    )

                    def update_progress(
                        completed: int, *_: object
                    ) -> None:
                        progress.update(task_id, completed=completed)

                    stats = processor.process(
                        font_path=input_font,
                        output_path=actual_output_path,
                        max_workers=workers,
                        progress_callback=update_progress,
                    )
            else:
                stats = processor.process(
                    font_path=input_font,
                    output_path=actual_output_path,
                    max_workers=workers,
                )
        except KeyboardInterrupt:
            if not quiet:
                print_cancellation_notice()
                print_cancellation_summary(
                    processed=stats.processed_count if stats else 0,
                    cancelled=stats.cancelled_count if stats else 0,
                )
            raise typer.Exit(code=130) from None  # Standard Unix SIGINT exit code

        # Get file size and print success
        file_size = _format_file_size(actual_output_path)

        if not quiet:
            print_success(
                output_path=str(actual_output_path),
                file_size=file_size,
                total_time_s=stats.duration_seconds,
                processed=stats.processed_count,
                bridges=stats.bridges_added,
                errors=stats.error_count,
                avg_time_ms=stats.avg_glyph_time_ms,
                min_time_ms=stats.min_glyph_time_ms,
                max_time_ms=stats.max_glyph_time_ms,
            )

    except FontLoadError as e:
        print_error(f"Could not load font: {e.reason}")
        raise typer.Exit(code=1)
    except FontSaveError as e:
        print_error(f"Could not save font: {e.reason}")
        raise typer.Exit(code=1)
    except StencilizerError as e:
        print_error(str(e))
        raise typer.Exit(code=1)
    except typer.Exit:
        # Re-raise typer.Exit to allow clean exits
        raise
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(code=1)


def _handle_list_islands(font_path: Path, quiet: bool) -> None:
    """Handle --list-islands mode.

    Args:
        font_path: Path to font file
        quiet: Suppress output
    """
    if not quiet:
        print_step("Loading font")

    try:
        reader = FontReader(font_path)
        reader.load()

        if not quiet:
            print_font_info(
                font_path=str(font_path),
                font_type=reader.format,
                glyph_count=reader.glyph_count,
                upm=reader.units_per_em,
            )
            print_step("Scanning for islands")

        analyzer = GlyphAnalyzer()
        island_glyphs = []

        for glyph in reader.iter_glyphs():
            if glyph.is_empty() or glyph.is_composite():
                continue
            hierarchy = analyzer.analyze(glyph)
            if hierarchy.has_islands():
                island_count = len(hierarchy.get_islands())
                island_glyphs.append((glyph.name, island_count))

        reader.close()

        if not quiet:
            console.print(f"\n[bold]{len(island_glyphs)} glyphs with islands[/bold]\n")

        for glyph_name, island_count in island_glyphs:
            plural = "island" if island_count == 1 else "islands"
            console.print(f"  {glyph_name}: {island_count} {plural}")

    except Exception as e:
        print_error(f"Could not analyze font: {e}")
        raise typer.Exit(code=1)


def _handle_dry_run(
    font_path: Path, settings: StencilizerSettings, quiet: bool, verbose: bool
) -> None:
    """Handle --dry-run mode.

    Args:
        font_path: Path to font file
        settings: Stencilizer settings
        quiet: Suppress output
        verbose: Show verbose output
    """
    if not quiet:
        print_step("Loading font")

    try:
        reader = FontReader(font_path)
        reader.load()

        if not quiet:
            print_font_info(
                font_path=str(font_path),
                font_type=reader.format,
                glyph_count=reader.glyph_count,
                upm=reader.units_per_em,
            )
            print_step("Analyzing (dry run)")

        analyzer = GlyphAnalyzer()
        island_glyphs = []
        total_islands = 0

        for glyph in reader.iter_glyphs():
            if glyph.is_empty() or glyph.is_composite():
                continue
            hierarchy = analyzer.analyze(glyph)
            if hierarchy.has_islands():
                island_count = len(hierarchy.get_islands())
                island_glyphs.append((glyph.name, island_count))
                total_islands += island_count

        reader.close()

        if not quiet:
            console.print("\n[bold]Analysis[/bold]\n")
            console.print(f"  Glyphs with islands   {len(island_glyphs)}")
            console.print(f"  Total islands         {total_islands}")
            console.print(f"  Estimated bridges     {total_islands}")
            console.print(f"  Bridge width          {settings.bridge.width_percent}% of stroke")

            if verbose and island_glyphs:
                console.print("\n[bold]Glyphs[/bold]")
                for glyph_name, island_count in island_glyphs[:20]:
                    plural = "island" if island_count == 1 else "islands"
                    console.print(f"  {glyph_name}: {island_count} {plural}")
                if len(island_glyphs) > 20:
                    console.print(f"  ... +{len(island_glyphs) - 20} more")

            console.print(f"\n[bold green]{SYM_OK} Dry run complete[/bold green] – no changes made")

    except Exception as e:
        print_error(f"Could not analyze font: {e}")
        raise typer.Exit(code=1)


def _format_file_size(path: Path) -> str:
    """Format file size in human-readable form.

    Args:
        path: Path to file

    Returns:
        Human-readable file size (e.g., "428 KB")
    """
    try:
        size_bytes = path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.0f} KB"
        else:
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
