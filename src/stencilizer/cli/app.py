"""CLI application entry point for stencilizer."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from stencilizer import __version__
from stencilizer.cli.handlers import (
    _handle_dry_run,
    _handle_list_islands,
    finish_run,
    resolve_width_scaling,
    stencil_pinned,
    validate_input,
)
from stencilizer.cli.output import (
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
    variable_axes,
)
from stencilizer.cli.pinning import pinned_input, validate_instance
from stencilizer.config import (
    BridgeConfig,
    BridgeWidthScaling,
    LoggingConfig,
    ProcessingConfig,
    StencilizerSettings,
)
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification
from stencilizer.exceptions import (
    FontLoadError,
    FontProcessingError,
    FontSaveError,
    StencilizerError,
)
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
WidthScalingOption = Annotated[
    BridgeWidthScaling,
    typer.Option(
        "--width-scaling",
        help="Variable fonts: keep bridge gaps fixed or scale them with each master's stroke weight",
    ),
]
ScalingStrengthOption = Annotated[
    float,
    typer.Option(
        "--scaling-strength",
        help="Proportional width scaling: 0 (fixed) to 100 (fully proportional)",
        min=0.0,
        max=100.0,
    ),
]
MinBridgeWidthOption = Annotated[
    float,
    typer.Option(
        "--min-bridge-width",
        help="Proportional width scaling: smallest gap as percent of a reference stroke (10-110)",
        min=10.0,
        max=110.0,
    ),
]
InstanceOption = Annotated[
    str | None,
    typer.Option(
        "--instance", help="Pin a variable font to a static instance, e.g. wght=700,wdth=90"
    ),
]
OutputOption = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="Output path (default: {name}-Stenciled.{ext})"),
]
WorkersOption = Annotated[
    int | None,
    typer.Option("--workers", "-j", help="Number of parallel workers (default: auto)", min=1),
]
DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run", help="Analyze and show what would be done without modifying the font"
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
    output: OutputOption = None,
    instance: InstanceOption = None,
    bridge_width: BridgeWidthOption = 60.0,
    width_scaling: WidthScalingOption = BridgeWidthScaling.FIXED,
    scaling_strength: ScalingStrengthOption = 100.0,
    min_bridge_width: MinBridgeWidthOption = 30.0,
    workers: WorkersOption = None,
    list_islands: Annotated[
        bool, typer.Option("--list-islands", help="List all glyphs with islands and exit")
    ] = False,
    dry_run: DryRunOption = False,
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
    bridge = BridgeConfig(
        width_percent=bridge_width,
        width_scaling=width_scaling,
        scaling_strength=scaling_strength,
        min_width_percent=min_bridge_width,
    )
    _run_command(
        input_font,
        output,
        instance,
        bridge,
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
    instance: str | None,
    bridge: BridgeConfig,
    workers: int | None,
    list_islands: bool,
    dry_run: bool,
    log_file: Path | None,
    log_level: str,
    verbose: bool,
    quiet: bool,
) -> None:
    validate_input(input_font, verbose, quiet)
    if not quiet:
        print_header(__version__)
    settings = StencilizerSettings(
        bridge=bridge,
        processing=ProcessingConfig(max_workers=workers),
        logging=LoggingConfig(log_file=log_file, log_level=log_level if not quiet else "WARNING"),
    )
    try:
        settings = resolve_width_scaling(settings, input_font, instance, quiet)
        output_path = output or FontWriter.get_stenciled_path(input_font)
        if instance and not (list_islands or dry_run) and _is_proportional(settings):
            _run_stencil_first(input_font, output_path, instance, settings, workers, quiet, verbose)
            return
        with pinned_input(input_font, instance) as font_path:
            if list_islands:
                _handle_list_islands(font_path, input_font, quiet)
                raise typer.Exit(code=0)
            if dry_run:
                _handle_dry_run(font_path, input_font, settings, quiet, verbose)
                raise typer.Exit(code=0)
            _run_standard(font_path, input_font, output_path, settings, workers, quiet, verbose)
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


def _is_proportional(settings: StencilizerSettings) -> bool:
    return settings.bridge.width_scaling is BridgeWidthScaling.PROPORTIONAL


def _classify_font(
    font_path: Path, processor: FontProcessor, quiet: bool, display_path: Path | None = None
) -> GlyphClassification:
    # With --instance, font_path is a temporary file; messages name the font the user gave.
    shown = str(display_path or font_path)
    if not quiet:
        print_step("Loading font")
    try:
        with FontReader(font_path) as reader:
            if not quiet:
                print_font_info(
                    font_path=shown,
                    font_type=reader.format,
                    glyph_count=reader.glyph_count,
                    upm=reader.units_per_em,
                    axes=variable_axes(reader.font),
                )
                print_step("Analyzing glyphs")
            return processor.classify_glyphs(reader)
    except StencilizerError:
        raise
    except Exception as error:
        raise FontLoadError(shown, str(error)) from error


def _stencil(
    font_path: Path,
    display_path: Path,
    output_path: Path,
    settings: StencilizerSettings,
    workers: int | None,
    quiet: bool,
    verbose: bool,
) -> ProcessingStats:
    processor = FontProcessor(settings, quiet=quiet)
    classification = _classify_font(font_path, processor, quiet, display_path)
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
    stats = _process_font(processor, font_path, output_path, workers, quiet, classification)
    if stats.error_count:
        raise FontProcessingError(stats.errors)
    return stats


def _run_standard(
    font_path: Path,
    display_path: Path,
    output_path: Path,
    settings: StencilizerSettings,
    workers: int | None,
    quiet: bool,
    verbose: bool,
) -> None:
    stats = _stencil(font_path, display_path, output_path, settings, workers, quiet, verbose)
    finish_run(output_path, stats, quiet)


def _run_stencil_first(
    input_font: Path,
    output_path: Path,
    instance: str,
    settings: StencilizerSettings,
    workers: int | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Stencil the variable font, pin it at ``instance``, then stencil the static result."""
    validate_instance(input_font, instance)
    with tempfile.TemporaryDirectory(prefix="stencilizer-instance-") as tmp:
        stenciled = Path(tmp) / f"{input_font.stem}-variable{input_font.suffix}"
        first = _stencil(input_font, input_font, stenciled, settings, workers, quiet, verbose)
        second = stencil_pinned(
            stenciled, input_font, instance, Path(tmp), output_path, settings, workers
        )
    stats = replace(
        first,
        bridges_added=first.bridges_added + second.bridges_added,
        unbridged_count=second.unbridged_count,
    )
    finish_run(output_path, stats, quiet)


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


def cli() -> None:
    """Entry point for the CLI application."""
    app()


def main() -> None:
    """Entry point for the CLI application (alias for cli)."""
    cli()


if __name__ == "__main__":
    cli()
