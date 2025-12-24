"""Rich console output helpers for the CLI.

This module provides user-friendly console output using Rich library
with progress bars, tables, and formatted messages.
"""


from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

console = Console()

# Unicode symbols for consistent visual language
SYM_STEP = "▸"  # Step indicator
SYM_OK = "✓"  # Success
SYM_ERR = "✗"  # Error
SYM_DOT = "·"  # Separator/secondary info


def create_progress() -> Progress:
    """Create a rich progress bar for glyph processing.

    Returns:
        Configured Progress instance with bar and time elapsed.
    """
    return Progress(
        TextColumn("  "),
        BarColumn(bar_width=40, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_header(version: str) -> None:
    """Print application header.

    Args:
        version: Application version string
    """
    console.print(f"\n[bold]Stencilizer[/bold] v{version}")
    console.print("─" * 44)


def print_step(message: str) -> None:
    """Print a processing step indicator.

    Args:
        message: Step description message
    """
    console.print(f"\n{SYM_STEP} {message}")


def print_font_info(font_path: str, font_type: str, glyph_count: int, upm: int) -> None:
    """Print font information.

    Args:
        font_path: Path to the font file
        font_type: Font format type (e.g., "TrueType", "OpenType")
        glyph_count: Total number of glyphs in font
        upm: Units per em value
    """
    # Use Text to safely handle paths with special characters
    line1 = Text("  ")
    line1.append(font_path)
    line1.append(f" ({font_type})")
    console.print(line1)
    console.print(f"  {glyph_count:,} glyphs {SYM_DOT} {upm:,} UPM")


def print_islands_found(count: int, glyph_names: list[str], verbose: bool) -> None:
    """Print islands discovery result.

    Args:
        count: Number of glyphs with islands found
        glyph_names: List of glyph names that have islands
        verbose: Whether to show detailed glyph list
    """
    console.print(f"  [green]{count}[/green] glyphs with islands")
    if verbose and glyph_names:
        names_str = ", ".join(glyph_names[:20])
        if len(glyph_names) > 20:
            names_str += f" {SYM_DOT}{SYM_DOT}{SYM_DOT} (+{len(glyph_names) - 20} more)"
        console.print(f"  {names_str}")


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def print_processing_info(workers: int, is_auto: bool = False) -> None:
    """Print processing configuration.

    Args:
        workers: Number of parallel workers
        is_auto: Whether the count was auto-detected
    """
    auto_suffix = " (auto)" if is_auto else ""
    console.print(f"  {workers} workers{auto_suffix} {SYM_DOT} Ctrl+C to cancel")


def print_success(
    output_path: str,
    file_size: str,
    total_time_s: float,
    processed: int,
    bridges: int,
    errors: int,
    avg_time_ms: float | None = None,
    min_time_ms: float | None = None,
    max_time_ms: float | None = None,
) -> None:
    """Print success message with summary.

    Args:
        output_path: Path to output file
        file_size: Human-readable file size string
        total_time_s: Total processing time in seconds
        processed: Number of glyphs processed
        bridges: Total number of bridges added
        errors: Number of errors encountered
        avg_time_ms: Average processing time per glyph in milliseconds
        min_time_ms: Minimum processing time per glyph in milliseconds
        max_time_ms: Maximum processing time per glyph in milliseconds
    """
    time_str = _format_time(total_time_s)

    # Success header
    console.print(f"\n[bold green]{SYM_OK} Complete[/bold green] in {time_str}")

    # Output file info
    line = Text("  ")
    line.append(output_path, style="bold")
    line.append(f" ({file_size})")
    console.print(line)

    # Stats line
    error_style = "red" if errors > 0 else "green"
    console.print(
        f"  {processed} glyphs {SYM_DOT} {bridges} bridges {SYM_DOT} "
        f"[{error_style}]{errors} errors[/{error_style}]"
    )

    # Per-glyph timing
    if avg_time_ms is not None:
        timing_str = f"{avg_time_ms:.1f}ms avg"
        if min_time_ms is not None and max_time_ms is not None:
            timing_str += f" ({min_time_ms:.1f}–{max_time_ms:.1f}ms range)"
        console.print(f"  {timing_str}")


def print_error(message: str, details: str | None = None) -> None:
    """Print error message.

    Args:
        message: Main error message
        details: Optional detailed error information
    """
    console.print(f"\n[bold red]{SYM_ERR} Error:[/bold red] {message}")
    if details:
        console.print(f"  {details}")


def print_cancellation_notice() -> None:
    """Print cancellation acknowledgment."""
    console.print(f"\n{SYM_DOT} Cancelling... waiting for in-progress glyphs")


def print_cancellation_summary(processed: int, cancelled: int) -> None:
    """Print cancellation summary.

    Args:
        processed: Number of glyphs successfully processed before cancellation
        cancelled: Number of pending tasks that were cancelled
    """
    console.print(f"\n{SYM_DOT} [bold]Cancelled[/bold]")
    console.print(f"  {processed} glyphs completed {SYM_DOT} {cancelled} tasks cancelled")
    console.print("  No output file created")
