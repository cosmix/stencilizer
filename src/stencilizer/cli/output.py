"""Rich console output helpers for the CLI.

This module provides user-friendly console output using Rich library
with progress bars, tables, and formatted messages.
"""


from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def create_progress() -> Progress:
    """Create a rich progress bar for glyph processing.

    Returns:
        Configured Progress instance with spinner, bar, and time estimates.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,  # Keep progress visible after completion
    )


def print_header(version: str) -> None:
    """Print application header.

    Args:
        version: Application version string
    """
    console.print(f"\n[bold blue]Stencilizer[/bold blue] v{version}")
    console.print("=" * 40)


def print_step(step: int, total: int, message: str) -> None:
    """Print a processing step indicator.

    Args:
        step: Current step number
        total: Total number of steps
        message: Step description message
    """
    console.print(f"\n[bold cyan][{step}/{total}][/bold cyan] {message}")


def print_font_info(font_path: str, font_type: str, glyph_count: int, upm: int) -> None:
    """Print font information.

    Args:
        font_path: Path to the font file
        font_type: Font format type (e.g., "TrueType", "OpenType")
        glyph_count: Total number of glyphs in font
        upm: Units per em value
    """
    console.print(f"      {font_path} ({font_type})")
    console.print(f"      {glyph_count:,} glyphs, {upm:,} UPM")


def print_islands_found(count: int, glyph_names: list[str], verbose: bool) -> None:
    """Print islands discovery result.

    Args:
        count: Number of glyphs with islands found
        glyph_names: List of glyph names that have islands
        verbose: Whether to show detailed glyph list
    """
    console.print(f"      Found [green]{count}[/green] glyphs with islands")
    if verbose and glyph_names:
        # Format as comma-separated list, wrapped
        names_str = ", ".join(glyph_names[:20])
        if len(glyph_names) > 20:
            names_str += f", ... ({len(glyph_names) - 20} more)"
        console.print(f"      {names_str}")


def print_processing_summary(
    processed: int,
    bridges: int,
    errors: int,
    avg_time_ms: float | None = None,
    min_time_ms: float | None = None,
    max_time_ms: float | None = None,
) -> None:
    """Print processing summary table.

    Args:
        processed: Number of glyphs processed
        bridges: Total number of bridges added
        errors: Number of errors encountered
        avg_time_ms: Average processing time per glyph in milliseconds
        min_time_ms: Minimum processing time per glyph in milliseconds
        max_time_ms: Maximum processing time per glyph in milliseconds
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", style="green")

    table.add_row("Processed:", f"{processed} glyphs")
    table.add_row("Bridges added:", f"{bridges} total")

    # Error count in red if non-zero
    error_style = "red" if errors > 0 else "green"
    table.add_row("Errors:", f"[{error_style}]{errors}[/{error_style}]")

    # Timing stats if available
    if avg_time_ms is not None:
        timing_str = f"{avg_time_ms:.1f}ms avg"
        if min_time_ms is not None and max_time_ms is not None:
            timing_str += f" ({min_time_ms:.1f}-{max_time_ms:.1f}ms)"
        table.add_row("Per-glyph:", timing_str)

    console.print()
    console.print(Panel(table, title="Summary", border_style="blue"))


def print_success(output_path: str, file_size: str) -> None:
    """Print success message.

    Args:
        output_path: Path to output file
        file_size: Human-readable file size string
    """
    console.print("\n[bold green]Done![/bold green] Stencil font created successfully.")
    console.print(f"      Output: [cyan]{output_path}[/cyan] ({file_size})")


def print_error(message: str, details: str | None = None) -> None:
    """Print error message.

    Args:
        message: Main error message
        details: Optional detailed error information
    """
    console.print(f"\n[bold red]ERROR:[/bold red] {message}")
    if details:
        console.print(f"      Details: {details}")


def print_cancellation_notice() -> None:
    """Print cancellation acknowledgment."""
    console.print(
        "\n[yellow]Cancelling...[/yellow] Waiting for in-progress glyphs to complete."
    )


def print_cancellation_summary(processed: int, cancelled: int) -> None:
    """Print cancellation summary.

    Args:
        processed: Number of glyphs successfully processed before cancellation
        cancelled: Number of pending tasks that were cancelled
    """
    console.print("\n[bold yellow]Processing Cancelled[/bold yellow]")
    console.print(f"      Completed: [green]{processed}[/green] glyphs before cancel")
    console.print(f"      Cancelled: [yellow]{cancelled}[/yellow] pending tasks")
    console.print("\n      [dim]No output file created.[/dim]")
