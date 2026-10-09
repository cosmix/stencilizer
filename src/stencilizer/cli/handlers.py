"""CLI handlers for read-only font analysis commands."""

from pathlib import Path

import typer

from stencilizer.cli.output import (
    SYM_OK,
    console,
    print_error,
    print_font_info,
    print_step,
    variable_axes,
)
from stencilizer.config import StencilizerSettings
from stencilizer.core import GlyphAnalyzer
from stencilizer.io import FontReader
from stencilizer.variable.reader import is_variable


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
                    axes=variable_axes(reader.font),
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
    if verbose and island_glyphs:
        console.print("\n[bold]Glyphs[/bold]")
        for glyph_name, island_count in island_glyphs[:20]:
            plural = "island" if island_count == 1 else "islands"
            console.print(f"  {glyph_name}: {island_count} {plural}")
        if len(island_glyphs) > 20:
            console.print(f"  ... +{len(island_glyphs) - 20} more")
    console.print(f"\n[bold green]{SYM_OK} Dry run complete[/bold green] – no changes made")  # noqa: RUF001
