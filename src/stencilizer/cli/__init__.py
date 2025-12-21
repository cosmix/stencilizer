"""Command-line interface for stencilizer.

This module provides the CLI using Typer with rich output for
user-friendly feedback and progress reporting.

Key features:
- Progress bars for glyph processing
- Verbose/quiet output modes
- Preview mode for debugging
- Detailed error reporting
"""

from stencilizer.cli.app import cli, main

__all__ = ["cli", "main"]
