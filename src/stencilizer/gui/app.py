"""Launch the stencilizer desktop application."""

import argparse
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "stencilizer-gui needs the 'gui' extra "
        "(from a source checkout: uv pip install -e '.[gui]') "
        f"({error})"
    ) from error

from stencilizer.gui.controller import GuiController
from stencilizer.gui.main_window import MainWindow
from stencilizer.gui.theme import apply_theme


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the desktop application."""
    parser = argparse.ArgumentParser(
        prog="stencilizer-gui",
        description="Open and stencilize TrueType or OpenType fonts.",
    )
    parser.add_argument(
        "font",
        type=Path,
        nargs="?",
        help="Font file (TTF/OTF) to open on launch",
    )
    return parser


def default_log_file() -> Path:
    """Create a private (0600), randomly named log file for this run in the temp dir."""
    descriptor, name = tempfile.mkstemp(prefix="stencilizer-gui-", suffix=".log")
    os.close(descriptor)
    return Path(name)


def create_window(font: Path | None, log_file: Path) -> MainWindow:
    """Create the main window and optionally begin loading a font."""
    window = MainWindow(GuiController(log_file))
    if font is not None:
        window.load_font(font)
    return window


def main(argv: list[str] | None = None) -> int:
    """Run the desktop application and return Qt's process exit code."""
    args = build_parser().parse_args(argv)
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    application = QApplication(sys.argv[:1])
    apply_theme(application)
    window = create_window(args.font, default_log_file())
    window.show()
    return application.exec()
