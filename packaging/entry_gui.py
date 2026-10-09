"""PyInstaller entry point for the stencilizer desktop GUI."""

import multiprocessing
import sys

from stencilizer.gui.app import main

if __name__ == "__main__":
    # Must run first: frozen worker processes would otherwise re-execute main().
    multiprocessing.freeze_support()
    sys.exit(main())
