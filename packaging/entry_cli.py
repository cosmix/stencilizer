"""PyInstaller entry point for the stencilizer command-line tool."""

import multiprocessing

from stencilizer.cli.app import main

if __name__ == "__main__":
    # Must run first: frozen worker processes would otherwise re-execute main().
    multiprocessing.freeze_support()
    main()
