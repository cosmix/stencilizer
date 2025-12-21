"""Stencilizer - Convert fonts to stencil-ready versions.

Stencilizer is a CLI tool that converts TrueType/OpenType fonts into stencil-ready
versions by automatically detecting enclosed contours (islands) in glyphs and adding
bridges to connect them to outer contours.

Example:
    $ stencilizer Roboto-Regular.ttf

This will create Roboto-Regular-stenciled.ttf with bridges added to glyphs like
O, A, B, D, P, R, Q, 4, 6, 8, 9, @, etc.
"""

__version__ = "0.1.0"
__author__ = "Dimosthenis Kaponis"

__all__ = ["__author__", "__version__"]
