"""Font I/O layer for stencilizer.

This module handles reading and writing font files using fonttools.
It provides a clean abstraction layer between fonttools and the
domain models.

Key responsibilities:
- Load TTF/OTF fonts
- Convert fonttools representations to domain models
- Write modified fonts with proper naming convention
- Format detection (TTF vs OTF)

Key classes:
- FontReader: Load fonts and extract glyphs
- FontWriter: Save modified fonts
"""

from stencilizer.io.reader import FontReader
from stencilizer.io.writer import FontWriter

__all__ = [
    "FontReader",
    "FontWriter",
]
