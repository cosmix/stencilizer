"""Font writer for saving modified fonts.

This module provides the FontWriter class for writing modified fonts
with the stencilized naming convention.
"""

from pathlib import Path

from fontTools.ttLib import TTFont

from stencilizer.domain.glyph import Glyph
from stencilizer.io.converter import domain_glyph_to_fonttools


class FontWriter:
    """Writes modified fonts with stencilized naming convention.

    The FontWriter handles updating glyphs in a font and saving
    with the appropriate naming convention.

    Example:
        writer = FontWriter(font, Path("output.ttf"))
        writer.update_glyph(modified_glyph)
        writer.save()
    """

    def __init__(self, font: TTFont, output_path: Path) -> None:
        """Initialize the font writer.

        Args:
            font: The fonttools TTFont object to write
            output_path: Path where the font will be saved
        """
        self._font = font
        self._output_path = output_path

    def update_glyph(self, glyph: Glyph) -> None:
        """Update a glyph in the font from domain model.

        Converts the domain Glyph model back to fonttools representation
        and updates it in the font.

        Args:
            glyph: Domain glyph model with modifications

        Raises:
            ValueError: If glyph name not found in font
        """
        glyph_name = glyph.name

        if glyph_name not in self._font.getGlyphOrder():
            raise ValueError(f"Glyph '{glyph_name}' not found in font")

        glyph_set = self._font.getGlyphSet()
        original_glyph = glyph_set[glyph_name]

        domain_glyph_to_fonttools(glyph, original_glyph, self._font)

    def save(self) -> None:
        """Save the font file to the output path.

        Raises:
            IOError: If file cannot be written
        """
        self._font.save(str(self._output_path))

    @staticmethod
    def get_stenciled_path(input_path: Path) -> Path:
        """Generate output path with stencilized naming convention.

        Converts: font.ttf -> font-stenciled.ttf
                  Roboto-Regular.otf -> Roboto-Regular-stenciled.otf

        Args:
            input_path: Original font file path

        Returns:
            Path with -stenciled suffix before extension
        """
        stem = input_path.stem
        suffix = input_path.suffix
        parent = input_path.parent

        stenciled_name = f"{stem}-stenciled{suffix}"
        return parent / stenciled_name
