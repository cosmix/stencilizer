"""Font writer for saving modified fonts.

This module provides the FontWriter class for writing modified fonts
with the stencilized naming convention.
"""

from datetime import datetime
from pathlib import Path

from fontTools.ttLib import TTFont

from stencilizer.domain.glyph import Glyph
from stencilizer.io.converter import domain_glyph_to_fonttools

# Name table IDs we modify
NAME_ID_FAMILY = 1
NAME_ID_FULL_NAME = 4
NAME_ID_VERSION = 5
NAME_ID_POSTSCRIPT = 6
NAME_ID_TYPOGRAPHIC_FAMILY = 16


def update_font_names(font: TTFont, suffix: str = " Stenciled") -> None:
    """Update font name table entries with stenciled suffix.

    Modifies the font's name table so the stencilized font can coexist
    with the original when installed on the same system.

    Args:
        font: The fonttools TTFont object to modify
        suffix: Suffix to add (default: " Stenciled")
    """
    name_table = font["name"]

    # Collect updates to apply (avoid modifying while iterating)
    updates: list[tuple[int, int, int, int, str]] = []

    for record in name_table.names:
        name_id = record.nameID
        platform_id = record.platformID
        plat_enc_id = record.platEncID
        lang_id = record.langID

        try:
            original = record.toUnicode()
        except UnicodeDecodeError:
            continue

        new_name: str | None = None

        if name_id == NAME_ID_FAMILY:
            # "Roboto" → "Roboto Stenciled"
            new_name = original + suffix

        elif name_id == NAME_ID_TYPOGRAPHIC_FAMILY:
            # Same treatment as family name
            new_name = original + suffix

        elif name_id == NAME_ID_FULL_NAME:
            # "Roboto Regular" → "Roboto Stenciled Regular"
            # Insert suffix before the last word (style name)
            parts = original.rsplit(" ", 1)
            if len(parts) == 2:
                new_name = f"{parts[0]}{suffix} {parts[1]}"
            else:
                new_name = original + suffix

        elif name_id == NAME_ID_POSTSCRIPT:
            # "Roboto-Regular" → "RobotoStenciled-Regular"
            # PostScript names can't have spaces
            ps_suffix = suffix.replace(" ", "")
            if "-" in original:
                parts = original.split("-", 1)
                new_name = f"{parts[0]}{ps_suffix}-{parts[1]}"
            else:
                new_name = original + ps_suffix

        elif name_id == NAME_ID_VERSION:
            # Append stencilization timestamp to version
            # "Version 2.015" → "Version 2.015; Stencilized 2025-12-23T14:30:45"
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            new_name = f"{original}; Stencilized {timestamp}"

        if new_name is not None:
            updates.append((name_id, platform_id, plat_enc_id, lang_id, new_name))

    # Apply all updates
    for name_id, platform_id, plat_enc_id, lang_id, new_name in updates:
        name_table.setName(new_name, name_id, platform_id, plat_enc_id, lang_id)


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

        Updates font name table with stenciled suffix before saving
        so the font can coexist with the original.

        Raises:
            IOError: If file cannot be written
        """
        update_font_names(self._font)
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

        stenciled_name = f"{stem}-Stenciled{suffix}"
        return parent / stenciled_name
