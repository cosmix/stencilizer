"""Font reader for loading TTF/OTF fonts.

This module provides the FontReader class for loading font files
and extracting glyph data into domain models.
"""

from collections.abc import Iterator
from pathlib import Path

from fontTools.ttLib import TTFont

from stencilizer.domain.glyph import Glyph
from stencilizer.io.converter import fonttools_glyph_to_domain


class FontReader:
    """Loads TTF/OTF fonts and extracts glyph data.

    The FontReader provides a high-level interface for loading fonts
    and converting fonttools representations to domain models.

    Example:
        reader = FontReader(Path("font.ttf"))
        reader.load()
        for glyph in reader.iter_glyphs():
            print(glyph.name)
    """

    def __init__(self, font_path: Path) -> None:
        """Initialize the font reader.

        Args:
            font_path: Path to the TTF or OTF font file
        """
        self._font_path = font_path
        self._font: TTFont | None = None

    def load(self) -> None:
        """Load the font file.

        Raises:
            FileNotFoundError: If font file does not exist
            Exception: If font file is invalid or cannot be loaded
        """
        if not self._font_path.exists():
            raise FileNotFoundError(f"Font file not found: {self._font_path}")

        self._font = TTFont(str(self._font_path))

    @property
    def format(self) -> str:
        """Return font format.

        Returns:
            'TrueType' for TTF fonts, 'OpenType' for OTF fonts

        Raises:
            RuntimeError: If font has not been loaded yet
        """
        if self._font is None:
            raise RuntimeError("Font not loaded. Call load() first.")

        if "CFF " in self._font or "CFF2" in self._font:
            return "OpenType"
        return "TrueType"

    @property
    def units_per_em(self) -> int:
        """Return font's units per em.

        The units per em (UPM) defines the resolution of the font's
        coordinate system. Common values are 1000 or 2048.

        Returns:
            Units per em value

        Raises:
            RuntimeError: If font has not been loaded yet
        """
        if self._font is None:
            raise RuntimeError("Font not loaded. Call load() first.")

        return self._font["head"].unitsPerEm  # type: ignore[attr-defined]

    @property
    def glyph_count(self) -> int:
        """Return total number of glyphs in the font.

        Returns:
            Number of glyphs

        Raises:
            RuntimeError: If font has not been loaded yet
        """
        if self._font is None:
            raise RuntimeError("Font not loaded. Call load() first.")

        return self._font["maxp"].numGlyphs

    def iter_glyphs(self) -> Iterator[Glyph]:
        """Iterate over all glyphs, converting to domain model.

        Yields glyphs in the order they appear in the font.
        Converts fonttools glyph representations to domain Glyph models.

        Yields:
            Glyph domain models

        Raises:
            RuntimeError: If font has not been loaded yet
        """
        if self._font is None:
            raise RuntimeError("Font not loaded. Call load() first.")

        _glyph_set = self._font.getGlyphSet()
        glyph_order = self._font.getGlyphOrder()

        for glyph_name in glyph_order:
            glyph = self.get_glyph(glyph_name)
            if glyph is not None:
                yield glyph

    def get_glyph(self, name: str) -> Glyph | None:
        """Get a specific glyph by name.

        Args:
            name: Name of the glyph to retrieve

        Returns:
            Glyph domain model, or None if glyph not found

        Raises:
            RuntimeError: If font has not been loaded yet
        """
        if self._font is None:
            raise RuntimeError("Font not loaded. Call load() first.")

        if name not in self._font.getGlyphOrder():
            return None

        glyph_set = self._font.getGlyphSet()
        fonttools_glyph = glyph_set[name]

        try:
            return fonttools_glyph_to_domain(
                name=name,
                fonttools_glyph=fonttools_glyph,
                font=self._font
            )
        except Exception:
            return None

    def close(self) -> None:
        """Close the font file and free resources."""
        if self._font is not None:
            self._font.close()
            self._font = None

    def __enter__(self) -> "FontReader":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
