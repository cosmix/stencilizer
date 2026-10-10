"""Stem outlines, support and loaders shared by the variable replay, validate and transform tests.

The name does not match ``test_*.py``, so pytest does not collect this module; tests import
from it with ``from tests.unit._variable_cases import ...``.
"""

from pathlib import Path

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.reader import read_variable_glyph
from tests.font_helpers import units_per_em

UPM = 1000
WGHT = Support((("wght", 0.0, 1.0, 1.0),))

Outline = list[tuple[float, float]]

# A stem with an extra vertex at y=210, and its cut by a horizontal bridge from y=100 to
# y=200: BELOW's top (y=100) faces ABOVE's bottom (y=200), and every cut lies on one of the
# two vertical edges.
STEM: Outline = [(0.0, 0.0), (0.0, 300.0), (100.0, 300.0), (100.0, 210.0), (100.0, 0.0)]
BELOW: Outline = [(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)]
ABOVE: Outline = [(0.0, 200.0), (0.0, 300.0), (100.0, 300.0), (100.0, 210.0), (100.0, 200.0)]


def glyph_from_outlines(*contours: Outline) -> Glyph:
    """A glyph whose contours are the given coordinate lists, all points on-curve."""
    return Glyph(
        metadata=GlyphMetadata("test", None, 1000, 0),
        contours=[Contour([Point(x, y) for x, y in contour]) for contour in contours],
    )


def read_variable(path: Path, char: str) -> tuple[VariableGlyph, int]:
    """The variable glyph for ``char`` in the font at ``path`` and the font's units per em."""
    font = TTFont(path)
    vg = read_variable_glyph(font, str(font.getBestCmap()[ord(char)]))
    assert vg is not None
    return vg, units_per_em(font)
