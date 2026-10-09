"""Fixture paths, font builders and glyph helpers shared by the test suite.

The name does not match ``test_*.py``, so pytest does not collect this module; tests
import from it with ``from tests.font_helpers import ...``.
"""

import copy
from pathlib import Path

from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2  # type: ignore[import-untyped]
from fontTools.misc.textTools import Tag  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.tables._f_v_a_r import Axis, table__f_v_a_r  # type: ignore[import-untyped]

from stencilizer.core import GlyphAnalyzer
from stencilizer.domain.contour import Point
from stencilizer.domain.glyph import Glyph
from stencilizer.io.converter import fonttools_glyph_to_domain

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VARIABLE_DIR = FIXTURES_DIR / "variable"
ROBOTO = FIXTURES_DIR / "Roboto-Regular.ttf"
COMMIT_MONO = FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf"
UBUNTU = VARIABLE_DIR / "Ubuntu-VF-subset.ttf"
INTER = VARIABLE_DIR / "Inter-VF-subset.ttf"
CANTARELL = VARIABLE_DIR / "Cantarell-VF-subset.otf"


def fvar_only_roboto() -> TTFont:
    """Roboto plus a one-axis fvar and no gvar: a variable font without variation data."""
    font = TTFont(ROBOTO)
    axis = Axis()
    axis.axisTag = Tag("wght")
    axis.minValue = 100.0
    axis.defaultValue = 400.0
    axis.maxValue = 900.0
    axis.axisNameID = 256
    fvar = table__f_v_a_r()
    fvar.axes = [axis]
    fvar.instances = []
    font["fvar"] = fvar
    return font


def commit_mono_cff2() -> TTFont:
    """The CommitMono fixture converted to static CFF2 outlines."""
    font = TTFont(COMMIT_MONO)
    convertCFFToCFF2(font)
    return font


def write_commit_mono_cff2(path: Path) -> Path:
    """Save the CommitMono fixture as a static CFF2 font at ``path`` and return ``path``."""
    commit_mono_cff2().save(path)
    return path


def vsindex_cantarell() -> TTFont:
    """Cantarell with a second VarData listing VarData 0's regions in reverse order."""
    font = TTFont(CANTARELL)
    store = font["CFF2"].cff.topDictIndex[0].VarStore.otVarStore
    extra = copy.deepcopy(store.VarData[0])
    extra.VarRegionIndex = list(reversed(extra.VarRegionIndex))
    store.VarData.append(extra)
    store.VarDataCount = len(store.VarData)
    return font


def cff2_program(font: TTFont, name: str) -> list[object]:
    """The decompiled charstring program of glyph ``name`` in a CFF2 font."""
    charstring = font["CFF2"].cff.topDictIndex[0].CharStrings[name]
    charstring.decompile()
    return list(charstring.program)


def units_per_em(font: TTFont) -> int:
    return int(font["head"].unitsPerEm)


def island_count(glyph: Glyph, upm: int) -> int:
    return len(GlyphAnalyzer().analyze(glyph, upm).get_islands())


def points(glyph: Glyph) -> list[Point]:
    return [point for contour in glyph.contours for point in contour.points]


def glyph_at(font: TTFont, name: str, location: dict[str, float]) -> Glyph:
    """Read glyph ``name`` of ``font`` at a normalized axis location."""
    glyph_set = font.getGlyphSet(location=location, normalized=True)
    return fonttools_glyph_to_domain(name, glyph_set[name], font)
