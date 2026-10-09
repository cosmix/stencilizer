"""Stenciled CFF2 output stays closed when fontTools' instancer pins it to a static font."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.pens.basePen import NullPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.varLib.instancer import instantiateVariableFont  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.io import FontReader
from stencilizer.io.converter import fonttools_glyph_to_domain
from tests.font_helpers import CANTARELL, island_count, units_per_em

GLYPHS = ("o", "O", "zero", "B")


@pytest.fixture(scope="module")
def stenciled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("instancing")
    settings = StencilizerSettings(logging=LoggingConfig(log_file=workdir / "log.txt"))
    with FontReader(CANTARELL) as reader:
        classification = FontProcessor(settings).classify_glyphs(reader)
    out = workdir / CANTARELL.name
    stats = FontProcessor(settings).process(
        font_path=CANTARELL, output_path=out, classification=classification
    )
    assert stats.error_count == 0, stats.errors
    return out


def _wght_values(font: TTFont) -> list[float]:
    axis = next(a for a in font["fvar"].axes if a.axisTag == "wght")
    return [axis.minValue, axis.defaultValue, axis.maxValue]


@pytest.mark.parametrize("which", [0, 1, 2], ids=["min", "default", "max"])
def test_instances_have_no_islands(stenciled: Path, which: int) -> None:
    wght = _wght_values(TTFont(stenciled))[which]
    instance = instantiateVariableFont(TTFont(stenciled), {"wght": wght}, inplace=False)
    upm = units_per_em(instance)
    glyph_set = instance.getGlyphSet()
    for name in GLYPHS:
        glyph = fonttools_glyph_to_domain(name, glyph_set[name], instance)
        assert island_count(glyph, upm) == 0, (name, wght)


def _charstrings(font: TTFont) -> Any:
    return font["CFF2"].cff.topDictIndex[0].CharStrings


def _bytecode(charstrings: Any, name: str) -> bytes:
    charstring = charstrings[name]
    charstring.compile(isCFF2=True)
    return bytes(charstring.bytecode)


def test_blend_operands_are_integers(stenciled: Path) -> None:
    """Every rewritten charstring blends integer deltas.

    Untouched glyphs keep the source's charstrings, which may carry fractional deltas
    (Cantarell's ``four`` does), so only rewritten glyphs are checked.
    """
    font = TTFont(stenciled)
    source, output = _charstrings(TTFont(CANTARELL)), _charstrings(font)
    rewritten = [
        name for name in font.getGlyphOrder() if _bytecode(output, name) != _bytecode(source, name)
    ]
    assert set(GLYPHS) <= set(rewritten)
    fractional: list[tuple[str, float]] = []
    for name in rewritten:

        def blender(_vs_index: int, deltas: list[float], name: str = name) -> float:
            fractional.extend((name, d) for d in deltas if not float(d).is_integer())
            return 0

        output[name].draw(NullPen(), blender)
    assert fractional == []
