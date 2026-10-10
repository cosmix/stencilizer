"""Variable-font name and writer-dispatch tests."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.exceptions import FontFormatError
from stencilizer.io.writer import FontWriter
from stencilizer.variable.model import VariableGlyph
from tests.font_helpers import CANTARELL, INTER


def test_variable_save_suffixes_postscript_and_full_names(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"
    FontWriter(TTFont(INTER), output).save()

    names = TTFont(output)["name"]
    assert names.getDebugName(25) == "InterVariableStenciled"
    assert names.getDebugName(280) == "InterVariableStenciled-Thin"
    assert names.getDebugName(4) == "Inter Variable Stenciled"
    assert names.getDebugName(1) == "Inter Variable Stenciled"
    assert names.getDebugName(6) == "InterVariableStenciled"


def test_variable_save_suffixes_shared_postscript_name_once(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"
    font = TTFont(INTER)
    font["fvar"].instances[0].postscriptNameID = 280
    font["fvar"].instances[1].postscriptNameID = 280

    FontWriter(font, output).save()

    assert TTFont(output)["name"].getDebugName(280) == "InterVariableStenciled-Thin"


def test_update_variable_glyph_rejects_unknown_name(tmp_path: Path) -> None:
    vg = VariableGlyph(
        default=Glyph(metadata=GlyphMetadata("nope", None, 500, 0), contours=[]),
        supports=(),
        masters=(),
        axis_tags=("wght",),
    )

    with pytest.raises(ValueError, match="nope"):
        FontWriter(TTFont(INTER), tmp_path / "out.ttf").update_variable_glyph(vg)


def test_update_glyph_rejects_variable_font(tmp_path: Path) -> None:
    glyph = Glyph(metadata=GlyphMetadata("A", None, 500, 0), contours=[])

    with pytest.raises(FontFormatError, match="update_variable_glyph"):
        FontWriter(TTFont(CANTARELL), tmp_path / "out.ttf").update_glyph(glyph)
