"""Tests for opening a GUI font session and previewing its glyphs."""

import hashlib
from pathlib import Path

import pytest
from fontTools.fontBuilder import FontBuilder  # type: ignore[import-untyped]
from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.exceptions import FontLoadError, GlyphNotFoundError
from stencilizer.gui.session import FontSession


def test_open_roboto(processor: FontProcessor, roboto_path: Path) -> None:
    """Opening Roboto exposes its measured metadata and island selection."""
    session = FontSession.open(roboto_path, processor)

    assert session.font_format == "TrueType"
    assert session.units_per_em == 2048
    assert session.ascender == 2146
    assert session.descender == -555
    assert len(session.island_glyphs) == 562
    assert session.glyph("O") is not None
    assert session.glyph("space") is None


def test_open_commit_mono_previews_first_glyph(
    processor: FontProcessor, commit_mono_path: Path
) -> None:
    """Opening the CFF fixture supports an in-process glyph preview."""
    session = FontSession.open(commit_mono_path, processor)

    assert session.font_format == "OpenType"
    assert len(session.island_glyphs) == 467
    result = session.preview(session.island_glyphs[0].name, BridgeConfig(), _geometry())
    assert result.stenciled is not None


def _geometry() -> GeometryConfig:
    """Return the default geometry configuration for previews."""
    return GeometryConfig()


def test_open_wraps_invalid_and_missing_files(processor: FontProcessor, tmp_path: Path) -> None:
    """Invalid and missing inputs become GUI load errors."""
    invalid_path = tmp_path / "invalid.ttf"
    invalid_path.write_bytes(b"not a font")

    with pytest.raises(FontLoadError):
        FontSession.open(invalid_path, processor)
    with pytest.raises(FontLoadError):
        FontSession.open(tmp_path / "missing.ttf", processor)


def test_open_rejects_non_regular_file(processor: FontProcessor, tmp_path: Path) -> None:
    """A directory named like a font is refused before any read is attempted."""
    folder = tmp_path / "folder.ttf"
    folder.mkdir()

    with pytest.raises(FontLoadError, match="not a regular file"):
        FontSession.open(folder, processor)


def test_open_pins_source_digest(processor: FontProcessor, roboto_path: Path) -> None:
    """Opening records the exact source revision used for previews and saves."""
    session = FontSession.open(roboto_path, processor)

    assert session.source_sha256 == hashlib.sha256(roboto_path.read_bytes()).hexdigest()


def test_preview_uses_parameters(processor: FontProcessor, roboto_path: Path) -> None:
    """Preview output changes with bridge width and spanning behavior."""
    session = FontSession.open(roboto_path, processor)
    default_o = session.preview("O", BridgeConfig(), _geometry())
    narrow_o = session.preview("O", BridgeConfig(width_percent=30.0), _geometry())
    wide_o = session.preview("O", BridgeConfig(width_percent=110.0), _geometry())
    spanning_b = session.preview("B", BridgeConfig(use_spanning_bridges=True), _geometry())
    split_b = session.preview("B", BridgeConfig(use_spanning_bridges=False), _geometry())

    assert default_o.bridges_added == 1
    assert default_o.error is None
    assert len(default_o.original.contours) == 2
    assert default_o.stenciled is not None
    assert len(default_o.stenciled.contours) == 4
    assert narrow_o.stenciled is not None
    assert wide_o.stenciled is not None
    assert spanning_b.stenciled is not None
    assert split_b.stenciled is not None
    assert narrow_o.stenciled.to_dict() != wide_o.stenciled.to_dict()
    assert spanning_b.stenciled.to_dict() != split_b.stenciled.to_dict()


def test_preview_rejects_unselected_glyph(processor: FontProcessor, roboto_path: Path) -> None:
    """A glyph outside the island selection cannot be previewed."""
    session = FontSession.open(roboto_path, processor)

    with pytest.raises(GlyphNotFoundError):
        session.preview("space", BridgeConfig(), _geometry())


def test_preview_reports_transform_error(
    processor: FontProcessor, roboto_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A glyph the core fails to transform yields an error preview without outlines."""
    session = FontSession.open(roboto_path, processor)

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        """Report a transform failure the way process_glyph does."""
        return {"error": "boom", "duration_ms": 1.0}

    monkeypatch.setattr("stencilizer.gui.session.process_glyph", fail)
    result = session.preview("O", BridgeConfig(), _geometry())

    assert result.glyph_name == "O"
    assert result.original is session.glyph("O")
    assert result.stenciled is None
    assert result.error == "boom"
    assert result.bridges_added == 0
    assert result.duration_ms == 1.0


def _build_cyclic_component_font(path: Path) -> Path:
    """Build a tiny TrueType font whose glyphs 'x' and 'y' reference each other."""
    names = (".notdef", "x", "y")
    glyph_set: dict[str, None] = dict.fromkeys(names)

    notdef_pen = TTGlyphPen(glyph_set)
    notdef_pen.moveTo((0, 0))
    notdef_pen.lineTo((0, 10))
    notdef_pen.lineTo((10, 10))
    notdef_pen.lineTo((10, 0))
    notdef_pen.closePath()
    notdef_glyph = notdef_pen.glyph()
    notdef_glyph.xMin, notdef_glyph.yMin, notdef_glyph.xMax, notdef_glyph.yMax = 0, 0, 10, 10

    x_pen = TTGlyphPen(glyph_set)
    x_pen.addComponent("y", (1, 0, 0, 1, 0, 0))
    x_glyph = x_pen.glyph()
    x_glyph.xMin = x_glyph.yMin = x_glyph.xMax = x_glyph.yMax = 0

    y_pen = TTGlyphPen(glyph_set)
    y_pen.addComponent("x", (1, 0, 0, 1, 0, 0))
    y_glyph = y_pen.glyph()
    y_glyph.xMin = y_glyph.yMin = y_glyph.xMax = y_glyph.yMax = 0

    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(list(names))
    builder.setupCharacterMap({})
    # Skip bounds recalculation: fontTools recurses through addComponent
    # references with no cycle guard, so compiling a self-referencing pair
    # would hang here instead of ever reaching FontSession.open.
    builder.font.recalcBBoxes = False
    builder.setupGlyf({".notdef": notdef_glyph, "x": x_glyph, "y": y_glyph}, calcGlyphBounds=False)
    builder.setupHorizontalMetrics(dict.fromkeys(names, (600, 0)))
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Cycle", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)
    return path


def test_open_wraps_component_cycle(processor: FontProcessor, tmp_path: Path) -> None:
    """A font whose components form a cycle is reported as a load error, not raised raw."""
    path = _build_cyclic_component_font(tmp_path / "cycle.ttf")

    with pytest.raises(FontLoadError, match="references itself"):
        FontSession.open(path, processor)
