"""Tests for the glyph preview widgets."""

from pathlib import Path

from PySide6.QtGui import QColor
from pytestqt.qtbot import QtBot

from stencilizer.config import BridgeConfig
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point
from stencilizer.gui.glyph_view import ComparisonView, GlyphCanvas
from stencilizer.gui.outline import glyph_frame
from stencilizer.gui.session import FontSession, PreviewResult


def _o_preview(processor: FontProcessor, roboto_path: Path) -> tuple[FontSession, PreviewResult]:
    """Open Roboto and stencilize its O glyph with default settings."""
    session = FontSession.open(roboto_path, processor)
    return session, session.preview("O", BridgeConfig(), GeometryConfig())


def _synthetic_glyph(advance_width: int, coordinates: list[tuple[float, float]]) -> Glyph:
    """Build a minimal glyph from raw coordinates, as tests/gui/test_outline.py does."""
    metadata = GlyphMetadata("square", None, advance_width, 0)
    return Glyph(metadata, [Contour([Point(x, y) for x, y in coordinates])])


def _contains_colour(canvas: GlyphCanvas, colour: QColor) -> bool:
    """Return whether the canvas raster includes the requested RGB colour."""
    image = canvas.grab().toImage()
    return any(
        QColor(image.pixel(x, y)).rgb() == colour.rgb()
        for x in range(image.width())
        for y in range(image.height())
    )


def test_show_preview_shares_one_frame(
    qtbot: QtBot, processor: FontProcessor, roboto_path: Path
) -> None:
    """A successful preview uses one frame and describes its transform."""
    session, result = _o_preview(processor, roboto_path)
    assert result.stenciled is not None
    view = ComparisonView()
    qtbot.addWidget(view)

    view.show_preview(result, session.ascender, session.descender)

    expected_frame = glyph_frame(result.original, session.ascender, session.descender).united(
        glyph_frame(result.stenciled, session.ascender, session.descender)
    )
    assert view.before_canvas.glyph is result.original
    assert view.after_canvas.glyph is result.stenciled
    assert view.before_canvas.frame == view.after_canvas.frame
    assert view.before_canvas.frame == expected_frame
    assert "O (U+004F)" in view.info_label.text()
    assert "1 island(s) bridged" in view.info_label.text()


def test_show_preview_shares_the_union_frame(qtbot: QtBot) -> None:
    """The shared frame covers stenciled geometry that grows past the original."""
    original = _synthetic_glyph(100, [(0, 0), (0, 100), (100, 100), (100, 0)])
    stenciled = _synthetic_glyph(100, [(0, 0), (0, 100), (250, 300), (100, 0)])
    ascender, descender = 200, -50
    result = PreviewResult(
        glyph_name="square",
        original=original,
        stenciled=stenciled,
        bridges_added=0,
        error=None,
        duration_ms=0.0,
    )
    view = ComparisonView()
    qtbot.addWidget(view)

    view.show_preview(result, ascender, descender)

    original_frame = glyph_frame(original, ascender, descender)
    expected_frame = original_frame.united(glyph_frame(stenciled, ascender, descender))
    assert expected_frame != original_frame
    assert view.before_canvas.frame == expected_frame
    assert view.after_canvas.frame == expected_frame


def test_show_preview_displays_transform_failure(
    qtbot: QtBot, processor: FontProcessor, roboto_path: Path
) -> None:
    """A failed transform clears the after canvas and names the error."""
    session, result = _o_preview(processor, roboto_path)
    failure = PreviewResult(
        glyph_name=result.glyph_name,
        original=result.original,
        stenciled=None,
        bridges_added=0,
        error="boom",
        duration_ms=result.duration_ms,
    )
    view = ComparisonView()
    qtbot.addWidget(view)

    view.show_preview(failure, session.ascender, session.descender)

    assert view.after_canvas.glyph is None
    assert "transform failed: boom" in view.info_label.text()


def test_canvas_paints_and_clears_glyph(
    qtbot: QtBot, processor: FontProcessor, roboto_path: Path
) -> None:
    """A canvas paints glyph outlines in the text colour and clears them."""
    session, result = _o_preview(processor, roboto_path)
    canvas = GlyphCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas.show()
    frame = glyph_frame(result.original, session.ascender, session.descender)

    canvas.set_glyph(result.original, frame)

    assert _contains_colour(canvas, canvas.palette().text().color())
    canvas.set_glyph(None, None)
    assert not _contains_colour(canvas, canvas.palette().text().color())


def test_clear_empties_canvases_and_label(
    qtbot: QtBot, processor: FontProcessor, roboto_path: Path
) -> None:
    """Clearing a comparison removes its canvases and status text."""
    session, result = _o_preview(processor, roboto_path)
    view = ComparisonView()
    qtbot.addWidget(view)
    view.show_preview(result, session.ascender, session.descender)

    view.clear()

    assert view.before_canvas.glyph is None
    assert view.before_canvas.frame is None
    assert view.after_canvas.glyph is None
    assert view.after_canvas.frame is None
    assert view.info_label.text() == ""
