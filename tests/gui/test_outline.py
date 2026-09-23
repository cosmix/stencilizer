"""Tests for Qt rendering of stencilizer domain glyphs."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.pens.boundsPen import BoundsPen  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor

from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point
from stencilizer.gui.outline import (
    draw_glyph,
    font_to_widget_transform,
    glyph_frame,
    glyph_path,
    render_glyph_image,
)
from stencilizer.io import FontReader


def _glyph(contours: list[Contour], advance_width: int = 100) -> Glyph:
    """Build a small test glyph from the supplied contours."""
    metadata = GlyphMetadata("test", None, advance_width, 0)
    return Glyph(metadata, contours)


def _contour(coordinates: list[tuple[float, float]]) -> Contour:
    """Build an on-curve contour from coordinate pairs."""
    return Contour([Point(x, y) for x, y in coordinates])


@pytest.mark.parametrize("name", ["O", "B", "eight", "a", "g", "at"])
def test_draw_truetype_glyph_matches_fonttools(roboto_path: Path, name: str) -> None:
    """The domain-to-pen walk reproduces each non-composite TrueType outline."""
    reader = FontReader(roboto_path)
    reader.load()
    glyph = reader.get_glyph(name)
    assert glyph is not None

    expected = RecordingPen()
    reader.font.getGlyphSet()[name].draw(expected)
    actual = RecordingPen()
    draw_glyph(glyph, actual)

    assert actual.value == expected.value
    reader.close()


@pytest.mark.parametrize("name", ["O", "B"])
def test_cff_path_bounds_match_fonttools(commit_mono_path: Path, name: str) -> None:
    """Qt paths preserve the bounds of CFF glyphs."""
    reader = FontReader(commit_mono_path)
    reader.load()
    glyph = reader.get_glyph(name)
    assert glyph is not None

    bounds_pen = BoundsPen(None)
    reader.font.getGlyphSet()[name].draw(bounds_pen)
    expected_bounds = bounds_pen.bounds
    assert expected_bounds is not None

    actual_bounds = glyph_path(glyph).boundingRect()
    actual_coordinates = (
        actual_bounds.left(),
        actual_bounds.top(),
        actual_bounds.right(),
        actual_bounds.bottom(),
    )
    assert actual_coordinates == pytest.approx(expected_bounds)
    reader.close()


def test_winding_fill_shows_broken_hole(qapp: Any) -> None:
    """A same-winding inner contour remains filled under the nonzero rule."""
    assert qapp is not None
    outer = _contour([(0, 0), (0, 100), (100, 100), (100, 0)])
    intact_inner = _contour([(30, 30), (70, 30), (70, 70), (30, 70)])
    broken_inner = _contour([(30, 30), (30, 70), (70, 70), (70, 30)])
    frame = QRectF(0.0, 0.0, 100.0, 100.0)
    black = QColor("black")
    white = QColor("white")

    intact_image = render_glyph_image(_glyph([outer, intact_inner]), frame, 32, black, white)
    broken_image = render_glyph_image(_glyph([outer, broken_inner]), frame, 32, black, white)

    assert intact_image.pixelColor(16, 16) == white
    assert broken_image.pixelColor(16, 16) == black


def test_font_to_widget_transform_maps_frame_landmarks() -> None:
    """The transform centers the frame and flips its font-unit y-axis."""
    frame = QRectF(0.0, -500.0, 1000.0, 2500.0)
    target = QRectF(0.0, 0.0, 200.0, 100.0)
    transform = font_to_widget_transform(frame, target)

    assert transform.map(frame.center()) == QPointF(100.0, 50.0)
    assert transform.map(QPointF(500.0, frame.bottom())) == QPointF(100.0, 0.0)
    assert transform.map(QPointF(500.0, frame.top())) == QPointF(100.0, 100.0)


def test_glyph_frame_contains_font_metrics_and_outline_extents(roboto_path: Path) -> None:
    """Frames include advance width, vertical metrics, and outlying points."""
    reader = FontReader(roboto_path)
    reader.load()
    glyph = reader.get_glyph("O")
    assert glyph is not None

    max_x = max(point.x for contour in glyph.contours for point in contour.points)
    assert max_x < glyph.metadata.advance_width

    frame = glyph_frame(glyph, 2146, -555)
    assert frame.left() == 0.0
    assert frame.right() == glyph.metadata.advance_width
    assert frame.top() == -555.0
    assert frame.bottom() == 2146.0

    tall_glyph = _glyph([_contour([(0, 0), (100, 2500), (100, 0)])])
    tall_frame = glyph_frame(tall_glyph, 2146, -555)
    assert tall_frame.bottom() == 2500.0

    wide_glyph = _glyph([_contour([(0, 0), (150, 50), (150, 0)])], advance_width=100)
    wide_frame = glyph_frame(wide_glyph, 2146, -555)
    assert wide_frame.right() == 150.0
    reader.close()
