"""Tests for Qt rendering of stencilizer domain glyphs."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.pens.reverseContourPen import ReverseContourPen  # type: ignore[import-untyped]
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QTransform

from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point
from stencilizer.gui.outline import (
    draw_glyph,
    font_to_widget_transform,
    glyph_frame,
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


def _round_point(point: tuple[float, float]) -> tuple[float, float]:
    """Round a coordinate pair for stable comparison across pen implementations."""
    return (round(point[0], 3), round(point[1], 3))


def _split_contours(
    recording: list[tuple[str, tuple[Any, ...]]],
) -> list[list[tuple[str, tuple[Any, ...]]]]:
    """Split a RecordingPen value into one command list per closed contour."""
    contours: list[list[tuple[str, tuple[Any, ...]]]] = []
    current: list[tuple[str, tuple[Any, ...]]] = []
    for command, args in recording:
        current.append((command, args))
        if command in ("closePath", "endPath"):
            contours.append(current)
            current = []
    return contours


def _closed_edge_cycle(
    commands: list[tuple[str, tuple[Any, ...]]],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Round a contour's segments after its moveTo, with any implied closing line made explicit.

    The result traces the same closed shape regardless of which point the recording started
    from, so two contours drawn from different starting points can still be compared.
    """
    move_command, move_args = commands[0]
    assert move_command == "moveTo"
    start = _round_point(move_args[0])
    edges = [
        (command, tuple(_round_point(point) for point in args)) for command, args in commands[1:-1]
    ]
    if not edges or edges[-1][1][-1] != start:
        edges.append(("lineTo", (start,)))
    return tuple(edges)


def _is_rotation(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> bool:
    """Return whether actual traces the same closed cycle as expected from some other start."""
    if len(actual) != len(expected):
        return False
    doubled = expected + expected
    return any(doubled[index : index + len(expected)] == actual for index in range(len(expected)))


@pytest.mark.parametrize("name", ["O", "B"])
def test_draw_cff_glyph_matches_reversed_fonttools_segments(
    commit_mono_path: Path, name: str
) -> None:
    """draw_glyph reproduces each CFF contour's exact segments, normalized to TrueType winding.

    The reader reverses CFF contour point order on load, so the domain glyph's contours start
    at a different point than the font's own charstring; contours are compared as cyclic
    sequences of segments rather than requiring an identical starting point.
    """
    reader = FontReader(commit_mono_path)
    reader.load()
    glyph = reader.get_glyph(name)
    assert glyph is not None

    actual = RecordingPen()
    draw_glyph(glyph, actual)

    expected = RecordingPen()
    reader.font.getGlyphSet()[name].draw(ReverseContourPen(expected))

    actual_contours = _split_contours(actual.value)
    expected_contours = _split_contours(expected.value)
    assert len(actual_contours) == len(expected_contours)
    for actual_commands, expected_commands in zip(actual_contours, expected_contours, strict=True):
        actual_cycle = _closed_edge_cycle(actual_commands)
        expected_cycle = _closed_edge_cycle(expected_commands)
        assert _is_rotation(actual_cycle, expected_cycle)
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


def test_glyph_frame_with_no_points_uses_advance_and_vertical_metrics() -> None:
    """A glyph with no outline points frames to its advance width and font vertical metrics."""
    empty_glyph = _glyph([], advance_width=500)

    frame = glyph_frame(empty_glyph, 800, -200)

    assert frame == QRectF(0.0, -200.0, 500.0, 1000.0)


@pytest.mark.parametrize("frame", [QRectF(0.0, 0.0, 0.0, 100.0), QRectF(0.0, 0.0, 100.0, 0.0)])
def test_font_to_widget_transform_zero_size_frame_returns_identity(frame: QRectF) -> None:
    """A frame with no width or no height cannot be scaled, so the transform is identity."""
    target = QRectF(0.0, 0.0, 200.0, 100.0)

    assert font_to_widget_transform(frame, target) == QTransform()
