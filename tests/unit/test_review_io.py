"""Regression tests for font format and outline IO edge cases."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.exceptions import FontFormatError, GlyphProcessingError
from stencilizer.io.converter import (
    _draw_closed_contour,
    _recording_to_contours,
    _update_truetype_glyph,
)
from stencilizer.io.reader import FontReader
from stencilizer.io.writer import FontWriter


def _glyph(name: str = "A") -> Glyph:
    return Glyph(GlyphMetadata(name, ord(name), 500, 0), [])


@pytest.mark.parametrize("table", ["fvar", "CFF2"])
def test_reader_rejects_unsupported_fonts_before_exposing_font(table: str) -> None:
    font = MagicMock()
    font.__contains__.side_effect = lambda name: name == table
    reader = FontReader(Path("input.ttf"))
    with (
        patch.object(Path, "exists", return_value=True),
        patch("stencilizer.io.reader.TTFont", return_value=font),
        pytest.raises(FontFormatError, match=table if table == "CFF2" else "variable"),
    ):
        reader.load()
    assert reader._font is None
    font.close.assert_called_once()


@pytest.mark.parametrize("table", ["fvar", "CFF2"])
def test_writer_rejects_unsupported_fonts_without_output(table: str) -> None:
    font = MagicMock()
    font.__contains__.side_effect = lambda name: name == table
    writer = FontWriter(font, Path("output.ttf"))
    with pytest.raises(FontFormatError):
        writer.update_glyph(_glyph())
    with pytest.raises(FontFormatError):
        writer.save()
    font.save.assert_not_called()


def test_all_off_curve_quadratic_loop_gets_implied_start() -> None:
    contours = _recording_to_contours(
        [
            ("qCurveTo", ((0, 0), (100, 0), (100, 100), None)),
            ("closePath", ()),
        ]
    )
    assert len(contours) == 1
    points = contours[0].points
    assert points[0] == Point(50, 50)
    assert [point.point_type for point in points[1:]] == [PointType.OFF_CURVE_QUAD] * 3
    pen = RecordingPen()
    _draw_closed_contour(pen, points, PointType.OFF_CURVE_QUAD)
    assert pen.value[1] == ("qCurveTo", ((0, 0), (100, 0), (100, 100), (50, 50)))


def test_trailing_quadratic_control_closes_to_first_on_curve() -> None:
    points = [Point(0, 0), Point(100, 0), Point(100, 100, PointType.OFF_CURVE_QUAD)]
    pen = RecordingPen()
    _draw_closed_contour(pen, points, PointType.OFF_CURVE_QUAD)
    assert pen.value == [
        ("moveTo", ((0, 0),)),
        ("lineTo", ((100, 0),)),
        ("qCurveTo", ((100, 100), (0, 0))),
        ("closePath", ()),
    ]


def test_trailing_quadratic_control_survives_truetype_serialization() -> None:
    font = MagicMock()
    glyf: dict[str, Any] = {}
    font.__getitem__.side_effect = lambda _: glyf
    font.getGlyphSet.return_value = {}
    glyph = _glyph()
    glyph.contours = [
        Contour([Point(0, 0), Point(100, 0), Point(100, 100, PointType.OFF_CURVE_QUAD)])
    ]
    _update_truetype_glyph(glyph, None, font)
    raw = glyf["A"]
    assert list(raw.flags) == [1, 1, 0]
    assert list(raw.coordinates) == [(0, 0), (100, 0), (100, 100)]


def test_cubic_contour_with_off_curve_first_keeps_closing_curve() -> None:
    points = [
        Point(100, 100, PointType.OFF_CURVE_CUBIC),
        Point(100, 0),
        Point(0, 0),
        Point(0, 100, PointType.OFF_CURVE_CUBIC),
    ]
    pen = RecordingPen()
    _draw_closed_contour(pen, points, PointType.OFF_CURVE_CUBIC)
    assert pen.value == [
        ("moveTo", ((100, 0),)),
        ("lineTo", ((0, 0),)),
        ("curveTo", ((0, 100), (100, 100), (100, 0))),
        ("closePath", ()),
    ]


def test_reader_iteration_surfaces_glyph_conversion_errors() -> None:
    reader = FontReader(Path("input.ttf"))
    reader._font = MagicMock()
    reader._font.getGlyphOrder.return_value = ["A"]
    bad_glyph = MagicMock()
    bad_glyph.draw.side_effect = ValueError("bad outline")
    reader._font.getGlyphSet.return_value = {"A": bad_glyph}
    reader._font.getBestCmap.return_value = {65: "A"}
    with pytest.raises(GlyphProcessingError, match="bad outline"):
        list(reader.iter_glyphs())


def test_reader_reuses_font_wide_glyph_and_unicode_lookups() -> None:
    font = MagicMock()
    font.getGlyphOrder.return_value = ["A", "B"]
    font.getGlyphSet.return_value = {"A": object(), "B": object()}
    font.getBestCmap.return_value = {65: "A", 66: "B"}
    reader = FontReader(Path("input.ttf"))
    reader._font = font
    with patch(
        "stencilizer.io.reader.fonttools_glyph_to_domain",
        side_effect=[_glyph("A"), _glyph("B")],
    ) as convert:
        assert reader.get_glyph("A") == _glyph("A")
        assert reader.get_glyph("B") == _glyph("B")
    assert reader.get_glyph("missing") is None
    font.getGlyphOrder.assert_called_once()
    font.getGlyphSet.assert_called_once()
    font.getBestCmap.assert_called_once()
    assert convert.call_args_list[0].kwargs["unicode_by_name"] == {"A": 65, "B": 66}
