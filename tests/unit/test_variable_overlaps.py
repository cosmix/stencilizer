"""Tests for compatible overlap removal on flattened variable glyphs."""

import math
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.reader import read_variable_glyph
from tests.font_helpers import INTER, UBUNTU, units_per_em
from tests.font_helpers import island_count as _islands

_Box = tuple[float, float, float, float]


def _flat(path: Path, char: str) -> tuple[int, VariableGlyph]:
    font = TTFont(path)
    upm = units_per_em(font)
    vg = read_variable_glyph(font, str(font.getBestCmap()[ord(char)]))
    assert vg is not None
    return upm, flatten_compatible(vg, upm)


def _shape(glyph: Glyph) -> list[int]:
    return [len(contour.points) for contour in glyph.contours]


def _clockwise_box(box: _Box) -> Contour:
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    return Contour([Point(x, y) for x, y in corners])


def _boxes(*boxes: _Box) -> Glyph:
    contours = [_clockwise_box(box) for box in boxes]
    return Glyph(metadata=GlyphMetadata("boxes", None, 600, 0), contours=contours)


def _two_box_glyph(default: tuple[_Box, _Box], master: tuple[_Box, _Box]) -> VariableGlyph:
    vg = VariableGlyph(
        _boxes(*default),
        (Support((("wght", 0.0, 1.0, 1.0),)),),
        (_boxes(*master),),
        ("wght",),
    )
    return flatten_compatible(vg, 1000)


@pytest.mark.parametrize("char", ["P", "e"])
def test_union_builds_counter(char: str) -> None:
    upm, flat = _flat(INTER, char)
    assert _islands(flat.default, upm) == 0

    result = remove_overlaps_compatible(flat)

    assert result is not None
    assert len(result.default.contours) == 2
    assert _islands(result.default, upm) == 1
    assert result.supports == flat.supports
    for master in result.masters:
        assert _shape(master) == _shape(result.default)
        assert _islands(master, upm) == 1


def test_union_replays_exact_input_coordinates() -> None:
    _, flat = _flat(INTER, "P")
    result = remove_overlaps_compatible(flat)
    assert result is not None

    inputs = [(p.x, p.y) for contour in flat.default.contours for p in contour.points]
    outputs = [(p.x, p.y) for contour in result.default.contours for p in contour.points]

    # A vertex taken from the input keeps its float64 value, never pathops' float32 copy.
    for x, y in outputs:
        nearest = min(math.hypot(x - ix, y - iy) for ix, iy in inputs)
        assert nearest == 0.0 or nearest > 1e-3, (x, y)


def test_glyph_without_overlaps_takes_fast_path() -> None:
    _, flat = _flat(UBUNTU, "o")

    assert remove_overlaps_compatible(flat) is flat


def test_disjoint_boxes_take_fast_path() -> None:
    flat = _two_box_glyph(
        ((0, 0, 100, 100), (200, 0, 300, 100)),
        ((0, 0, 120, 100), (220, 0, 320, 100)),
    )

    assert remove_overlaps_compatible(flat) is flat


def test_overlap_shrinking_to_touch_never_raises() -> None:
    flat = _two_box_glyph(
        ((0, 0, 100, 100), (60, 20, 160, 80)),
        ((0, 0, 100, 100), (100, 20, 200, 80)),
    )

    result = remove_overlaps_compatible(flat)

    if result is not None:
        assert len(result.default.contours) == 1
        assert [_shape(m) for m in result.masters] == [_shape(result.default)]


def test_overlap_pulled_apart_is_rejected() -> None:
    # The master's crossings land on the extended edge lines at x=100, so the replay
    # bridges the 100-unit gap with a rectangle that keeps the outline's area sign.
    flat = _two_box_glyph(
        ((0, 0, 100, 100), (60, 20, 160, 80)),
        ((0, 0, 100, 100), (200, 20, 300, 80)),
    )

    assert remove_overlaps_compatible(flat) is None


def test_crossing_sliding_onto_neighbouring_chord_is_kept() -> None:
    upm, flat = _flat(INTER, "R")

    result = remove_overlaps_compatible(flat)

    assert result is not None
    assert all(_islands(master, upm) == 1 for master in result.masters)
