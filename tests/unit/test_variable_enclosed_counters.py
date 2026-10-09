"""Stenciled variable glyphs keep no enclosed counter that the island analyzer misses.

Holes are counted independently of the engine: the non-zero union of the outline through
skia-pathops, then every contour whose orientation is opposite to the union's total. The
count runs with the contours as given and reversed, as the CFF2 writer stores them, since
skia-pathops can resolve a tangle of near-coincident edges differently by direction.
"""

from pathlib import Path

import pathops  # type: ignore[import-untyped]
import pytest
from fontTools.pens.areaPen import AreaPen  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.domain.contour import Contour, Point, PointType
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.holes import enclosed_counters
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.transform import VariableOutcome, transform_variable_glyph
from stencilizer.variable.validate import validate, validation_locations
from tests.font_helpers import CANTARELL, INTER, island_count
from tests.unit._variable_cases import UPM, WGHT, Outline, read_variable
from tests.unit._variable_cases import glyph_from_outlines as _glyph

# The left half of a ring cut by a vertical bridge at x=500, and counters inside it.
LEFT_PIECE: Outline = [(0.0, 0.0), (0.0, 700.0), (500.0, 700.0), (500.0, 0.0)]
OPEN_COUNTER: Outline = [(100.0, 200.0), (500.0, 200.0), (500.0, 500.0), (100.0, 500.0)]
# The counter's cut edge leans off the bridge line: a wall of ink up to 0.5 wide closes it.
WALLED_COUNTER: Outline = [(100.0, 200.0), (499.5, 200.0), (500.0, 500.0), (100.0, 500.0)]
FULL_OUTER: Outline = [(0.0, 0.0), (0.0, 700.0), (1000.0, 700.0), (1000.0, 0.0)]
# A self-intersecting counter: one lobe is a hole, the other doubles the ink.
BOW_TIE: Outline = [(100.0, 200.0), (300.0, 500.0), (300.0, 200.0), (100.0, 500.0)]


def _union_holes(outlines: list[Outline]) -> list[float]:
    """Areas above 1 unit squared of the hole contours of the outlines' non-zero union."""
    path = pathops.Path()
    pen = path.getPen()
    for outline in outlines:
        pen.moveTo(outline[0])
        for point in outline[1:]:
            pen.lineTo(point)
        pen.closePath()
    path.simplify(fix_winding=True)
    areas = []
    for contour in path.contours:
        area_pen = AreaPen()
        contour.draw(area_pen)
        areas.append(area_pen.value)
    total = sum(areas)
    return [abs(a) for a in areas if a * total < 0 and abs(a) > 1.0]


def _holes(glyph: Glyph) -> list[float]:
    """The longer hole list of the glyph's union, contours as given or reversed."""
    outlines = []
    for contour in glyph.contours:
        assert all(point.point_type == PointType.ON_CURVE for point in contour.points)
        outlines.append([(point.x, point.y) for point in contour.points])
    reversed_outlines = [outline[::-1] for outline in outlines]
    return max(_union_holes(outlines), _union_holes(reversed_outlines), key=len)


def _grown(*outlines: Outline) -> VariableGlyph:
    """A one-axis variable glyph whose wght master is the default scaled by 1.25."""
    default = _glyph(*outlines)
    master = _glyph(*[[(x * 1.25, y * 1.25) for x, y in outline] for outline in outlines])
    return VariableGlyph(default, (WGHT,), (master,), ("wght",))


def test_open_counter_validates() -> None:
    assert validate(_grown(LEFT_PIECE, OPEN_COUNTER), UPM, allowed_islands=0)


@pytest.mark.parametrize(
    "outlines",
    [(LEFT_PIECE, WALLED_COUNTER), (FULL_OUTER, BOW_TIE)],
    ids=["hairline-wall", "bow-tie"],
)
def test_counter_the_analyzer_misses_fails_validation(outlines: tuple[Outline, ...]) -> None:
    vg = _grown(*outlines)
    assert island_count(vg.default, UPM) == 0
    assert not validate(vg, UPM, allowed_islands=0)
    assert validate(vg, UPM, allowed_islands=1)


def _assert_open(outcome: VariableOutcome) -> None:
    """No more holes at any validation location than the glyph has unbridged islands."""
    for location in validation_locations(outcome.glyph):
        holes = _holes(outcome.glyph.instance(location))
        assert len(holes) <= outcome.unbridged_count, (location, holes)


@pytest.mark.parametrize(
    ("path", "char"),
    [(CANTARELL, "B"), (CANTARELL, "6"), (CANTARELL, "8"), (CANTARELL, "9"), (INTER, "A")],
)
def test_glyph_is_bridged_and_open_at_every_validation_location(path: Path, char: str) -> None:
    vg, upm = read_variable(path, char)
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count > 0
    _assert_open(outcome)


def test_ampersand_is_open_at_every_validation_location_or_unchanged() -> None:
    vg, upm = read_variable(CANTARELL, "&")
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    if outcome.bridge_count == 0:
        assert outcome.glyph is vg
        assert outcome.unbridged_count >= 1
        return
    _assert_open(outcome)


@pytest.mark.parametrize(("path", "char"), [(CANTARELL, "B"), (INTER, "A")])
def test_overlap_built_counter_stays_bridged_and_open(path: Path, char: str) -> None:
    vg, upm = read_variable(path, char)
    outcome = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count >= 1
    assert outcome.unbridged_count == 0
    for location in validation_locations(outcome.glyph):
        assert _holes(outcome.glyph.instance(location)) == [], location


def test_unflattenable_glyph_cannot_be_counted() -> None:
    square = _glyph(FULL_OUTER)
    assert enclosed_counters(square, UPM) == 0
    assert enclosed_counters(square, 0) is None
    cubic = Glyph(
        metadata=square.metadata,
        contours=[Contour([Point(x, y, PointType.OFF_CURVE_CUBIC) for x, y in FULL_OUTER])],
    )
    assert enclosed_counters(cubic, UPM) is None
