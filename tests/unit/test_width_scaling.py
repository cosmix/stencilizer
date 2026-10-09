"""Unit tests for variable bridge width scaling: gap formula, ink, pairing and fallback."""

from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from statistics import fmean

import pytest
from pydantic import ValidationError

from stencilizer.config import BridgeConfig, BridgeWidthScaling, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain.glyph import Glyph
from stencilizer.io import FontReader
from stencilizer.variable.align import align_to_lines, snap_distance
from stencilizer.variable.bridge_width import (
    disjoint_pairs,
    fallback_steps,
    ink,
    scaled_gap,
    stroke_ratio,
    width_rule,
)
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.processing import variable_island_counts
from stencilizer.variable.reader import read_variable_glyph
from stencilizer.variable.realign import contour_spans
from stencilizer.variable.replay import BridgeLine, SurgeryMap, map_surgery, replay, slot_values
from stencilizer.variable.transform import transform_variable_glyph
from tests.font_helpers import CANTARELL, INTER, UBUNTU
from tests.unit._variable_cases import glyph_from_outlines, read_variable

FIXED = BridgeWidthScaling.FIXED
PROPORTIONAL = BridgeWidthScaling.PROPORTIONAL
OUTER = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0)]
INTER_GAP = 122.88  # 60% of a 204.8-unit reference stroke at 2048 UPM


def _counter(low: float, high: float) -> list[tuple[float, float]]:
    return [(low, low), (high, low), (high, high), (low, high)]


def _one_bridge(glyphs: str) -> dict[str, tuple[int, int]]:
    return dict.fromkeys(glyphs.split(), (1, 0))


# Per glyph (bridge_count, unbridged_count) with BridgeConfig() before width scaling.
BASELINE: dict[str, dict[str, tuple[int, int]]] = {
    INTER.name: {
        ".notdef": (7, 0),
        "B": (2, 0),
        "eight": (2, 0),
        "ampersand": (0, 2),
        **_one_bridge("A D O P R a b d e g o p q zero four six nine"),
    },
    UBUNTU.name: {
        "B": (2, 0),
        "eight": (2, 0),
        "ampersand": (0, 2),
        **_one_bridge(".notdef O A D P R a b d e g o p q zero four six nine"),
    },
    CANTARELL.name: {
        "B": (2, 0),
        "eight": (2, 0),
        "ampersand": (0, 2),
        **_one_bridge("A Aacute D O P R a b d e g o p q zero four six nine"),
    },
}


@dataclass(frozen=True)
class _Surgery:
    """A glyph's merged masters, its default surgery map and the aligned default output."""

    vg: VariableGlyph
    merged: VariableGlyph
    smap: SurgeryMap
    default: Glyph
    upm: int


def _surgery(path: Path, char: str, bridge: BridgeConfig) -> _Surgery:
    vg, upm = read_variable(path, char)
    merged = remove_overlaps_compatible(flatten_compatible(vg, upm))
    assert merged is not None
    transformer = GlyphTransformer(GlyphAnalyzer(), bridge_config=bridge)
    outcome = transformer.transform_with_outcome(merged.default, upm=upm)
    assert outcome.bridge_count >= 1
    snap = snap_distance(GeometryConfig(), upm)
    smap = map_surgery(merged.default, outcome.glyph, snap)
    assert smap is not None
    default = align_to_lines(smap, merged.default, outcome.glyph, snap=snap)
    assert default is not None
    return _Surgery(vg, merged, smap, default, upm)


def _gap(glyph: Glyph, low: BridgeLine, high: BridgeLine) -> float:
    def value(line: BridgeLine) -> float:
        return fmean(slot_values(glyph, [member.slot for member in line.members], line.axis))

    return value(high) - value(low)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.5])
def test_fixed_gap_is_the_base(ratio: float) -> None:
    assert scaled_gap(60.0, ratio, BridgeConfig(), 1000) == 60.0


@pytest.mark.parametrize(("strength", "expected"), [(0, 60.0), (50, 60.0 * 1.5**0.5), (100, 90.0)])
def test_proportional_gap_follows_the_ratio(strength: float, expected: float) -> None:
    bridge = BridgeConfig(width_scaling=PROPORTIONAL, scaling_strength=strength)
    assert scaled_gap(60.0, 1.5, bridge, 1000) == pytest.approx(expected)


@pytest.mark.parametrize(("minimum", "expected"), [(30, 30.0), (40, 40.0), (10, 15.0)])
def test_proportional_gap_clamps_to_the_minimum(minimum: float, expected: float) -> None:
    bridge = BridgeConfig(width_scaling=PROPORTIONAL, min_width_percent=minimum)
    assert scaled_gap(60.0, 0.25, bridge, 1000) == pytest.approx(expected)


def test_minimum_never_exceeds_the_base() -> None:
    bridge = BridgeConfig(width_scaling=PROPORTIONAL, min_width_percent=110)
    assert scaled_gap(20.0, 0.1, bridge, 1000) == pytest.approx(20.0)
    assert scaled_gap(20.0, 3.0, bridge, 1000) == pytest.approx(60.0)


def test_zero_master_ink_gives_the_minimum_unless_strength_is_zero() -> None:
    assert stroke_ratio(0.0, 400.0) == 0.0
    full = BridgeConfig(width_scaling=PROPORTIONAL)
    assert scaled_gap(60.0, 0.0, full, 1000) == pytest.approx(30.0)
    none = BridgeConfig(width_scaling=PROPORTIONAL, scaling_strength=0)
    assert scaled_gap(60.0, 0.0, none, 1000) == pytest.approx(60.0)


def test_no_default_ink_gives_ratio_one() -> None:
    assert stroke_ratio(250.0, 0.0) == 1.0
    assert stroke_ratio(600.0, 400.0) == pytest.approx(1.5)


def test_settings_reject_out_of_range_values() -> None:
    with pytest.raises(ValidationError, match="scaling_strength"):
        BridgeConfig(scaling_strength=101)
    with pytest.raises(ValidationError, match="min_width_percent"):
        BridgeConfig(min_width_percent=9)


@pytest.mark.parametrize(
    ("counter", "extent", "expected"),
    [
        ((200.0, 800.0), (0.0, 1000.0), 400.0),
        ((300.0, 700.0), (0.0, 1000.0), 600.0),
        ((50.0, 950.0), (0.0, 1000.0), 100.0),
        ((200.0, 800.0), (100.0, 900.0), 200.0),
        ((200.0, 800.0), (300.0, 700.0), 0.0),
    ],
)
def test_ink_on_the_square_ring(
    counter: tuple[float, float], extent: tuple[float, float], expected: float
) -> None:
    ring = glyph_from_outlines(OUTER, _counter(*counter))
    assert ink(ring, 0, 500.0, extent) == pytest.approx(expected)
    assert ink(ring, 1, 500.0, extent) == pytest.approx(expected)


def test_fallback_steps_follow_the_configured_mode() -> None:
    assert fallback_steps(BridgeConfig()) == (FIXED, None)
    assert fallback_steps(BridgeConfig(width_scaling=PROPORTIONAL)) == (PROPORTIONAL, FIXED, None)


@pytest.mark.parametrize(("peak", "expected", "tolerance"), [(-1.0, 0.27, 0.03), (1.0, 1.88, 0.05)])
def test_inter_o_vertical_ratio_follows_the_stroke(
    peak: float, expected: float, tolerance: float
) -> None:
    bridge = BridgeConfig(
        direction=BridgeDirection.VERTICAL, width_scaling=PROPORTIONAL, min_width_percent=10
    )
    surgery = _surgery(INTER, "o", bridge)
    rule = width_rule(surgery.smap, surgery.merged.default, surgery.default, bridge, surgery.upm)
    assert len(rule.pairs) == 1
    pair = rule.pairs[0]
    index = [support.peak() for support in surgery.merged.supports].index({"wght": peak})
    smap = replace(surgery.smap, widths=rule)
    master = replay(smap, surgery.merged.default, surgery.default, surgery.merged.masters[index])
    assert master is not None
    lines = surgery.smap.lines
    ratio = _gap(master, lines[pair.lower], lines[pair.upper]) / pair.base
    assert ratio == pytest.approx(expected, abs=tolerance)


def test_inter_b_horizontal_bridges_pair_disjointly() -> None:
    bridge = BridgeConfig(direction=BridgeDirection.HORIZONTAL)
    surgery = _surgery(INTER, "B", bridge)
    lines = surgery.smap.lines
    pairs = disjoint_pairs(lines, surgery.merged.default, surgery.default)
    assert len(pairs) == 2
    assert len({index for pair in pairs for index in pair}) == 4
    for low, high in pairs:
        assert lines[high].coordinate - lines[low].coordinate == pytest.approx(INTER_GAP)
    outcome = transform_variable_glyph(surgery.vg, bridge, GeometryConfig(), surgery.upm)
    assert outcome.bridge_count == 2
    for master in outcome.glyph.masters:
        for low, high in pairs:
            assert _gap(master, lines[low], lines[high]) == pytest.approx(INTER_GAP, abs=1.0)


@pytest.mark.parametrize(
    ("direction", "contour_counts"),
    [(BridgeDirection.HORIZONTAL, [2, 2]), (BridgeDirection.AUTO, [3])],
)
def test_spanning_pairs_share_contour_set(
    direction: BridgeDirection, contour_counts: list[int]
) -> None:
    # With horizontal bridges Inter's eight gets one bridge per counter; with the
    # analyzer's direction it gets one bridge spanning both counters.
    bridge = BridgeConfig(use_spanning_bridges=True, direction=direction)
    surgery = _surgery(INTER, "8", bridge)
    lines = surgery.smap.lines
    pairs = disjoint_pairs(lines, surgery.merged.default, surgery.default)
    assert len({index for pair in pairs for index in pair}) == 2 * len(pairs)
    spans = contour_spans(surgery.merged.default)
    contour_sets = []
    for low, high in pairs:
        sets = [{spans[m.edges[0]][0] for m in lines[i].members} for i in (low, high)]
        assert sets[0] == sets[1]
        contour_sets.append(sets[0])
    assert sorted(len(contours) for contours in contour_sets) == contour_counts
    outcome = transform_variable_glyph(surgery.vg, bridge, GeometryConfig(), surgery.upm)
    assert outcome.bridge_count >= 1
    for low, high in pairs:
        default_gap = _gap(outcome.glyph.default, lines[low], lines[high])
        for master in outcome.glyph.masters:
            assert _gap(master, lines[low], lines[high]) == pytest.approx(default_gap, abs=1.0)


@pytest.mark.parametrize("char", ["o", "B", "8"])
def test_modes_share_the_default_master(char: str) -> None:
    vg, upm = read_variable(INTER, char)
    fixed = transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    scaled = transform_variable_glyph(
        vg, BridgeConfig(width_scaling=PROPORTIONAL), GeometryConfig(), upm
    )
    assert fixed.bridge_count >= 1
    assert scaled.bridge_count == fixed.bridge_count
    assert scaled.glyph.default.to_dict() == fixed.glyph.default.to_dict()
    assert scaled.glyph.masters != fixed.glyph.masters


@cache
def _outcomes(path: Path, scaling: BridgeWidthScaling) -> dict[str, tuple[int, int]]:
    """(bridge_count, unbridged_count) of every island glyph of a fixture font."""
    bridge = BridgeConfig(width_scaling=scaling)
    outcomes: dict[str, tuple[int, int]] = {}
    with FontReader(path) as reader:
        upm = reader.units_per_em
        for name, _ in variable_island_counts(reader):
            vg = read_variable_glyph(reader.font, name)
            assert vg is not None
            outcome = transform_variable_glyph(vg, bridge, GeometryConfig(), upm)
            outcomes[name] = (outcome.bridge_count, outcome.unbridged_count)
    return outcomes


@pytest.mark.parametrize("scaling", [FIXED, PROPORTIONAL])
@pytest.mark.parametrize("path", [INTER, UBUNTU, CANTARELL], ids=["inter", "ubuntu", "cantarell"])
def test_fixture_glyphs_keep_baseline_outcome(path: Path, scaling: BridgeWidthScaling) -> None:
    outcomes = _outcomes(path, scaling)
    assert outcomes == BASELINE[path.name]
    bridged = {name for name, (bridges, _) in outcomes.items() if bridges}
    fixed = _outcomes(path, FIXED)
    assert {name for name, (bridges, _) in fixed.items() if bridges} <= bridged
