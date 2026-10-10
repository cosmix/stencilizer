"""Bridge gaps per master: the line pairs that form each bridge, the ink they cut, the gap.

A bridge is a disjoint pair of same-axis bridge lines. In every master its two lines go to
the pair's centre -/+ half a gap: the default master's gap in fixed mode, and that gap
scaled by the stroke the bridge cuts in proportional mode. The stroke is measured as the
ink along the pair's centre line through the merged (flattened, overlap-removed) polygon.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stencilizer.config import BridgeConfig, BridgeWidthScaling
from stencilizer.config.settings import REFERENCE_STROKE_FRACTION
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.realign import Extent, Placed, Span, contour_spans, extents_overlap

if TYPE_CHECKING:
    from stencilizer.variable.replay import BridgeLine, SurgeryMap


@dataclass(frozen=True, slots=True)
class BridgePair:
    """Two bridge lines (indices into ``SurgeryMap.lines``) that form one bridge.

    ``base`` is their gap in the default output; ``default_ink`` the ink along their
    centre line in the default, or None when the gap does not follow the stroke.
    """

    lower: int
    upper: int
    base: float
    default_ink: float | None


@dataclass(frozen=True, slots=True)
class WidthRule:
    """The bridges of a surgery map and how their gaps are sized in every master."""

    pairs: tuple[BridgePair, ...]
    bridge: BridgeConfig
    upm: int


def scaled_gap(base: float, ratio: float, bridge: BridgeConfig, upm: int) -> float:
    """A bridge's gap in one master from its default gap and its stroke ratio there.

    Fixed mode keeps ``base``. Proportional mode gives
    ``max(minimum, base * ratio ** (scaling_strength / 100))``, the minimum being
    ``min_width_percent`` of the reference stroke and never above ``base``.
    """
    if bridge.width_scaling == BridgeWidthScaling.FIXED:
        return base
    minimum = min(bridge.min_width_percent / 100.0 * REFERENCE_STROKE_FRACTION * upm, base)
    return max(minimum, base * math.pow(ratio, bridge.scaling_strength / 100.0))


def stroke_ratio(master_ink: float, default_ink: float) -> float:
    """Master ink over default ink; 1 when the default has no ink."""
    return master_ink / default_ink if default_ink > 0.0 else 1.0


def ink(glyph: Glyph, axis: int, centre: float, extent: Extent) -> float:
    """Length of ``glyph``'s fill along the line ``axis == centre``, clipped to ``extent``.

    Every edge whose ends straddle the line (half-open, so an edge lying on it is
    skipped) crosses it once; the sorted crossings pair up even-odd into ink intervals.
    """
    other = 1 - axis
    crossings: list[float] = []
    for contour in glyph.contours:
        points = [(point.x, point.y) for point in contour.points]
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            if (a[axis] <= centre < b[axis]) or (b[axis] <= centre < a[axis]):
                t = (centre - a[axis]) / (b[axis] - a[axis])
                crossings.append(a[other] + t * (b[other] - a[other]))
    crossings.sort()
    low, high = extent
    intervals = zip(crossings[::2], crossings[1::2], strict=True)
    return sum(max(0.0, min(high, end) - max(low, start)) for start, end in intervals)


def _positions(glyph: Glyph) -> Placed:
    return [[[point.x, point.y] for point in contour.points] for contour in glyph.contours]


def _extent(positions: Placed, lines: Iterable["BridgeLine"]) -> Extent:
    """Cross-axis range of the lines' points."""
    values = [positions[ci][pi][1 - line.axis] for line in lines for ci, pi in line.slots]
    return min(values), max(values)


def _contours(line: "BridgeLine", spans: Sequence[Span]) -> frozenset[int]:
    """The input contours (by first point index) the line's points lie on."""
    return frozenset(spans[member.edges[0]][0] for member in line.members)


def disjoint_pairs(
    lines: Sequence["BridgeLine"], input_default: Glyph, output_default: Glyph
) -> list[tuple[int, int]]:
    """(lower, upper) line indices of every bridge, each line in one pair at most.

    Walking each axis's lines upward from the lowest default coordinate, a line pairs
    with the nearest unpaired line above it on the same input contours whose points
    overlap its own along the other axis. Lines left over form no bridge.
    """
    spans = contour_spans(input_default)
    positions = _positions(output_default)
    contours = [_contours(line, spans) for line in lines]
    extents = [_extent(positions, (line,)) for line in lines]
    order = sorted(range(len(lines)), key=lambda i: (lines[i].axis, lines[i].coordinate))
    pairs: list[tuple[int, int]] = []
    paired: set[int] = set()
    for k, low in enumerate(order):
        if low in paired:
            continue
        for high in order[k + 1 :]:
            if lines[high].axis != lines[low].axis:
                break
            if high in paired or contours[high] != contours[low]:
                continue
            if extents_overlap(extents[low], extents[high]):
                pairs.append((low, high))
                paired.update((low, high))
                break
    return pairs


def width_rule(
    smap: "SurgeryMap", input_default: Glyph, output_default: Glyph, bridge: BridgeConfig, upm: int
) -> WidthRule:
    """The bridges of ``smap`` sized by ``bridge``'s mode, measured on the default.

    ``input_default`` is the merged polygon surgery ran on and ``output_default`` the
    default output. A pair's default centre is the mean of its two line coordinates.
    """
    positions = _positions(output_default)
    pairs: list[BridgePair] = []
    for low, high in disjoint_pairs(smap.lines, input_default, output_default):
        lower, upper = smap.lines[low], smap.lines[high]
        default_ink = None
        if bridge.width_scaling == BridgeWidthScaling.PROPORTIONAL:
            centre = (lower.coordinate + upper.coordinate) / 2.0
            extent = _extent(positions, (lower, upper))
            default_ink = ink(input_default, lower.axis, centre, extent)
        pairs.append(BridgePair(low, high, upper.coordinate - lower.coordinate, default_ink))
    return WidthRule(tuple(pairs), bridge, upm)


def pair_targets(
    rule: WidthRule,
    lines: Sequence["BridgeLine"],
    placed: Placed,
    master: Glyph,
    means: Sequence[float],
) -> list[float]:
    """Every line's target in one master: its pair's centre -/+ half the gap, else its mean.

    ``placed`` holds the master's points with cut points at their default edge parameter,
    ``means`` each line's mean cut coordinate there, and ``master`` the master's merged
    polygon. A pair's centre is the mean of its two lines' means; the ink is clipped to
    the cross-axis range of both lines' points.
    """
    targets = list(means)
    for pair in rule.pairs:
        lower, upper = lines[pair.lower], lines[pair.upper]
        centre = (means[pair.lower] + means[pair.upper]) / 2.0
        ratio = 1.0
        if pair.default_ink is not None:
            master_ink = ink(master, lower.axis, centre, _extent(placed, (lower, upper)))
            ratio = stroke_ratio(master_ink, pair.default_ink)
        half = scaled_gap(pair.base, ratio, rule.bridge, rule.upm) / 2.0
        targets[pair.lower], targets[pair.upper] = centre - half, centre + half
    return targets


def fallback_steps(bridge: BridgeConfig) -> tuple[BridgeWidthScaling | None, ...]:
    """The width modes to try in order; None stands for per-line mean targets, unpaired.

    The configured mode comes first, then fixed, then the mean targets; fixed mode starts
    at the second step.
    """
    if bridge.width_scaling == BridgeWidthScaling.FIXED:
        return (BridgeWidthScaling.FIXED, None)
    return (bridge.width_scaling, BridgeWidthScaling.FIXED, None)
