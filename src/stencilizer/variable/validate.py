"""Where a replayed variable glyph is checked, and the checks it must pass."""

import itertools
import math
from collections.abc import Sequence
from statistics import fmean

from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.geometry import signed_area
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable.holes import enclosed_counters
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.realign import Extent, extents_overlap
from stencilizer.variable.replay import BridgeLine, slot_values

MAX_GRID_LOCATIONS = 64

Location = dict[str, float]


def _axis_values(tag: str, peaks: list[Location]) -> list[float]:
    """0, the axis ends the supports reach, and every intermediate peak coordinate."""
    coordinates = {peak.get(tag, 0.0) for peak in peaks}
    values = {0.0} | {c for c in coordinates if 0.0 < abs(c) < 1.0}
    if any(c < 0.0 for c in coordinates):
        values.add(-1.0)
    if any(c > 0.0 for c in coordinates):
        values.add(1.0)
    return sorted(values)


def _reduced_grid(tags: tuple[str, ...], values: list[list[float]]) -> list[Location]:
    """Each axis's non-zero values alone, then all axes at their minimum and at their maximum."""
    singles = [{tag: v} for tag, axis in zip(tags, values, strict=True) for v in axis if v]
    lowest = {tag: min(axis) for tag, axis in zip(tags, values, strict=True)}
    highest = {tag: max(axis) for tag, axis in zip(tags, values, strict=True)}
    return [*singles, lowest, highest]


def _normalized(location: Location, tags: tuple[str, ...]) -> Location:
    """Non-zero coordinates only, in axis order."""
    order = {tag: index for index, tag in enumerate(tags)}
    ordered = sorted(location.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))
    return {tag: value for tag, value in ordered if value != 0.0}


def validation_locations(vg: VariableGlyph) -> list[Location]:
    """The default, every support peak, and a grid over the axis ranges the supports use.

    The grid is the full product of the per-axis value sets when that has at most
    ``MAX_GRID_LOCATIONS`` locations, otherwise a reduced set that still covers each axis
    end alone and the all-minimum and all-maximum corners.
    """
    peaks = [support.peak() for support in vg.supports]
    values = [_axis_values(tag, peaks) for tag in vg.axis_tags]
    if math.prod(len(axis) for axis in values) <= MAX_GRID_LOCATIONS:
        combos = itertools.product(*values)
        grid = [dict(zip(vg.axis_tags, combo, strict=True)) for combo in combos]
    else:
        grid = _reduced_grid(vg.axis_tags, values)
    unique: dict[tuple[tuple[str, float], ...], Location] = {}
    for location in [{}, *peaks, *grid]:
        normalized = _normalized(location, vg.axis_tags)
        unique.setdefault(tuple(normalized.items()), normalized)
    return list(unique.values())


def _line_value(glyph: Glyph, line: BridgeLine) -> float:
    return fmean(slot_values(glyph, line.slots, line.axis))


def _cross_span(glyph: Glyph, line: BridgeLine) -> Extent:
    values = slot_values(glyph, line.slots, 1 - line.axis)
    return min(values), max(values)


def _facing_pairs(glyph: Glyph, lines: Sequence[BridgeLine]) -> list[tuple[int, int, float]]:
    """Pairs of same-axis lines that face each other, with the sign of their default gap.

    Two lines face each other when their cut points overlap along the other axis; the
    two sides of one bridge always do. Lines of bridges elsewhere in the glyph may cross
    harmlessly and are not paired.
    """
    pairs: list[tuple[int, int, float]] = []
    for i, j in itertools.combinations(range(len(lines)), 2):
        if lines[i].axis != lines[j].axis:
            continue
        if extents_overlap(_cross_span(glyph, lines[i]), _cross_span(glyph, lines[j])):
            gap = _line_value(glyph, lines[i]) - _line_value(glyph, lines[j])
            pairs.append((i, j, math.copysign(1.0, gap)))
    return pairs


def _piece_signs(glyph: Glyph, lines: Sequence[BridgeLine]) -> dict[int, float]:
    """Orientation sign of every contour that carries a bridge cut."""
    pieces = {ci for line in lines for ci, _ in line.slots}
    return {ci: math.copysign(1.0, signed_area(glyph.contours[ci].points)) for ci in pieces}


def _bridges_intact(
    glyph: Glyph,
    lines: Sequence[BridgeLine],
    pairs: list[tuple[int, int, float]],
    signs: dict[int, float],
) -> bool:
    """No facing line pair closes or swaps, and no bridge piece collapses or turns over."""
    for i, j, sign in pairs:
        if (_line_value(glyph, lines[i]) - _line_value(glyph, lines[j])) * sign <= 0.0:
            return False
    return all(signed_area(glyph.contours[ci].points) * s > 0.0 for ci, s in signs.items())


def validate(
    vg: VariableGlyph, upm: int, allowed_islands: int, lines: Sequence[BridgeLine] = ()
) -> bool:
    """True when every validation location keeps its bridges and at most ``allowed_islands``.

    Islands are counted by the analyzer and as the counters the fill encloses, which also
    catches bow-ties and hairline walls the analyzer misses. ``lines`` are the bridge lines
    of ``vg``'s default, as ``map_surgery`` grouped them; without them only island counts
    are checked.
    """
    try:
        vg.deltas()
    except VariationDataError:
        return False
    default = vg.instance({})
    pairs = _facing_pairs(default, lines)
    signs = _piece_signs(default, lines)
    analyzer = GlyphAnalyzer()
    for location in validation_locations(vg):
        glyph = vg.instance(location)
        if len(analyzer.analyze(glyph, upm).get_islands()) > allowed_islands:
            return False
        counters = enclosed_counters(glyph, upm)
        if counters is None or counters > allowed_islands:
            return False
        if not _bridges_intact(glyph, lines, pairs, signs):
            return False
    return True
