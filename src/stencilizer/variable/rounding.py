"""Round a variable glyph to the values its target format stores."""

from fontTools.misc.roundTools import otRound  # type: ignore[import-untyped]

from stencilizer.variable.model import VariableGlyph, glyph_coordinates, with_coordinates
from stencilizer.variable.solver import Coords


def round_coords(coords: Coords) -> Coords:
    """Every coordinate rounded to the nearest integer, halves up as fontTools rounds."""
    return [(otRound(x), otRound(y)) for x, y in coords]


def _combine(base: Coords, terms: list[tuple[float, Coords]]) -> Coords:
    """``base + sum(scalar * delta)`` per point."""
    result = list(base)
    for scalar, delta in terms:
        if scalar == 0.0:
            continue
        result = [
            (x + scalar * dx, y + scalar * dy)
            for (x, y), (dx, dy) in zip(result, delta, strict=True)
        ]
    return result


def _visit_order(vg: VariableGlyph) -> list[int]:
    """Supports by (axis count, then largest |peak| first)."""
    return sorted(
        range(len(vg.supports)),
        key=lambda k: (
            len(vg.supports[k].axes),
            -max((abs(peak) for _, _, peak, _ in vg.supports[k].axes), default=0.0),
        ),
    )


def _round_deltas(vg: VariableGlyph, rdef: Coords) -> list[Coords]:
    """Integer deltas chosen so every support peak lands within 0.5 of the exact outline.

    Supports are visited in order; the delta of support k absorbs the error left by the
    already-rounded supports that are non-zero at its peak.
    """
    peaks = [support.peak() for support in vg.supports]
    scalars = [[support.scalar(peak) for support in vg.supports] for peak in peaks]
    exact = vg.deltas()
    rounded: dict[int, Coords] = {}
    for k in _visit_order(vg):
        if any(scalars[k][j] != 0.0 for j in range(len(peaks)) if j != k and j not in rounded):
            rounded[k] = round_coords(exact[k])
            continue
        accumulated = _combine(rdef, [(scalars[k][j], rd) for j, rd in rounded.items()])
        target = glyph_coordinates(vg.masters[k])
        rounded[k] = round_coords(
            [(tx - ax, ty - ay) for (tx, ty), (ax, ay) in zip(target, accumulated, strict=True)]
        )
    return [rounded[k] for k in range(len(peaks))]


def round_variable_glyph(vg: VariableGlyph) -> VariableGlyph:
    """The glyph as stored: integer default and integer deltas, masters rebuilt from them.

    CFF2 gets integer deltas too: fontTools' instancer rounds the blended delta of every
    relative charstring operand, so fractional deltas make edges that must coincide on
    separate contours drift apart. Integer absolute deltas difference to integer operand
    deltas, so every master is exact under that rounding.
    """
    rdef = round_coords(glyph_coordinates(vg.default))
    default = with_coordinates(vg.default, rdef)
    if not vg.supports:
        return VariableGlyph(default, (), (), vg.axis_tags, vg.cff2)
    deltas = _round_deltas(vg, rdef)
    masters = []
    for support in vg.supports:
        peak = support.peak()
        terms = [(s.scalar(peak), d) for s, d in zip(vg.supports, deltas, strict=True)]
        masters.append(with_coordinates(vg.default, _combine(rdef, terms)))
    return VariableGlyph(default, vg.supports, tuple(masters), vg.axis_tags, vg.cff2)
