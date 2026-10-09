"""Write a variable TrueType glyph: new glyf outline plus rebuilt gvar tuples."""

import copy
from array import array
from dataclasses import dataclass

from fontTools.misc.roundTools import otRound  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.tables import ttProgram  # type: ignore[import-untyped]
from fontTools.ttLib.tables._g_l_y_f import Glyph as GlyfGlyph  # type: ignore[import-untyped]
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables.TupleVariation import TupleVariation  # type: ignore[import-untyped]

from stencilizer.domain.contour import PointType
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.rounding import round_coords, round_variable_glyph

_ON_CURVE_FLAG = 0x01
_PHANTOM_COUNT = 4

Phantoms = list[tuple[float, float]]
SupportKey = tuple[tuple[str, float, float, float], ...]


@dataclass
class _OldGlyph:
    """What the rewrite needs from the glyph being replaced."""

    x_min: int
    advance: int
    lsb: int
    phantoms: dict[SupportKey, Phantoms]


def _capture_phantoms(font: TTFont, name: str) -> dict[SupportKey, Phantoms]:
    """Phantom-point deltas of every original gvar tuple, with inferred deltas filled in."""
    if "gvar" not in font:
        return {}
    originals = font["gvar"].variations.get(name, [])
    if not originals:
        return {}
    coords, controls = font["glyf"]._getCoordinatesAndControls(name, font["hmtx"].metrics)
    end_pts = controls[1]
    captured: dict[SupportKey, Phantoms] = {}
    for original in originals:
        variation = copy.deepcopy(original)
        variation.calcInferredDeltas(coords, end_pts)
        key = tuple(sorted((tag, s, p, e) for tag, (s, p, e) in variation.axes.items()))
        captured[key] = [
            (0.0, 0.0) if d is None else (float(d[0]), float(d[1]))
            for d in variation.coordinates[-_PHANTOM_COUNT:]
        ]
    return captured


def _capture_old(font: TTFont, name: str) -> _OldGlyph:
    advance, lsb = font["hmtx"].metrics[name]
    old = font["glyf"][name]
    return _OldGlyph(
        x_min=int(getattr(old, "xMin", 0)),
        advance=advance,
        lsb=lsb,
        phantoms=_capture_phantoms(font, name),
    )


def _build_glyf_glyph(vg: VariableGlyph) -> GlyfGlyph:
    """A simple glyf glyph holding every domain point as one stored point."""
    points = [p for contour in vg.default.contours for p in contour.points]
    flags = array("B")
    for point in points:
        if point.point_type == PointType.ON_CURVE:
            flags.append(_ON_CURVE_FLAG)
        elif point.point_type == PointType.OFF_CURVE_QUAD:
            flags.append(0)
        else:
            raise ValueError(f"{vg.name}: cubic points cannot be stored in a glyf table")
    end_pts: list[int] = []
    total = 0
    for contour in vg.default.contours:
        total += len(contour.points)
        end_pts.append(total - 1)
    glyph = GlyfGlyph()
    glyph.numberOfContours = len(end_pts)
    glyph.coordinates = GlyphCoordinates([(otRound(p.x), otRound(p.y)) for p in points])
    glyph.flags = flags
    glyph.endPtsOfContours = end_pts
    glyph.program = ttProgram.Program()
    glyph.program.fromBytecode(b"")
    return glyph


def _replace_glyph(font: TTFont, vg: VariableGlyph, old: _OldGlyph) -> None:
    """Install the new glyph and move lsb with xMin so phantom point 1 stays put."""
    name = vg.name
    glyph = _build_glyf_glyph(vg)
    font["glyf"][name] = glyph
    glyph.recalcBounds(font["glyf"])
    font["hmtx"][name] = (old.advance, old.lsb + glyph.xMin - old.x_min)


def _build_tuples(
    font: TTFont, r: VariableGlyph, phantoms: dict[SupportKey, Phantoms]
) -> list[TupleVariation]:
    """One tuple per support from the rounded deltas, original phantom deltas kept.

    A support with no original tuple gets zero phantom deltas: the source's metrics took no
    contribution from a region it never had, so zero leaves its metric variation unchanged.
    """
    new_coords, controls = font["glyf"]._getCoordinatesAndControls(r.name, font["hmtx"].metrics)
    end_pts = controls[1]
    zero_phantoms: Phantoms = [(0.0, 0.0)] * _PHANTOM_COUNT
    tuples = []
    for support, deltas in zip(r.supports, r.deltas(), strict=True):
        tail = phantoms.get(support.axes, zero_phantoms)
        axes = {tag: (s, p, e) for tag, s, p, e in support.axes}
        variation = TupleVariation(axes, round_coords(deltas) + round_coords(tail))
        # Tolerance 0 keeps inferred deltas exact, so bridge lines stay closed.
        variation.optimize(new_coords, end_pts, tolerance=0.0)
        tuples.append(variation)
    return tuples


def _check_supports(font: TTFont, supports: tuple[Support, ...], name: str) -> None:
    if supports and "gvar" not in font:
        raise ValueError(f"{name}: glyph varies but the font has no gvar table")


def write_truetype_variable_glyph(font: TTFont, vg: VariableGlyph) -> None:
    """Replace one glyph's glyf outline and gvar tuples; other glyphs stay untouched.

    A glyph without supports has its gvar entry removed: with no variation tuples there is
    nothing to store, and a missing entry means the glyph is the same at every location.
    """
    r = round_variable_glyph(vg)
    _check_supports(font, r.supports, r.name)
    old = _capture_old(font, r.name)
    _replace_glyph(font, r, old)
    if "gvar" not in font:
        return
    if r.supports:
        font["gvar"].variations[r.name] = _build_tuples(font, r, old.phantoms)
    else:
        font["gvar"].variations.pop(r.name, None)
