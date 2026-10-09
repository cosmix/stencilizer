"""Write a variable glyph back to a CFF2 charstring with ``blend`` operands."""

from typing import Any

from fontTools.cffLib import maxStackLimit  # type: ignore[import-untyped]
from fontTools.cffLib.specializer import (  # type: ignore[import-untyped]
    commandsToProgram,
    specializeCommands,
)
from fontTools.misc.psCharStrings import T2CharString  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.domain.contour import PointType
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.model import VariableGlyph, glyph_coordinates
from stencilizer.variable.reader import _cff2_supports, cff2_vsindex
from stencilizer.variable.rounding import round_variable_glyph
from stencilizer.variable.solver import Coords

Command = tuple[str, list[Any]]


def write_cff2_variable_glyph(font: TTFont, vg: VariableGlyph) -> None:
    """Replace ``vg.name``'s charstring with one that blends to every master.

    Untouched glyphs keep their original charstrings; the new one calls no subroutines.
    """
    rounded = round_variable_glyph(vg)
    vs_index = _checked_vsindex(font, rounded) if rounded.supports else None
    deltas = rounded.deltas() if rounded.supports else []
    commands = _glyph_commands(rounded.default, deltas)
    program = commandsToProgram(
        specializeCommands(
            commands, generalizeFirst=False, preserveTopology=True, maxstack=maxStackLimit
        )
    )
    if vs_index:
        program[:0] = [vs_index, "vsindex"]
    charstrings = font["CFF2"].cff.topDictIndex[0].CharStrings
    old = charstrings[vg.name]
    charstrings[vg.name] = T2CharString(
        program=program, private=old.private, globalSubrs=old.globalSubrs
    )


def _checked_vsindex(font: TTFont, vg: VariableGlyph) -> int | None:
    """The glyph's VarData index, after checking its supports match that VarData's regions.

    ``blend`` takes one delta per region in region order, so the supports must be exactly
    the regions the reader extracted, in the same order.
    """
    regions = _cff2_supports(font, vg.name)
    if [s.axes for s in regions] != [s.axes for s in vg.supports]:
        raise ValueError(
            f"{vg.name}: {len(vg.supports)} supports do not match the "
            f"{len(regions)} regions of its CFF2 VarData"
        )
    return cff2_vsindex(font, vg.name)


def _glyph_commands(default: Glyph, deltas: list[Coords]) -> list[Command]:
    """rmoveto/rlineto/rrcurveto commands for every point, contours back in CFF winding."""
    types = [p.point_type for contour in default.contours for p in contour.points]
    layers = [glyph_coordinates(default), *deltas]
    current = [(0.0, 0.0)] * len(layers)
    commands: list[Command] = []
    start = 0
    for contour in default.contours:
        count = len(contour.points)
        # Plain reversal undoes the reader's normalization, so point i keeps its meaning.
        indices = list(reversed(range(start, start + count)))
        start += count
        indices = _on_curve_first(default.name, types, indices)
        commands.extend(_contour_commands(default.name, types, indices, layers, current))
    return commands


def _on_curve_first(name: str, types: list[PointType], indices: list[int]) -> list[int]:
    """Rotate a contour to start on-curve, as every CFF ``rmoveto`` does."""
    for offset, index in enumerate(indices):
        if types[index] == PointType.ON_CURVE:
            return indices[offset:] + indices[:offset]
    if indices:
        raise ValueError(f"{name}: cubic contour has no on-curve point")
    return indices


def _contour_commands(
    name: str,
    types: list[PointType],
    indices: list[int],
    layers: list[Coords],
    current: list[tuple[float, float]],
) -> list[Command]:
    if not indices:
        return []
    commands: list[Command] = [("rmoveto", _relative(indices[0], layers, current))]
    controls: list[int] = []
    for index in indices[1:]:
        if types[index] == PointType.OFF_CURVE_CUBIC:
            controls.append(index)
        elif types[index] == PointType.ON_CURVE:
            commands.append(_segment(name, controls, index, layers, current))
            controls = []
        else:
            raise ValueError(f"{name}: unexpected {types[index]} point in a CFF2 contour")
    if controls:
        # A trailing pair of controls curves back to the contour's first point.
        commands.append(_segment(name, controls, indices[0], layers, current))
    return commands


def _segment(
    name: str,
    controls: list[int],
    end: int,
    layers: list[Coords],
    current: list[tuple[float, float]],
) -> Command:
    if not controls:
        return ("rlineto", _relative(end, layers, current))
    if len(controls) != 2:
        raise ValueError(f"{name}: cubic segment needs exactly two control points")
    return ("rrcurveto", [arg for i in (*controls, end) for arg in _relative(i, layers, current)])


def _relative(index: int, layers: list[Coords], current: list[tuple[float, float]]) -> list[Any]:
    """dx, dy from the current point for the default and every delta layer, then advance.

    Each layer is differenced on its own, so the absolute default stays integral and the
    deltas stay float (encoded as 16.16 fixed); rounding after differencing would drift.
    """
    dxs: list[float] = []
    dys: list[float] = []
    for layer, coords in enumerate(layers):
        x, y = coords[index]
        cx, cy = current[layer]
        dxs.append(x - cx)
        dys.append(y - cy)
        current[layer] = (x, y)
    return [_operand(dxs), _operand(dys)]


def _operand(values: list[float]) -> Any:
    """A plain number, or ``[default, *deltas, 1]``: one blended operand for commandsToProgram."""
    default, deltas = values[0], values[1:]
    default_operand = int(default) if float(default).is_integer() else default
    if all(delta == 0 for delta in deltas):
        return default_operand
    return [default_operand, *deltas, 1]
