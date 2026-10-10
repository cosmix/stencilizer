"""Variable glyph model: a default outline plus one full master per variation support."""

from dataclasses import dataclass, field
from typing import Any

from fontTools.varLib.models import supportScalar  # type: ignore[import-untyped]

from stencilizer.domain.contour import Contour, Point
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable.solver import Coords, solve_deltas


@dataclass(frozen=True)
class Support:
    """One variation region: (tag, start, peak, end) per axis, sorted by tag."""

    axes: tuple[tuple[str, float, float, float], ...]

    def peak(self) -> dict[str, float]:
        """The location where this support's scalar is 1."""
        return {tag: peak for tag, _, peak, _ in self.axes}

    def scalar(self, location: dict[str, float]) -> float:
        """OpenType support scalar at a normalized location (absent axes count as 0)."""
        region = {tag: (start, peak, end) for tag, start, peak, end in self.axes}
        return float(supportScalar(location, region, ot=True))


def glyph_coordinates(glyph: Glyph) -> Coords:
    """All point coordinates of a glyph, contours in order."""
    return [(p.x, p.y) for contour in glyph.contours for p in contour.points]


def with_coordinates(template: Glyph, coords: Coords) -> Glyph:
    """A copy of ``template`` (metadata, point types) at new coordinates."""
    contours: list[Contour] = []
    index = 0
    for contour in template.contours:
        points = [
            Point(x, y, p.point_type)
            for p, (x, y) in zip(
                contour.points, coords[index : index + len(contour.points)], strict=True
            )
        ]
        index += len(contour.points)
        contours.append(Contour(points, direction=contour.direction))
    return Glyph(
        metadata=template.metadata, contours=contours, _is_composite=template._is_composite
    )


def _structure(glyph: Glyph) -> list[list[Any]]:
    return [[p.point_type for p in contour.points] for contour in glyph.contours]


@dataclass
class VariableGlyph:
    """A default glyph with ``masters[i]`` the full outline at ``supports[i].peak()``."""

    default: Glyph
    supports: tuple[Support, ...]
    masters: tuple[Glyph, ...]
    axis_tags: tuple[str, ...]
    cff2: bool = False
    _deltas: list[Coords] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.masters) != len(self.supports):
            raise VariationDataError(self.name, "incompatible master structure")
        shape = _structure(self.default)
        if any(_structure(master) != shape for master in self.masters):
            raise VariationDataError(self.name, "incompatible master structure")

    @property
    def name(self) -> str:
        return self.default.name

    def deltas(self) -> list[Coords]:
        """Per-support (dx, dy) for every point, solved once and cached."""
        if self._deltas is None:
            self._deltas = solve_deltas(
                self.supports,
                glyph_coordinates(self.default),
                [glyph_coordinates(master) for master in self.masters],
                glyph_name=self.name,
            )
        return self._deltas

    def instance(self, location: dict[str, float]) -> Glyph:
        """The outline at a normalized location."""
        coords = glyph_coordinates(self.default)
        for support, delta in zip(self.supports, self.deltas(), strict=True):
            scalar = support.scalar(location)
            if scalar == 0.0:
                continue
            coords = [
                (x + scalar * dx, y + scalar * dy)
                for (x, y), (dx, dy) in zip(coords, delta, strict=True)
            ]
        return with_coordinates(self.default, coords)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for IPC."""
        return {
            "default": self.default.to_dict(),
            "supports": [[list(axis) for axis in s.axes] for s in self.supports],
            "masters": [m.to_dict() for m in self.masters],
            "axis_tags": list(self.axis_tags),
            "cff2": self.cff2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariableGlyph":
        supports = tuple(
            Support(tuple((str(t), float(a), float(p), float(e)) for t, a, p, e in axes))
            for axes in data["supports"]
        )
        return cls(
            default=Glyph.from_dict(data["default"]),
            supports=supports,
            masters=tuple(Glyph.from_dict(m) for m in data["masters"]),
            axis_tags=tuple(data["axis_tags"]),
            cff2=bool(data.get("cff2", False)),
        )
