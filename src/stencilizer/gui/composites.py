"""Resolve and draw composite glyphs without importing Qt."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from fontTools.misc.transform import Transform  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]

from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point
from stencilizer.io import FontReader

Affine = tuple[float, float, float, float, float, float]

MAX_COMPONENT_DEPTH = 32
MAX_COMPONENT_PARTS = 1024


@dataclass(frozen=True)
class ComponentPart:
    """One outline glyph drawn into a composite, with its composed transform."""

    base: str
    transform: Affine


@dataclass(frozen=True)
class CompositeGlyph:
    """A composite glyph that draws at least one island glyph."""

    metadata: GlyphMetadata
    parts: tuple[ComponentPart, ...]
    sources: tuple[str, ...]

    @property
    def name(self) -> str:
        """Return the composite glyph name."""
        return self.metadata.name


def component_parts(glyph_set: Any, name: str) -> tuple[ComponentPart, ...]:
    """Flatten a composite glyph's outline components and their transforms."""
    return _component_parts(glyph_set, name, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), frozenset({name}))


def _component_parts(
    glyph_set: Any, name: str, parent: Affine, chain: frozenset[str]
) -> tuple[ComponentPart, ...]:
    """Return the flattened components of ``name`` below ``parent``.

    ``chain`` holds every glyph name on the current expansion path, so a base
    that reappears there is a cycle rather than a diamond (two independent
    components sharing a base stay fine, since each keeps its own chain).
    Expansion is bounded by ``MAX_COMPONENT_DEPTH`` and ``MAX_COMPONENT_PARTS``
    so an untrusted font's component graph cannot hang the caller.
    """
    if len(chain) > MAX_COMPONENT_DEPTH:
        raise ValueError(f"composite glyph {name!r} nests components too deeply")
    pen = RecordingPen()
    glyph_set[name].draw(pen)
    if not pen.value or any(operation != "addComponent" for operation, _ in pen.value):
        return ()

    parts: list[ComponentPart] = []
    for _, (base, transform) in pen.value:
        if base in chain:
            raise ValueError(f"composite glyph {name!r} references itself through {base!r}")
        total = _compose_transform(parent, transform)
        nested = _component_parts(glyph_set, base, total, chain | {base})
        if nested:
            parts.extend(nested)
        else:
            parts.append(ComponentPart(base, total))
        if len(parts) > MAX_COMPONENT_PARTS:
            raise ValueError(f"composite glyph {name!r} expands to too many component parts")
    return tuple(parts)


def _compose_transform(parent: Affine, child: Any) -> Affine:
    """Compose a child transform beneath its parent transform."""
    xx, xy, yx, yy, dx, dy = Transform(*parent).transform(child)
    return (float(xx), float(xy), float(yx), float(yy), float(dx), float(dy))


def find_bridged_composites(
    reader: FontReader, island_names: Collection[str]
) -> tuple[CompositeGlyph, ...]:
    """Return composites that draw at least one glyph selected for bridging."""
    island_bases = set(island_names)
    glyph_set = reader.font.getGlyphSet()
    composites: list[CompositeGlyph] = []
    for name in reader.font.getGlyphOrder():
        parts = component_parts(glyph_set, name)
        sources = _island_sources(parts, island_bases)
        if not sources:
            continue
        glyph = reader.get_glyph(name)
        if glyph is not None:
            composites.append(CompositeGlyph(glyph.metadata, parts, sources))
    return tuple(composites)


def _island_sources(parts: tuple[ComponentPart, ...], islands: set[str]) -> tuple[str, ...]:
    """Return island bases in component order with duplicates removed."""
    sources: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part.base in islands and part.base not in seen:
            sources.append(part.base)
            seen.add(part.base)
    return tuple(sources)


def load_component_outlines(
    reader: FontReader, composites: Collection[CompositeGlyph]
) -> dict[str, Glyph]:
    """Load each outline glyph referenced by the supplied composites once."""
    outlines: dict[str, Glyph] = {}
    for composite in composites:
        for part in composite.parts:
            if part.base in outlines:
                continue
            glyph = reader.get_glyph(part.base)
            if glyph is None:
                raise ValueError(f"component base glyph {part.base!r} not found")
            outlines[part.base] = glyph
    return outlines


def compose(composite: CompositeGlyph, outlines: Mapping[str, Glyph]) -> Glyph:
    """Build a composite outline from its parts without mutating source glyphs."""
    contours: list[Contour] = []
    for part in composite.parts:
        transform = Transform(*part.transform)
        mirrored = part.transform[0] * part.transform[3] - part.transform[1] * part.transform[2] < 0
        contours.extend(
            _transform_contour(contour, transform, mirrored)
            for contour in outlines[part.base].contours
        )
    return Glyph(metadata=composite.metadata, contours=contours)


def _transform_contour(contour: Contour, transform: Any, mirrored: bool) -> Contour:
    """Copy a contour through a transform, preserving TrueType winding on mirrors."""
    points = []
    for point in contour.points:
        x, y = transform.transformPoint((point.x, point.y))
        points.append(Point(float(x), float(y), point.point_type))
    if mirrored:
        points.reverse()
    return Contour(points=points, direction=contour.direction)
