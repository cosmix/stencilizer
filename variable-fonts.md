# Variable Font Support Implementation Plan

## Executive Summary

This document details the implementation plan for adding full variable font support to stencilizer. Unlike the CFF2 plan which explicitly rejected variable fonts, this plan **preserves variation data** in the output, producing a fully functional variable font with stencil bridges that interpolate correctly across the design space.

**Scope**: Both TrueType variable fonts (glyf + gvar) and CFF2 variable fonts (CFF2 + ItemVariationStore).

**Strategy**: Conservative union - add bridges wherever an island exists in ANY master, ensuring the stencil works at all design space positions.

---

## 1. Technical Background

### 1.1 Variable Font Tables

| Table  | Purpose                                                       | Format        |
| ------ | ------------------------------------------------------------- | ------------- |
| `fvar` | Defines variation axes (wght, wdth, etc.) and named instances | Both          |
| `avar` | Axis value mapping (user space to normalized)                 | Both          |
| `STAT` | Style attributes for UI presentation                          | Both          |
| `gvar` | Per-point coordinate deltas                                   | TrueType only |
| `HVAR` | Horizontal metrics variations                                 | Both          |
| `VVAR` | Vertical metrics variations                                   | Both          |
| `MVAR` | Font-wide metrics variations                                  | Both          |
| `CFF2` | Charstrings with blend operators                              | CFF2 only     |

### 1.2 Critical Constraint: Point Correspondence

From the [OpenType gvar specification](https://learn.microsoft.com/en-us/typography/opentype/spec/gvar):

> "Deltas for positions of points of a 'glyf' table are stored in a 'gvar' table"

**This means:**

- Point N in master A MUST correspond to point N in master B
- If we insert 4 bridge points after index 5, ALL masters must have those points at the same indices
- Coordinate VALUES differ per master; point STRUCTURE must be identical

### 1.3 The Island Problem Across Masters

Consider the letter "P" across a Weight axis:

- **Light (wght=100)**: Small counter fully enclosed → IS an island
- **Bold (wght=900)**: Larger counter opens at edges → NOT an island
- **Result**: Island status changes across the design space

**Conservative Union Solution**: Add bridge wherever island exists in ANY master. For masters where the contour isn't actually an island, insert "zero-width" bridge points (same structure, minimal visual impact).

---

## 2. Domain Model Architecture

### 2.1 New File: `src/stencilizer/domain/variable.py`

```python
"""Variable font domain models for stencilizer.

Extends core domain models to support variable fonts with multiple masters
and interpolation deltas. Preserves full variation data for round-trip processing.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator

from stencilizer.domain.contour import Point, PointType, WindingDirection, Contour
from stencilizer.domain.glyph import Glyph, GlyphMetadata


class VariableFontFormat(Enum):
    """Variable font outline format."""
    TRUETYPE_GVAR = auto()  # TrueType outlines with gvar table
    CFF2_VARIABLE = auto()  # CFF2 outlines with blend operators


@dataclass(frozen=True, slots=True)
class AxisDefinition:
    """Definition of a variation axis from fvar table."""
    tag: str           # e.g., "wght", "wdth", "slnt"
    name: str          # Human-readable name
    min_value: float
    default_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class MasterLocation:
    """A location in design space (hashable for dict keys).

    Coordinates are normalized (-1.0 to 1.0) or user-space values.
    """
    coordinates: tuple[tuple[str, float], ...]  # ((axis_tag, value), ...)

    @classmethod
    def from_dict(cls, coords: dict[str, float]) -> "MasterLocation":
        return cls(coordinates=tuple(sorted(coords.items())))

    def to_dict(self) -> dict[str, float]:
        return dict(self.coordinates)

    @classmethod
    def default(cls) -> "MasterLocation":
        return cls(coordinates=())


@dataclass(frozen=True, slots=True)
class PointDelta:
    """Coordinate delta for a point at a specific variation region."""
    dx: float
    dy: float

    @classmethod
    def zero(cls) -> "PointDelta":
        return cls(dx=0.0, dy=0.0)


@dataclass(frozen=True, slots=True)
class VariationRegion:
    """A region in variation space where deltas apply.

    Each axis has (min, peak, max) defining the region's influence.
    """
    axes: tuple[tuple[str, float, float, float], ...]  # ((tag, min, peak, max), ...)


@dataclass(frozen=True, slots=True)
class VariablePoint:
    """A point with per-region deltas for interpolation.

    Stores default master coordinates plus deltas for each variation region.
    Immutable for parallel processing safety.
    """
    x: float  # Default master coordinates
    y: float
    point_type: PointType
    deltas: tuple[tuple[VariationRegion, PointDelta], ...]  # Immutable

    def at_location(self, location: dict[str, float]) -> Point:
        """Get interpolated point at a design space location."""
        dx, dy = 0.0, 0.0
        for region, delta in self.deltas:
            scalar = _compute_region_scalar(region, location)
            dx += delta.dx * scalar
            dy += delta.dy * scalar
        return Point(self.x + dx, self.y + dy, self.point_type)

    def to_static(self) -> Point:
        """Convert to static Point (default master only)."""
        return Point(self.x, self.y, self.point_type)

    @classmethod
    def from_static(cls, point: Point) -> "VariablePoint":
        """Create from static Point (no deltas)."""
        return cls(x=point.x, y=point.y, point_type=point.point_type, deltas=())

    def with_added_delta(self, region: VariationRegion, delta: PointDelta) -> "VariablePoint":
        """Create new VariablePoint with an additional delta."""
        new_deltas = list(self.deltas) + [(region, delta)]
        return VariablePoint(x=self.x, y=self.y, point_type=self.point_type,
                            deltas=tuple(new_deltas))


@dataclass
class VariableContour:
    """A contour with consistent topology across all masters.

    Point count and types are identical across masters; only coordinates vary.
    """
    points: list[VariablePoint]
    direction: WindingDirection | None = None

    def at_location(self, location: dict[str, float]) -> Contour:
        """Get static contour at a design space location."""
        return Contour(
            points=[p.at_location(location) for p in self.points],
            direction=self.direction
        )

    def to_static(self) -> Contour:
        """Convert to static Contour (default master)."""
        return Contour(
            points=[p.to_static() for p in self.points],
            direction=self.direction
        )

    @classmethod
    def from_static(cls, contour: Contour) -> "VariableContour":
        """Create from static Contour (no deltas)."""
        return cls(
            points=[VariablePoint.from_static(p) for p in contour.points],
            direction=contour.direction
        )


@dataclass
class MasterIslandInfo:
    """Island analysis results for a specific master location."""
    location: MasterLocation
    island_contour_indices: list[int]
    containment: dict[int, int]  # island_idx -> outer_idx


@dataclass
class VariableGlyph:
    """A variable font glyph with full master support.

    Stores contours with per-region deltas and tracks island status per master.
    """
    metadata: GlyphMetadata
    contours: list[VariableContour]
    axes: list[AxisDefinition]
    master_locations: list[MasterLocation]
    island_info_per_master: list[MasterIslandInfo] = field(default_factory=list)
    format: VariableFontFormat = VariableFontFormat.TRUETYPE_GVAR
    _is_composite: bool = False

    @property
    def name(self) -> str:
        return self.metadata.name

    def has_islands_in_any_master(self) -> bool:
        """Check if any master has islands (conservative union check)."""
        return any(len(info.island_contour_indices) > 0
                   for info in self.island_info_per_master)

    def get_union_island_indices(self) -> set[int]:
        """Get contour indices that are islands in ANY master."""
        union: set[int] = set()
        for info in self.island_info_per_master:
            union.update(info.island_contour_indices)
        return union

    def get_default_glyph(self) -> Glyph:
        """Get static Glyph for default master."""
        return Glyph(
            metadata=self.metadata,
            contours=[c.to_static() for c in self.contours],
            _is_composite=self._is_composite
        )

    def at_location(self, location: dict[str, float]) -> Glyph:
        """Get static Glyph at a design space location."""
        return Glyph(
            metadata=self.metadata,
            contours=[c.at_location(location) for c in self.contours],
            _is_composite=self._is_composite
        )

    @classmethod
    def from_static(cls, glyph: Glyph, axes: list[AxisDefinition] | None = None) -> "VariableGlyph":
        """Create from static Glyph (for testing/compatibility)."""
        return cls(
            metadata=glyph.metadata,
            contours=[VariableContour.from_static(c) for c in glyph.contours],
            axes=axes or [],
            master_locations=[MasterLocation.default()],
            _is_composite=glyph._is_composite
        )


@dataclass
class VariableBridgeSpec:
    """Bridge specification with per-master positioning.

    Stores point INDICES (not coordinates) to maintain correspondence.
    """
    inner_contour_idx: int
    outer_contour_idx: int
    inner_point_idx: int  # Point index on inner contour
    outer_point_idx: int  # Point index on outer contour
    width_percent: float
    masters_with_island: list[MasterLocation]  # Which masters actually have this island
    score: float = 0.0


def _compute_region_scalar(region: VariationRegion, location: dict[str, float]) -> float:
    """Compute scalar contribution of a region at a location."""
    scalar = 1.0
    for axis_tag, minimum, peak, maximum in region.axes:
        value = location.get(axis_tag, 0.0)
        if value < minimum or value > maximum:
            return 0.0
        elif value == peak:
            continue
        elif value < peak:
            if peak == minimum:
                continue
            scalar *= (value - minimum) / (peak - minimum)
        else:
            if maximum == peak:
                continue
            scalar *= (maximum - value) / (maximum - peak)
    return scalar
```

### 2.2 Update `src/stencilizer/domain/__init__.py`

Add exports for new variable font types.

---

## 3. Multi-Master Island Analysis

### 3.1 New File: `src/stencilizer/core/variable_analyzer.py`

```python
"""Multi-master glyph analysis for variable fonts."""

from dataclasses import dataclass

from stencilizer.core.analyzer import GlyphAnalyzer, ContourHierarchy
from stencilizer.domain.variable import (
    MasterLocation, VariableGlyph, MasterIslandInfo
)


@dataclass
class MultiMasterIslandAnalysis:
    """Analysis of islands across all masters.

    Attributes:
        islands_by_master: Contour indices that are islands in each master
        union_islands: Contours that are islands in ANY master (conservative)
        intersection_islands: Contours that are islands in ALL masters
    """
    islands_by_master: dict[MasterLocation, set[int]]
    union_islands: set[int]
    intersection_islands: set[int]
    containment_by_master: dict[MasterLocation, dict[int, int]]

    def needs_bridge(self, contour_idx: int) -> bool:
        """Check if contour needs bridge (island in any master)."""
        return contour_idx in self.union_islands


class MultiMasterAnalyzer:
    """Analyzes variable glyphs across all masters for islands."""

    def __init__(self) -> None:
        self._single_analyzer = GlyphAnalyzer()

    def analyze(self, var_glyph: VariableGlyph) -> MultiMasterIslandAnalysis:
        """Analyze variable glyph across all masters.

        Conservative union: island in ANY master requires a bridge.
        """
        islands_by_master: dict[MasterLocation, set[int]] = {}
        containment_by_master: dict[MasterLocation, dict[int, int]] = {}

        for location in var_glyph.master_locations:
            # Get static glyph at this master location
            glyph = var_glyph.at_location(location.to_dict())
            hierarchy = self._single_analyzer.analyze(glyph)

            islands_by_master[location] = set(hierarchy.islands)
            containment_by_master[location] = hierarchy.containment

        # Union: island in ANY master
        union_islands = set()
        for islands in islands_by_master.values():
            union_islands.update(islands)

        # Intersection: island in ALL masters
        if islands_by_master:
            intersection_islands = set.intersection(*islands_by_master.values())
        else:
            intersection_islands = set()

        return MultiMasterIslandAnalysis(
            islands_by_master=islands_by_master,
            union_islands=union_islands,
            intersection_islands=intersection_islands,
            containment_by_master=containment_by_master,
        )

    def populate_island_info(self, var_glyph: VariableGlyph) -> None:
        """Populate island_info_per_master on the glyph."""
        analysis = self.analyze(var_glyph)

        var_glyph.island_info_per_master = [
            MasterIslandInfo(
                location=location,
                island_contour_indices=list(analysis.islands_by_master[location]),
                containment=analysis.containment_by_master[location]
            )
            for location in var_glyph.master_locations
        ]
```

---

## 4. Variable Font I/O Pipeline

### 4.1 New File: `src/stencilizer/io/variable_reader.py`

```python
"""Variable font reading with full variation data extraction."""

from pathlib import Path
from typing import Iterator

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.recordingPen import RecordingPen

from stencilizer.domain.variable import (
    AxisDefinition, MasterLocation, VariableGlyph, VariableContour,
    VariablePoint, VariationRegion, PointDelta, VariableFontFormat
)
from stencilizer.io.converter import _recording_to_contours, _extract_glyph_metadata


class VariableFontReader:
    """Reads variable fonts and extracts all master data."""

    def __init__(self, font_path: Path) -> None:
        self._font_path = font_path
        self._font: TTFont | None = None
        self._axes: list[AxisDefinition] = []
        self._master_locations: list[MasterLocation] = []

    def load(self) -> None:
        """Load font and extract variation metadata."""
        self._font = TTFont(str(self._font_path))

        if "fvar" not in self._font:
            raise ValueError("Not a variable font (missing fvar)")

        self._extract_axes()
        self._compute_master_locations()

    @property
    def is_variable(self) -> bool:
        return self._font is not None and "fvar" in self._font

    @property
    def format(self) -> VariableFontFormat:
        if self._font and "CFF2" in self._font:
            return VariableFontFormat.CFF2_VARIABLE
        return VariableFontFormat.TRUETYPE_GVAR

    @property
    def units_per_em(self) -> int:
        if self._font is None:
            return 1000
        return self._font["head"].unitsPerEm

    def _extract_axes(self) -> None:
        """Extract axis definitions from fvar."""
        fvar = self._font["fvar"]
        name_table = self._font.get("name")

        self._axes = []
        for axis in fvar.axes:
            name = axis.axisTag
            if name_table:
                name_record = name_table.getName(axis.axisNameID, 3, 1, 0x409)
                if name_record:
                    name = str(name_record)

            self._axes.append(AxisDefinition(
                tag=axis.axisTag,
                name=name,
                min_value=axis.minValue,
                default_value=axis.defaultValue,
                max_value=axis.maxValue
            ))

    def _compute_master_locations(self) -> None:
        """Compute master locations from gvar/CFF2 data."""
        # Default master
        locations = [MasterLocation.default()]

        # Extract from gvar tuple variations
        if "gvar" in self._font:
            gvar = self._font["gvar"]
            seen_regions: set[tuple] = set()

            for variations in gvar.variations.values():
                for var_tuple in variations:
                    # Extract the peak of each axis as master location
                    region = tuple(
                        (tag, peak)
                        for tag, (_, peak, _) in sorted(var_tuple.axes.items())
                    )
                    if region and region not in seen_regions:
                        seen_regions.add(region)
                        loc = MasterLocation.from_dict(dict(region))
                        locations.append(loc)

        self._master_locations = locations

    def get_variable_glyph(self, name: str) -> VariableGlyph | None:
        """Extract a glyph with full variation data."""
        if self._font is None or name not in self._font.getGlyphOrder():
            return None

        if self.format == VariableFontFormat.TRUETYPE_GVAR:
            return self._read_truetype_variable_glyph(name)
        else:
            return self._read_cff2_variable_glyph(name)

    def _read_truetype_variable_glyph(self, name: str) -> VariableGlyph:
        """Read TrueType variable glyph with gvar deltas."""
        # Get base contours from default master
        glyph_set = self._font.getGlyphSet()
        pen = RecordingPen()
        glyph_set[name].draw(pen)
        base_contours = _recording_to_contours(pen.value)

        # Get gvar variations
        gvar = self._font.get("gvar")
        glyph_variations = gvar.variations.get(name, []) if gvar else []

        # Build per-point delta mapping
        point_deltas = self._extract_gvar_deltas(name, glyph_variations)

        # Convert to VariableContours
        var_contours = self._build_variable_contours(base_contours, point_deltas)

        metadata = _extract_glyph_metadata(name, self._font)

        return VariableGlyph(
            metadata=metadata,
            contours=var_contours,
            axes=self._axes,
            master_locations=self._master_locations,
            format=VariableFontFormat.TRUETYPE_GVAR
        )

    def _extract_gvar_deltas(self, name: str, variations: list) -> list[dict]:
        """Extract per-point deltas from gvar variations."""
        glyf = self._font.get("glyf")
        if not glyf or name not in glyf:
            return []

        glyph = glyf[name]
        if not hasattr(glyph, 'coordinates') or glyph.coordinates is None:
            return []

        num_points = len(glyph.coordinates)
        point_deltas: list[dict[VariationRegion, PointDelta]] = [
            {} for _ in range(num_points)
        ]

        for var_tuple in variations:
            # Build region from axes
            region = VariationRegion(axes=tuple(
                (tag, min_val, peak, max_val)
                for tag, (min_val, peak, max_val) in sorted(var_tuple.axes.items())
            ))

            # Extract deltas for each point
            for pt_idx, (dx, dy) in enumerate(var_tuple.coordinates or []):
                if pt_idx < num_points and (dx != 0 or dy != 0):
                    point_deltas[pt_idx][region] = PointDelta(dx=dx, dy=dy)

        return point_deltas

    def _build_variable_contours(
        self,
        base_contours: list,
        point_deltas: list[dict]
    ) -> list[VariableContour]:
        """Build VariableContours from base contours and deltas."""
        var_contours = []
        delta_idx = 0

        for contour in base_contours:
            var_points = []
            for point in contour.points:
                deltas = point_deltas[delta_idx] if delta_idx < len(point_deltas) else {}

                var_points.append(VariablePoint(
                    x=point.x,
                    y=point.y,
                    point_type=point.point_type,
                    deltas=tuple(deltas.items())
                ))
                delta_idx += 1

            var_contours.append(VariableContour(
                points=var_points,
                direction=contour.direction
            ))

        return var_contours

    def _read_cff2_variable_glyph(self, name: str) -> VariableGlyph:
        """Read CFF2 variable glyph with blend operators."""
        # Similar structure but extracts from CFF2 charstrings
        # Implementation follows same pattern as TrueType
        # ... (detailed implementation)
        pass

    def iter_variable_glyphs(self) -> Iterator[VariableGlyph]:
        """Iterate over all glyphs with variation data."""
        if self._font is None:
            return

        for name in self._font.getGlyphOrder():
            var_glyph = self.get_variable_glyph(name)
            if var_glyph:
                yield var_glyph

    def close(self) -> None:
        if self._font:
            self._font.close()
```

### 4.2 New File: `src/stencilizer/io/variable_writer.py`

```python
"""Variable font writing with delta regeneration."""

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_v_a_r import TupleVariation

from stencilizer.domain.variable import VariableGlyph, VariationRegion


class VariableFontWriter:
    """Writes variable fonts with updated variation data."""

    def __init__(self, font: TTFont, output_path: Path) -> None:
        self._font = font
        self._output_path = output_path

    def update_variable_glyph(self, var_glyph: VariableGlyph) -> None:
        """Update glyph with new contours and regenerated deltas."""
        if "gvar" in self._font:
            self._update_truetype_variable_glyph(var_glyph)
        elif "CFF2" in self._font:
            self._update_cff2_variable_glyph(var_glyph)

    def _update_truetype_variable_glyph(self, var_glyph: VariableGlyph) -> None:
        """Update TrueType glyph and regenerate gvar."""
        # 1. Update base glyph in glyf table
        self._update_base_glyph(var_glyph)

        # 2. Regenerate gvar variations from VariablePoint deltas
        new_variations = self._build_gvar_variations(var_glyph)

        gvar = self._font["gvar"]
        gvar.variations[var_glyph.name] = new_variations

    def _update_base_glyph(self, var_glyph: VariableGlyph) -> None:
        """Update glyf table with new base outline."""
        from fontTools.pens.ttGlyphPen import TTGlyphPen

        glyf = self._font["glyf"]
        pen = TTGlyphPen(self._font.getGlyphSet())

        for var_contour in var_glyph.contours:
            if not var_contour.points:
                continue

            # Draw default master coordinates
            points = [p.to_static() for p in var_contour.points]
            first = points[0]
            pen.moveTo((first.x, first.y))

            # ... (standard TrueType drawing logic)

            pen.closePath()

        glyf[var_glyph.name] = pen.glyph()

    def _build_gvar_variations(self, var_glyph: VariableGlyph) -> list[TupleVariation]:
        """Build gvar TupleVariation objects from VariablePoint deltas."""
        # Collect all unique regions
        regions: dict[VariationRegion, list[tuple[int, float, float]]] = {}

        point_idx = 0
        for var_contour in var_glyph.contours:
            for var_point in var_contour.points:
                for region, delta in var_point.deltas:
                    if region not in regions:
                        regions[region] = []
                    regions[region].append((point_idx, delta.dx, delta.dy))
                point_idx += 1

        # Add phantom points (4 at end)
        total_points = point_idx + 4

        # Create TupleVariation for each region
        variations = []
        for region, point_deltas in regions.items():
            axes = {
                tag: (min_val, peak, max_val)
                for tag, min_val, peak, max_val in region.axes
            }

            coordinates = [(0, 0)] * total_points
            for pt_idx, dx, dy in point_deltas:
                if pt_idx < len(coordinates):
                    coordinates[pt_idx] = (dx, dy)

            variations.append(TupleVariation(axes, coordinates))

        return variations

    def _update_cff2_variable_glyph(self, var_glyph: VariableGlyph) -> None:
        """Update CFF2 glyph with new blend operators."""
        # ... (CFF2-specific implementation)
        pass

    def save(self) -> None:
        """Save the modified font."""
        self._font.save(str(self._output_path))
```

### 4.3 New File: `src/stencilizer/core/delta_generator.py`

```python
"""Delta generation for newly inserted bridge points."""

from stencilizer.domain.variable import (
    VariablePoint, VariableContour, VariationRegion, PointDelta
)


class DeltaGenerator:
    """Generates variation deltas for new bridge points.

    Strategy: Bridge points interpolate between their endpoints.
    - Inner bridge vertices use inner contour point deltas
    - Outer bridge vertices use outer contour point deltas
    - Intermediate points blend based on position
    """

    def generate_bridge_deltas(
        self,
        inner_endpoint: VariablePoint,
        outer_endpoint: VariablePoint,
        num_bridge_vertices: int = 4
    ) -> list[dict[VariationRegion, PointDelta]]:
        """Generate deltas for bridge rectangle vertices.

        Bridge layout: [inner_left, inner_right, outer_right, outer_left]
        - Inner vertices (0, 1) use inner_endpoint deltas
        - Outer vertices (2, 3) use outer_endpoint deltas
        """
        all_regions = set(r for r, _ in inner_endpoint.deltas) | \
                      set(r for r, _ in outer_endpoint.deltas)

        vertex_deltas: list[dict[VariationRegion, PointDelta]] = [
            {} for _ in range(num_bridge_vertices)
        ]

        inner_deltas = dict(inner_endpoint.deltas)
        outer_deltas = dict(outer_endpoint.deltas)

        for region in all_regions:
            inner_d = inner_deltas.get(region, PointDelta.zero())
            outer_d = outer_deltas.get(region, PointDelta.zero())

            # Inner vertices get inner delta
            vertex_deltas[0][region] = inner_d
            vertex_deltas[1][region] = inner_d
            # Outer vertices get outer delta
            vertex_deltas[2][region] = outer_d
            vertex_deltas[3][region] = outer_d

        return vertex_deltas

    def interpolate_contour_deltas(
        self,
        original_contour: VariableContour,
        new_point_count: int,
        point_mapping: list[int | None]  # New idx -> original idx (None for new)
    ) -> list[dict[VariationRegion, PointDelta]]:
        """Generate deltas for modified contour maintaining correspondence.

        For points that map to originals: use original deltas
        For new points: interpolate from nearest neighbors
        """
        result = []

        for new_idx in range(new_point_count):
            orig_idx = point_mapping[new_idx]

            if orig_idx is not None and orig_idx < len(original_contour.points):
                # Direct mapping - use original deltas
                result.append(dict(original_contour.points[orig_idx].deltas))
            else:
                # New point - interpolate from neighbors
                result.append(self._interpolate_from_neighbors(
                    new_idx, point_mapping, original_contour
                ))

        return result

    def _interpolate_from_neighbors(
        self,
        new_idx: int,
        point_mapping: list[int | None],
        contour: VariableContour
    ) -> dict[VariationRegion, PointDelta]:
        """Interpolate deltas from nearest mapped neighbors."""
        # Find nearest points with valid mappings
        before_idx = new_idx - 1
        after_idx = new_idx + 1

        while before_idx >= 0 and point_mapping[before_idx] is None:
            before_idx -= 1
        while after_idx < len(point_mapping) and point_mapping[after_idx] is None:
            after_idx += 1

        # Get deltas from neighbors
        before_deltas = {}
        after_deltas = {}

        if before_idx >= 0 and point_mapping[before_idx] is not None:
            orig = point_mapping[before_idx]
            before_deltas = dict(contour.points[orig].deltas)

        if after_idx < len(point_mapping) and point_mapping[after_idx] is not None:
            orig = point_mapping[after_idx]
            after_deltas = dict(contour.points[orig].deltas)

        # Average the deltas
        all_regions = set(before_deltas.keys()) | set(after_deltas.keys())
        result = {}

        for region in all_regions:
            bd = before_deltas.get(region, PointDelta.zero())
            ad = after_deltas.get(region, PointDelta.zero())
            result[region] = PointDelta(
                dx=(bd.dx + ad.dx) / 2,
                dy=(bd.dy + ad.dy) / 2
            )

        return result
```

---

## 5. Multi-Master Bridge Surgery

### 5.1 New File: `src/stencilizer/core/variable_bridge.py`

```python
"""Multi-master bridge placement for variable fonts."""

import math
from dataclasses import dataclass

from stencilizer.config.settings import BridgeConfig
from stencilizer.domain.variable import (
    VariableGlyph, VariableBridgeSpec, MasterLocation
)
from stencilizer.domain import Point, PointType
from stencilizer.core.geometry import nearest_point_on_contour


class MultiMasterBridgePlacer:
    """Places bridges that work across all masters.

    Strategy: Use geometric center of island as stable anchor point.
    This position is consistent even when island shape varies significantly.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config

    def find_bridge_positions(
        self,
        var_glyph: VariableGlyph,
        island_idx: int,
        outer_idx: int
    ) -> VariableBridgeSpec | None:
        """Find bridge positions that work across all masters.

        Uses geometric center strategy for stable anchoring.
        """
        # Find bridge at default master first
        default_glyph = var_glyph.get_default_glyph()
        island = default_glyph.contours[island_idx]
        outer = default_glyph.contours[outer_idx]

        # Get island center as stable anchor
        bbox = island.bounding_box()
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Find nearest on-curve point on island to center
        inner_point_idx = self._find_nearest_oncurve_idx(island, center_x, center_y)
        if inner_point_idx is None:
            return None

        inner_point = island.points[inner_point_idx]

        # Find nearest point on outer contour
        outer_point_idx = self._find_nearest_oncurve_idx_to_point(outer, inner_point)
        if outer_point_idx is None:
            return None

        # Determine which masters actually have this island
        masters_with_island = [
            info.location
            for info in var_glyph.island_info_per_master
            if island_idx in info.island_contour_indices
        ]

        return VariableBridgeSpec(
            inner_contour_idx=island_idx,
            outer_contour_idx=outer_idx,
            inner_point_idx=inner_point_idx,
            outer_point_idx=outer_point_idx,
            width_percent=self.config.width_percent,
            masters_with_island=masters_with_island,
            score=1.0
        )

    def _find_nearest_oncurve_idx(self, contour, x: float, y: float) -> int | None:
        """Find index of nearest ON_CURVE point to (x, y)."""
        best_idx = None
        best_dist = float('inf')

        for i, pt in enumerate(contour.points):
            if pt.point_type == PointType.ON_CURVE:
                dist = math.hypot(pt.x - x, pt.y - y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        return best_idx

    def _find_nearest_oncurve_idx_to_point(self, contour, target: Point) -> int | None:
        """Find index of nearest ON_CURVE point to target point."""
        return self._find_nearest_oncurve_idx(contour, target.x, target.y)
```

### 5.2 New File: `src/stencilizer/core/variable_surgery.py`

```python
"""Multi-master contour surgery for variable fonts."""

from dataclasses import dataclass

from stencilizer.config.settings import BridgeConfig
from stencilizer.domain.variable import (
    VariableGlyph, VariableContour, VariablePoint, VariableBridgeSpec,
    MasterLocation, VariationRegion, PointDelta
)
from stencilizer.domain import Point, PointType
from stencilizer.core.delta_generator import DeltaGenerator


@dataclass
class PointInsertionPlan:
    """Plan for inserting points with master correspondence.

    All masters get points at the SAME indices, just with different
    coordinate values. This maintains the critical point correspondence
    required for variable font interpolation.
    """
    contour_idx: int
    insertion_idx: int
    # For each master: the actual point values to insert
    points_per_master: dict[MasterLocation, list[Point]]
    # For new points: their delta data
    point_deltas: list[dict[VariationRegion, PointDelta]]


class MultiMasterSurgeon:
    """Performs contour surgery across all masters simultaneously.

    Key principle: Point indices MUST match across all masters.
    When inserting bridge points, insert at the SAME index in EVERY master.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.delta_generator = DeltaGenerator()

    def apply_bridge(
        self,
        var_glyph: VariableGlyph,
        bridge_spec: VariableBridgeSpec
    ) -> VariableGlyph:
        """Apply a single bridge to a variable glyph.

        Modifies both inner (island) and outer contours to create the bridge.
        """
        # Generate insertion plans for both contours
        inner_plan = self._plan_bridge_insertion(
            var_glyph, bridge_spec, is_inner=True
        )
        outer_plan = self._plan_bridge_insertion(
            var_glyph, bridge_spec, is_inner=False
        )

        # Execute plans (modifies contours)
        var_glyph = self._execute_insertion(var_glyph, inner_plan)
        var_glyph = self._execute_insertion(var_glyph, outer_plan)

        return var_glyph

    def _plan_bridge_insertion(
        self,
        var_glyph: VariableGlyph,
        bridge_spec: VariableBridgeSpec,
        is_inner: bool
    ) -> PointInsertionPlan:
        """Plan bridge point insertion for one contour."""
        if is_inner:
            contour_idx = bridge_spec.inner_contour_idx
            point_idx = bridge_spec.inner_point_idx
        else:
            contour_idx = bridge_spec.outer_contour_idx
            point_idx = bridge_spec.outer_point_idx

        var_contour = var_glyph.contours[contour_idx]
        anchor_point = var_contour.points[point_idx]

        points_per_master: dict[MasterLocation, list[Point]] = {}

        for location in var_glyph.master_locations:
            # Get coordinates at this master
            coords = anchor_point.at_location(location.to_dict())

            # Calculate bridge width for this master
            width = self._calculate_bridge_width_at_master(
                var_glyph, bridge_spec, location
            )

            # Determine if this master actually has the island
            has_island = location in bridge_spec.masters_with_island

            if has_island:
                # Normal bridge notch points
                notch_points = self._calculate_notch_points(coords, width)
            else:
                # Zero-width bridge (points coincide)
                notch_points = self._calculate_notch_points(coords, 0.0)

            points_per_master[location] = notch_points

        # Generate deltas for new points
        point_deltas = self.delta_generator.generate_bridge_deltas(
            anchor_point, anchor_point, num_bridge_vertices=4
        )

        return PointInsertionPlan(
            contour_idx=contour_idx,
            insertion_idx=point_idx + 1,  # Insert after anchor
            points_per_master=points_per_master,
            point_deltas=point_deltas
        )

    def _calculate_bridge_width_at_master(
        self,
        var_glyph: VariableGlyph,
        bridge_spec: VariableBridgeSpec,
        location: MasterLocation
    ) -> float:
        """Calculate bridge width for a specific master."""
        glyph = var_glyph.at_location(location.to_dict())

        inner = glyph.contours[bridge_spec.inner_contour_idx]
        outer = glyph.contours[bridge_spec.outer_contour_idx]

        # Sample stroke width
        inner_pt = inner.points[bridge_spec.inner_point_idx]
        outer_pt = outer.points[bridge_spec.outer_point_idx]

        stroke_width = math.hypot(outer_pt.x - inner_pt.x, outer_pt.y - inner_pt.y)

        return (bridge_spec.width_percent / 100.0) * stroke_width

    def _calculate_notch_points(self, center: Point, width: float) -> list[Point]:
        """Calculate 4 notch points for bridge insertion."""
        half = width / 2.0
        return [
            Point(center.x - half, center.y, PointType.ON_CURVE),
            Point(center.x - half, center.y - width, PointType.ON_CURVE),
            Point(center.x + half, center.y - width, PointType.ON_CURVE),
            Point(center.x + half, center.y, PointType.ON_CURVE),
        ]

    def _execute_insertion(
        self,
        var_glyph: VariableGlyph,
        plan: PointInsertionPlan
    ) -> VariableGlyph:
        """Execute point insertion plan."""
        var_contour = var_glyph.contours[plan.contour_idx]
        new_points = list(var_contour.points)

        # Create VariablePoints from the plan
        for i, delta_dict in enumerate(plan.point_deltas):
            # Use default master coordinates
            default_location = var_glyph.master_locations[0]
            default_pt = plan.points_per_master[default_location][i]

            var_point = VariablePoint(
                x=default_pt.x,
                y=default_pt.y,
                point_type=default_pt.point_type,
                deltas=tuple(delta_dict.items())
            )

            new_points.insert(plan.insertion_idx + i, var_point)

        # Create new contour
        new_contour = VariableContour(
            points=new_points,
            direction=var_contour.direction
        )

        # Update glyph
        new_contours = list(var_glyph.contours)
        new_contours[plan.contour_idx] = new_contour

        return VariableGlyph(
            metadata=var_glyph.metadata,
            contours=new_contours,
            axes=var_glyph.axes,
            master_locations=var_glyph.master_locations,
            island_info_per_master=var_glyph.island_info_per_master,
            format=var_glyph.format,
            _is_composite=var_glyph._is_composite
        )
```

---

## 6. Integration and Processing Pipeline

### 6.1 Updates to `src/stencilizer/core/processor.py`

Add variable font detection and dispatch:

```python
def process(self, font_path: Path, output_path: Path, ...) -> ProcessingStats:
    """Process font - dispatches to static or variable pipeline."""
    font = TTFont(str(font_path))
    is_variable = "fvar" in font
    font.close()

    if is_variable:
        return self._process_variable_font(font_path, output_path, ...)
    else:
        return self._process_static_font(font_path, output_path, ...)

def _process_variable_font(self, font_path, output_path, ...) -> ProcessingStats:
    """Process variable font preserving variation data."""
    reader = VariableFontReader(font_path)
    reader.load()

    analyzer = MultiMasterAnalyzer()
    placer = MultiMasterBridgePlacer(self.config.bridge)
    surgeon = MultiMasterSurgeon(self.config.bridge)

    processed_glyphs: dict[str, VariableGlyph] = {}

    for var_glyph in reader.iter_variable_glyphs():
        # Analyze for islands across all masters
        analyzer.populate_island_info(var_glyph)

        if not var_glyph.has_islands_in_any_master():
            continue

        # Get union of all islands
        union_islands = var_glyph.get_union_island_indices()

        for island_idx in union_islands:
            # Find outer contour
            outer_idx = self._find_outer_for_island(var_glyph, island_idx)
            if outer_idx is None:
                continue

            # Place bridge
            bridge_spec = placer.find_bridge_positions(var_glyph, island_idx, outer_idx)
            if bridge_spec:
                var_glyph = surgeon.apply_bridge(var_glyph, bridge_spec)

        processed_glyphs[var_glyph.name] = var_glyph

    # Write output
    font = TTFont(str(font_path))
    writer = VariableFontWriter(font, output_path)

    for name, var_glyph in processed_glyphs.items():
        writer.update_variable_glyph(var_glyph)

    writer.save()
    reader.close()
```

### 6.2 CLI Updates (`src/stencilizer/cli/app.py`)

Add variable font options and feedback:

```python
@app.command()
def process(
    font_path: Path,
    output: Path | None = None,
    # ... existing options ...
    variable_strategy: str = typer.Option(
        "conservative",
        help="Island detection strategy for variable fonts: 'conservative' (bridge if island in ANY master) or 'intersection' (only if island in ALL masters)"
    ),
):
    """Process font to create stencil version."""
    # Detect and report variable font
    font = TTFont(str(font_path))
    is_variable = "fvar" in font

    if is_variable:
        axes = [axis.axisTag for axis in font["fvar"].axes]
        console.print(f"[bold]Variable font detected[/bold] with axes: {', '.join(axes)}")
        console.print(f"Using [cyan]{variable_strategy}[/cyan] island detection strategy")

    font.close()
    # ... rest of processing ...
```

---

## 7. Exception Handling

### 7.1 Updates to `src/stencilizer/exceptions.py`

```python
class VariableFontError(FontError):
    """Errors specific to variable font processing."""
    pass

class VariationDataError(VariableFontError):
    """Error accessing or processing variation data."""
    def __init__(self, glyph_name: str, reason: str) -> None:
        super().__init__(f"Variation error for '{glyph_name}': {reason}")

class IncompatibleMastersError(VariableFontError):
    """Masters have incompatible contour structures."""
    def __init__(self, glyph_name: str, details: str) -> None:
        super().__init__(f"Incompatible masters in '{glyph_name}': {details}")

class DeltaGenerationError(VariableFontError):
    """Error generating deltas for new points."""
    def __init__(self, glyph_name: str, point_index: int, reason: str) -> None:
        super().__init__(f"Delta generation for '{glyph_name}' point {point_index}: {reason}")
```

---

## 8. Testing Strategy

### 8.1 Test Fixtures

Create or obtain variable font test fixtures:

```bash
tests/fixtures/
├── RobotoFlex-Variable.ttf    # TrueType variable (wght, wdth)
├── SourceSans-Variable.otf    # CFF2 variable (wght)
└── create_variable_fixtures.py
```

### 8.2 Unit Tests

**File: `tests/unit/test_variable_domain.py`**

```python
class TestVariablePoint:
    def test_at_location_default(self):
        """Default location returns base coordinates."""
        vp = VariablePoint(x=100, y=200, point_type=PointType.ON_CURVE, deltas=())
        pt = vp.at_location({})
        assert pt.x == 100 and pt.y == 200

    def test_at_location_with_delta(self):
        """Deltas applied at non-default locations."""
        region = VariationRegion(axes=(("wght", 0.0, 1.0, 1.0),))
        vp = VariablePoint(
            x=100, y=200,
            point_type=PointType.ON_CURVE,
            deltas=((region, PointDelta(dx=50, dy=-30)),)
        )
        pt = vp.at_location({"wght": 1.0})
        assert pt.x == 150 and pt.y == 170

class TestMultiMasterAnalysis:
    def test_union_islands(self):
        """Union includes islands from any master."""
        # ... test setup and assertions ...

class TestDeltaGenerator:
    def test_bridge_deltas_inherit_endpoints(self):
        """Bridge vertices inherit endpoint deltas."""
        # ... test setup and assertions ...
```

### 8.3 Integration Tests

**File: `tests/integration/test_variable_fonts.py`**

```python
class TestVariableFontProcessing:
    def test_variable_font_detected(self, variable_ttf):
        """Variable fonts correctly detected."""
        reader = VariableFontReader(variable_ttf)
        reader.load()
        assert reader.is_variable

    def test_processed_font_still_variable(self, variable_ttf, tmp_path):
        """Output retains variation tables."""
        output = tmp_path / "output.ttf"
        processor = FontProcessor(StencilizerSettings())
        processor.process(variable_ttf, output)

        font = TTFont(str(output))
        assert "fvar" in font
        assert "gvar" in font or "CFF2" in font

    def test_bridges_interpolate(self, variable_ttf, tmp_path):
        """Bridge positions scale with weight."""
        # Process and verify bridge geometry at different weights
        ...
```

---

## 9. Documentation Updates

### 9.1 README.md Updates

**Font Format Table:**

```markdown
| Format                               | Extension | Outline Type               | Status             |
| ------------------------------------ | --------- | -------------------------- | ------------------ |
| TrueType                             | `.ttf`    | TrueType (`glyf` table)    | ✅ Fully supported |
| OpenType with TrueType outlines      | `.otf`    | TrueType (`glyf` table)    | ✅ Fully supported |
| OpenType with CFF outlines           | `.otf`    | PostScript (`CFF` table)   | ✅ Fully supported |
| OpenType with CFF2 outlines (static) | `.otf`    | PostScript (`CFF2` table)  | ✅ Fully supported |
| TrueType Variable                    | `.ttf`    | Variable (`glyf` + `gvar`) | ✅ Fully supported |
| OpenType Variable (CFF2)             | `.otf`    | Variable (`CFF2` + `fvar`) | ✅ Fully supported |
```

**Future Work Section:**

```markdown
## Future Work

- Per-glyph bridge customization for variable fonts
- Support for intermediate masters
- avar2 (axis-level variations) support
```

---

## 10. Implementation Phases

### Phase 1: Domain Models (Days 1-2)

- [ ] Create `src/stencilizer/domain/variable.py`
- [ ] Add all dataclasses: AxisDefinition, MasterLocation, VariablePoint, etc.
- [ ] Add serialization (to_dict/from_dict) for all types
- [ ] Update `domain/__init__.py` exports
- [ ] Unit tests for domain models

### Phase 2: Multi-Master Analysis (Days 3-4)

- [ ] Create `src/stencilizer/core/variable_analyzer.py`
- [ ] Implement `MultiMasterAnalyzer.analyze()`
- [ ] Implement `populate_island_info()`
- [ ] Unit tests for analysis

### Phase 3: Variable Font Reading - TrueType (Days 5-7)

- [ ] Create `src/stencilizer/io/variable_reader.py`
- [ ] Implement gvar delta extraction
- [ ] Build VariableGlyph from gvar data
- [ ] Integration tests with real TrueType variable fonts

### Phase 4: Delta Generation (Days 8-9)

- [ ] Create `src/stencilizer/core/delta_generator.py`
- [ ] Implement bridge point delta generation
- [ ] Implement neighbor interpolation
- [ ] Unit tests for delta generation

### Phase 5: Multi-Master Bridge Placement (Days 10-11)

- [ ] Create `src/stencilizer/core/variable_bridge.py`
- [ ] Implement geometric center strategy
- [ ] Handle partial-island cases (zero-width bridges)
- [ ] Integration tests

### Phase 6: Multi-Master Surgery (Days 12-14)

- [ ] Create `src/stencilizer/core/variable_surgery.py`
- [ ] Implement point insertion with correspondence
- [ ] Generate deltas for inserted points
- [ ] Integration tests

### Phase 7: Variable Font Writing - TrueType (Days 15-17)

- [ ] Create `src/stencilizer/io/variable_writer.py`
- [ ] Implement gvar regeneration
- [ ] Validate output fonts with fonttools
- [ ] End-to-end tests

### Phase 8: CFF2 Variable Support (Days 18-21)

- [ ] Extend reader for CFF2 blend operators
- [ ] Extend writer for CFF2 charstring generation
- [ ] Integration tests with CFF2 variable fonts

### Phase 9: Integration (Days 22-24)

- [ ] Update `FontProcessor` with variable dispatch
- [ ] Update CLI with variable font options
- [ ] Add exception types
- [ ] Full regression testing

### Phase 10: Documentation & Polish (Days 25-28)

- [ ] Update README.md
- [ ] Update CLAUDE.md
- [ ] Performance optimization
- [ ] Edge case handling

---

## 11. Files Summary

### New Files

| File                                         | Purpose                         |
| -------------------------------------------- | ------------------------------- |
| `src/stencilizer/domain/variable.py`         | Variable font domain models     |
| `src/stencilizer/io/variable_reader.py`      | Variable font loading           |
| `src/stencilizer/io/variable_writer.py`      | Variable font writing           |
| `src/stencilizer/core/variable_analyzer.py`  | Multi-master island analysis    |
| `src/stencilizer/core/variable_bridge.py`    | Multi-master bridge placement   |
| `src/stencilizer/core/variable_surgery.py`   | Multi-master contour surgery    |
| `src/stencilizer/core/delta_generator.py`    | Delta generation for new points |
| `tests/unit/test_variable_domain.py`         | Domain model tests              |
| `tests/unit/test_variable_analysis.py`       | Analysis tests                  |
| `tests/integration/test_variable_fonts.py`   | End-to-end tests                |
| `tests/fixtures/create_variable_fixtures.py` | Fixture generation              |

### Modified Files

| File                                 | Changes                   |
| ------------------------------------ | ------------------------- |
| `src/stencilizer/domain/__init__.py` | Export new types          |
| `src/stencilizer/core/processor.py`  | Variable font dispatch    |
| `src/stencilizer/cli/app.py`         | Variable font CLI options |
| `src/stencilizer/exceptions.py`      | Variable font exceptions  |
| `README.md`                          | Documentation updates     |

---

## 12. Risk Assessment

| Risk                                       | Likelihood | Impact | Mitigation                                    |
| ------------------------------------------ | ---------- | ------ | --------------------------------------------- |
| Point correspondence breaks during surgery | Medium     | High   | Extensive testing, validation checks          |
| gvar regeneration produces invalid data    | Medium     | High   | Validate with fonttools, compare before/after |
| CFF2 blend operators incorrectly parsed    | Medium     | Medium | Use fonttools APIs, extensive CFF2 tests      |
| Performance with many masters              | Low        | Medium | Lazy evaluation, parallel processing          |
| Island detection differs unexpectedly      | Low        | Medium | Conservative union ensures safety             |
| Delta interpolation produces artifacts     | Medium     | Low    | Use endpoint inheritance, test visually       |

---

## 13. Verification Checklist

After implementation, verify:

- [ ] Variable TrueType font loads without error
- [ ] Variable CFF2 font loads without error
- [ ] All masters correctly extracted
- [ ] Island detection works per-master
- [ ] Conservative union identifies all islands
- [ ] Bridges placed at valid positions
- [ ] Point indices match across all masters
- [ ] gvar deltas regenerated correctly
- [ ] CFF2 blend operators regenerated correctly
- [ ] Output font validates with fonttools
- [ ] Output font renders correctly at all instances
- [ ] Bridges interpolate smoothly across design space
- [ ] Static font processing still works (no regression)
- [ ] All tests pass
- [ ] No type errors from mypy
- [ ] No lint errors from ruff