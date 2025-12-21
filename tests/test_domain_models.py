"""Tests for domain models to verify they work correctly."""

import pytest

from stencilizer.domain import (
    BridgeGeometry,
    BridgeSpec,
    Contour,
    Glyph,
    GlyphMetadata,
    Point,
    PointType,
    WindingDirection,
)


class TestPoint:
    """Tests for Point class."""

    def test_point_creation(self) -> None:
        """Test basic point creation."""
        p = Point(100.0, 200.0)
        assert p.x == 100.0
        assert p.y == 200.0
        assert p.point_type == PointType.ON_CURVE

    def test_point_with_type(self) -> None:
        """Test point with explicit type."""
        p = Point(100.0, 200.0, PointType.OFF_CURVE_QUAD)
        assert p.point_type == PointType.OFF_CURVE_QUAD

    def test_point_to_tuple(self) -> None:
        """Test point to tuple conversion."""
        p = Point(100.0, 200.0)
        assert p.to_tuple() == (100.0, 200.0)

    def test_point_serialization(self) -> None:
        """Test point serialization and deserialization."""
        p1 = Point(100.0, 200.0, PointType.OFF_CURVE_CUBIC)
        data = p1.to_dict()
        p2 = Point.from_dict(data)

        assert p2.x == p1.x
        assert p2.y == p1.y
        assert p2.point_type == p1.point_type

    def test_point_immutable(self) -> None:
        """Test that point is immutable."""
        p = Point(100.0, 200.0)
        with pytest.raises(AttributeError):
            p.x = 300.0  # type: ignore


class TestContour:
    """Tests for Contour class."""

    def test_contour_creation(self) -> None:
        """Test basic contour creation."""
        points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        contour = Contour(points=points)
        assert len(contour.points) == 4

    def test_signed_area_counterclockwise(self) -> None:
        """Test signed area for counter-clockwise square."""
        # Counter-clockwise square
        points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        contour = Contour(points=points)
        area = contour.signed_area()
        assert area > 0  # Positive for CCW
        assert abs(area - 10000.0) < 0.1

    def test_signed_area_clockwise(self) -> None:
        """Test signed area for clockwise square."""
        # Clockwise square (reversed)
        points = [Point(0, 0), Point(0, 100), Point(100, 100), Point(100, 0)]
        contour = Contour(points=points)
        area = contour.signed_area()
        assert area < 0  # Negative for CW
        assert abs(area + 10000.0) < 0.1

    def test_bounding_box(self) -> None:
        """Test bounding box calculation."""
        points = [Point(10, 20), Point(100, 30), Point(50, 150)]
        contour = Contour(points=points)
        bbox = contour.bounding_box()
        assert bbox == (10.0, 20.0, 100.0, 150.0)

    def test_contains_point_inside(self) -> None:
        """Test point containment for point inside."""
        # Square
        points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        contour = Contour(points=points)
        assert contour.contains_point(50, 50)

    def test_contains_point_outside(self) -> None:
        """Test point containment for point outside."""
        points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        contour = Contour(points=points)
        assert not contour.contains_point(150, 50)

    def test_sample_points(self) -> None:
        """Test point sampling."""
        points = [Point(i, 0) for i in range(100)]
        contour = Contour(points=points)
        sampled = contour.sample_points(10)
        assert len(sampled) == 10

    def test_contour_serialization(self) -> None:
        """Test contour serialization and deserialization."""
        points = [Point(0, 0), Point(100, 0), Point(100, 100)]
        c1 = Contour(points=points, direction=WindingDirection.CLOCKWISE)
        data = c1.to_dict()
        c2 = Contour.from_dict(data)

        assert len(c2.points) == len(c1.points)
        assert c2.direction == c1.direction


class TestGlyphMetadata:
    """Tests for GlyphMetadata class."""

    def test_metadata_creation(self) -> None:
        """Test metadata creation."""
        meta = GlyphMetadata(
            name="A",
            unicode=65,
            advance_width=600,
            left_side_bearing=50
        )
        assert meta.name == "A"
        assert meta.unicode == 65

    def test_metadata_serialization(self) -> None:
        """Test metadata serialization."""
        m1 = GlyphMetadata(name="B", unicode=66, advance_width=700, left_side_bearing=60)
        data = m1.to_dict()
        m2 = GlyphMetadata.from_dict(data)

        assert m2.name == m1.name
        assert m2.unicode == m1.unicode
        assert m2.advance_width == m1.advance_width


class TestGlyph:
    """Tests for Glyph class."""

    def test_glyph_creation(self) -> None:
        """Test glyph creation."""
        meta = GlyphMetadata(name="O", unicode=79, advance_width=800, left_side_bearing=50)
        outer = Contour(
            points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
            direction=WindingDirection.COUNTER_CLOCKWISE
        )
        inner = Contour(
            points=[Point(25, 25), Point(25, 75), Point(75, 75), Point(75, 25)],
            direction=WindingDirection.CLOCKWISE
        )
        glyph = Glyph(metadata=meta, contours=[outer, inner])

        assert glyph.name == "O"
        assert len(glyph.contours) == 2

    def test_glyph_is_empty(self) -> None:
        """Test empty glyph detection."""
        meta = GlyphMetadata(name="space", unicode=32, advance_width=250, left_side_bearing=0)
        glyph = Glyph(metadata=meta, contours=[])
        assert glyph.is_empty()

    def test_glyph_has_islands(self) -> None:
        """Test island detection."""
        meta = GlyphMetadata(name="O", unicode=79, advance_width=800, left_side_bearing=50)
        outer = Contour(
            points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
            direction=WindingDirection.COUNTER_CLOCKWISE
        )
        inner = Contour(
            points=[Point(25, 25), Point(25, 75), Point(75, 75), Point(75, 25)],
            direction=WindingDirection.CLOCKWISE
        )
        glyph = Glyph(metadata=meta, contours=[outer, inner])

        assert glyph.has_islands()
        assert len(glyph.get_islands()) == 1
        assert len(glyph.get_outer_contours()) == 1

    def test_glyph_serialization(self) -> None:
        """Test glyph serialization."""
        meta = GlyphMetadata(name="A", unicode=65, advance_width=600, left_side_bearing=50)
        contour = Contour(
            points=[Point(0, 0), Point(100, 0), Point(50, 100)],
            direction=WindingDirection.COUNTER_CLOCKWISE
        )
        g1 = Glyph(metadata=meta, contours=[contour])
        data = g1.to_dict()
        g2 = Glyph.from_dict(data)

        assert g2.name == g1.name
        assert len(g2.contours) == len(g1.contours)


class TestBridgeSpec:
    """Tests for BridgeSpec class."""

    def test_bridge_spec_creation(self) -> None:
        """Test bridge spec creation."""
        inner_pt = Point(50, 50)
        outer_pt = Point(100, 50)
        spec = BridgeSpec(
            inner_contour_idx=1,
            outer_contour_idx=0,
            inner_point=inner_pt,
            outer_point=outer_pt,
            width=30.0,
            score=85.0
        )
        assert spec.inner_contour_idx == 1
        assert spec.width == 30.0

    def test_bridge_spec_serialization(self) -> None:
        """Test bridge spec serialization."""
        s1 = BridgeSpec(
            inner_contour_idx=1,
            outer_contour_idx=0,
            inner_point=Point(50, 50),
            outer_point=Point(100, 50),
            width=30.0,
            score=85.0
        )
        data = s1.to_dict()
        s2 = BridgeSpec.from_dict(data)

        assert s2.inner_contour_idx == s1.inner_contour_idx
        assert s2.width == s1.width
        assert s2.score == s1.score


class TestBridgeGeometry:
    """Tests for BridgeGeometry class."""

    def test_bridge_geometry_creation(self) -> None:
        """Test bridge geometry creation."""
        spec = BridgeSpec(
            inner_contour_idx=1,
            outer_contour_idx=0,
            inner_point=Point(50, 50),
            outer_point=Point(100, 50),
            width=30.0
        )
        vertices = (
            Point(50, 35),
            Point(100, 35),
            Point(100, 65),
            Point(50, 65)
        )
        geom = BridgeGeometry(vertices=vertices, spec=spec)
        assert len(geom.vertices) == 4

    def test_bridge_as_contour(self) -> None:
        """Test converting bridge to contour."""
        spec = BridgeSpec(
            inner_contour_idx=1,
            outer_contour_idx=0,
            inner_point=Point(50, 50),
            outer_point=Point(100, 50),
            width=30.0
        )
        vertices = (
            Point(50, 35),
            Point(100, 35),
            Point(100, 65),
            Point(50, 65)
        )
        geom = BridgeGeometry(vertices=vertices, spec=spec)
        contour = geom.as_contour()

        assert len(contour.points) == 4
        assert contour.direction == WindingDirection.COUNTER_CLOCKWISE

    def test_bridge_geometry_serialization(self) -> None:
        """Test bridge geometry serialization."""
        spec = BridgeSpec(
            inner_contour_idx=1,
            outer_contour_idx=0,
            inner_point=Point(50, 50),
            outer_point=Point(100, 50),
            width=30.0
        )
        vertices = (
            Point(50, 35),
            Point(100, 35),
            Point(100, 65),
            Point(50, 65)
        )
        g1 = BridgeGeometry(vertices=vertices, spec=spec)
        data = g1.to_dict()
        g2 = BridgeGeometry.from_dict(data)

        assert len(g2.vertices) == len(g1.vertices)
        assert g2.spec.width == g1.spec.width
