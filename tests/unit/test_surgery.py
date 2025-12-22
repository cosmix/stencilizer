"""Unit tests for contour surgery module.

Tests cover:
- Creating bridge hole contours
- Glyph transformation with bridge holes
"""

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.surgery import BridgeHoleCreator, GlyphTransformer
from stencilizer.domain import (
    Contour,
    Glyph,
    GlyphMetadata,
    Point,
    WindingDirection,
)


class TestBridgeHoleCreator:
    """Tests for BridgeHoleCreator class."""

    def test_find_contour_bounds(self) -> None:
        """Test finding contour bounding box."""
        creator = BridgeHoleCreator()

        contour = Contour(
            points=[
                Point(10.0, 20.0),
                Point(50.0, 10.0),
                Point(40.0, 60.0),
            ]
        )

        min_x, min_y, max_x, max_y = creator.find_contour_bounds(contour)
        assert min_x == 10.0
        assert min_y == 10.0
        assert max_x == 50.0
        assert max_y == 60.0

    def test_merge_contours_with_bridges_vertical(self) -> None:
        """Test merging contours with vertical bridges (top + bottom)."""
        merger = BridgeHoleCreator()

        # Inner contour (hole)
        inner = Contour(
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        # Outer contour (parent)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 100.0),
                Point(0.0, 100.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        merged = merger.merge_contours_with_bridges(inner, outer, bridge_width=20.0)

        # Should create 2 merged contours (left piece + right piece)
        assert len(merged) == 2

        # Each merged contour should be CW (outer-like)
        for contour in merged:
            assert contour.direction == WindingDirection.CLOCKWISE


class TestGlyphTransformer:
    """Tests for GlyphTransformer class."""

    def test_transform_glyph_without_islands(self) -> None:
        """Test transforming a glyph with no islands returns unchanged glyph."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator)

        # Create simple glyph with only outer contour
        glyph = Glyph(
            metadata=GlyphMetadata(
                name="I", unicode=ord("I"), advance_width=100, left_side_bearing=10
            ),
            contours=[
                Contour(
                    points=[
                        Point(0.0, 0.0),
                        Point(0.0, 100.0),
                        Point(10.0, 100.0),
                        Point(10.0, 0.0),
                    ]
                )
            ],
        )

        transformed = transformer.transform(glyph, upm=1000)

        # Should have same number of contours (no islands = no bridges added)
        assert len(transformed.contours) == len(glyph.contours)
        assert transformed.metadata == glyph.metadata

    def test_transform_simple_o_shape_creates_merged_contours(self) -> None:
        """Test transforming an O-shape creates merged contours with bridge gaps."""
        analyzer = GlyphAnalyzer()
        # Use 30% bridge width so bridges fit within the 50-unit wide inner contour
        config = BridgeConfig(width_percent=30.0, min_bridges=1)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator)

        # Create O-shaped glyph: outer (CW) + inner (CCW)
        outer_points = [
            Point(0.0, 0.0),
            Point(0.0, 100.0),
            Point(100.0, 100.0),
            Point(100.0, 0.0),
        ]

        inner_points = [
            Point(25.0, 25.0),
            Point(75.0, 25.0),
            Point(75.0, 75.0),
            Point(25.0, 75.0),
        ]

        glyph = Glyph(
            metadata=GlyphMetadata(
                name="O", unicode=ord("O"), advance_width=100, left_side_bearing=0
            ),
            contours=[
                Contour(points=outer_points, direction=WindingDirection.CLOCKWISE),
                Contour(points=inner_points, direction=WindingDirection.COUNTER_CLOCKWISE),
            ],
        )

        transformed = transformer.transform(glyph, upm=1000)

        # Should have 2 contours (left piece + right piece from merged outer+inner)
        assert len(transformed.contours) == 2

        # Both merged contours should be CW (filled)
        for contour in transformed.contours:
            assert contour.direction == WindingDirection.CLOCKWISE

    def test_transform_preserves_glyph_metadata(self) -> None:
        """Test that transformation preserves glyph metadata."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator)

        metadata = GlyphMetadata(
            name="TestGlyph", unicode=ord("A"), advance_width=500, left_side_bearing=50
        )

        glyph = Glyph(
            metadata=metadata,
            contours=[Contour(points=[Point(0.0, 0.0), Point(10.0, 10.0)])],
        )

        transformed = transformer.transform(glyph, upm=1000)

        assert transformed.metadata == metadata

    def test_transform_handles_single_island(self) -> None:
        """Test transforming a glyph with a single island creates merged contours."""
        analyzer = GlyphAnalyzer()
        # Use 30% bridge width so bridges fit within the 50-unit wide inner contour
        config = BridgeConfig(width_percent=30.0, min_bridges=1)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator)

        # Create glyph with one outer and one inner contour
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 100.0),
                Point(100.0, 100.0),
                Point(100.0, 0.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        inner = Contour(
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        glyph = Glyph(
            metadata=GlyphMetadata(
                name="O", unicode=ord("O"), advance_width=100, left_side_bearing=0
            ),
            contours=[outer, inner],
        )

        transformed = transformer.transform(glyph, upm=1000)

        # Should have 2 merged contours (replacing original outer+inner)
        assert len(transformed.contours) == 2

        # All contours should be CW (no CCW holes - that's the whole point!)
        for contour in transformed.contours:
            assert contour.direction == WindingDirection.CLOCKWISE
