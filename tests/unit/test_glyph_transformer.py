"""Tests for live glyph transformation."""

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point, WindingDirection


class TestGlyphTransformer:
    """Tests for GlyphTransformer class."""

    def test_transform_glyph_without_islands(self) -> None:
        """Test transforming a glyph with no islands returns unchanged glyph."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

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
        config = BridgeConfig(width_percent=30.0)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

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

        # Should have 4 contours (2 outer CW + 2 inner CCW for left/right pieces)
        assert len(transformed.contours) == 4

        # All contours should have valid geometry
        for contour in transformed.contours:
            assert len(contour.points) >= 3

        # Verify correct winding distribution (2 CW outer + 2 CCW inner)
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(
            1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE
        )
        assert cw_count == 2, f"Expected 2 CW (outer) contours, got {cw_count}"
        assert ccw_count == 2, f"Expected 2 CCW (inner) contours, got {ccw_count}"

    def test_transform_preserves_glyph_metadata(self) -> None:
        """Test that transformation preserves glyph metadata."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

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
        config = BridgeConfig(width_percent=30.0)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

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

        # Should have 4 contours (2 outer CW + 2 inner CCW for left/right pieces)
        assert len(transformed.contours) == 4

        # All contours should have valid geometry
        for contour in transformed.contours:
            assert len(contour.points) >= 3

        # Verify correct winding distribution
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(
            1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE
        )
        assert cw_count == 2, f"Expected 2 CW (outer) contours, got {cw_count}"
        assert ccw_count == 2, f"Expected 2 CCW (inner) contours, got {ccw_count}"

    def test_transform_filled_encircled_digit_with_inverted_islands(self) -> None:
        """Test transforming a filled encircled digit creates bridges for inverted islands.

        Filled encircled digits (like ⑧) have:
        - Outer: filled circle (CW)
        - Island: digit outline cutout (CCW hole)
        - Inverted islands: filled areas within the digit (CW nested outers)

        The inverted islands (like bowls of 8) need bridges to connect them
        to the surrounding hole boundary.
        """
        analyzer = GlyphAnalyzer()
        # Use 30% bridge width
        config = BridgeConfig(width_percent=30.0)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

        # Create filled circle outer (CW - filled area)
        outer_circle = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(200.0, 200.0),
                Point(200.0, 0.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        # Create digit-8-shaped cutout (CCW hole in the circle)
        # This simulates the negative space of an "8" inside a filled circle
        digit_cutout = Contour(
            points=[
                Point(50.0, 20.0),
                Point(150.0, 20.0),
                Point(150.0, 180.0),
                Point(50.0, 180.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        # Create top bowl of 8 (CW - filled area inside the digit cutout)
        # This is an "inverted island" - a filled area inside a hole
        # CW order: bottom-left → top-left → top-right → bottom-right
        top_bowl = Contour(
            points=[
                Point(70.0, 110.0),  # bottom-left
                Point(70.0, 160.0),  # top-left
                Point(130.0, 160.0),  # top-right
                Point(130.0, 110.0),  # bottom-right
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        # Create bottom bowl of 8 (CW - filled area inside the digit cutout)
        # CW order: bottom-left → top-left → top-right → bottom-right
        bottom_bowl = Contour(
            points=[
                Point(70.0, 40.0),  # bottom-left
                Point(70.0, 90.0),  # top-left
                Point(130.0, 90.0),  # top-right
                Point(130.0, 40.0),  # bottom-right
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        glyph = Glyph(
            metadata=GlyphMetadata(
                name="eight.circle", unicode=0x2467, advance_width=200, left_side_bearing=0
            ),
            contours=[outer_circle, digit_cutout, top_bowl, bottom_bowl],
        )

        transformed = transformer.transform(glyph, upm=1000)

        # The transformation should create bridges:
        # 1. Between outer circle and digit cutout (main bridging)
        # 2. Between each bowl and the digit cutout (inverted island bridging)

        # We should have more contours than the original 4 due to splitting
        # At minimum: main outer/inner split creates 4, plus bridges for bowls
        assert len(transformed.contours) >= 4, (
            f"Expected at least 4 contours after transformation, got {len(transformed.contours)}"
        )

        # All contours should have valid geometry
        for contour in transformed.contours:
            assert len(contour.points) >= 3, f"Contour has only {len(contour.points)} points"

        # Verify we have both CW and CCW contours (indicating proper bridging)
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(
            1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE
        )
        assert cw_count >= 2, f"Expected at least 2 CW contours, got {cw_count}"
        assert ccw_count >= 2, f"Expected at least 2 CCW contours, got {ccw_count}"
