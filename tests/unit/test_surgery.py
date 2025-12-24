"""Unit tests for contour surgery module.

Tests cover:
- Creating bridge hole contours
- Glyph transformation with bridge holes
"""

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.multi_island import has_spanning_obstruction, merge_multi_island_vertical
from stencilizer.core.surgery import ContourMerger, GlyphTransformer
from stencilizer.domain import (
    Contour,
    Glyph,
    GlyphMetadata,
    Point,
    WindingDirection,
)


class TestContourMerger:
    """Tests for ContourMerger class."""

    def test_merge_contours_with_bridges_vertical(self) -> None:
        """Test merging contours with vertical bridges (top + bottom)."""
        merger = ContourMerger()

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

        # Should create 4 contours: 2 outer (CW) + 2 inner (CCW) for LEFT/RIGHT pieces
        assert len(merged) == 4

        # Each contour should have valid geometry
        for contour in merged:
            assert len(contour.points) >= 3
            assert contour.direction in (WindingDirection.CLOCKWISE, WindingDirection.COUNTER_CLOCKWISE)

        # Verify we have correct number of CW (outer) and CCW (inner) contours
        cw_count = sum(1 for c in merged if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(1 for c in merged if c.direction == WindingDirection.COUNTER_CLOCKWISE)
        assert cw_count == 2, f"Expected 2 CW (outer) contours, got {cw_count}"
        assert ccw_count == 2, f"Expected 2 CCW (inner) contours, got {ccw_count}"

    def test_merge_multi_island_vertical_creates_separate_contours(self) -> None:
        """Test that merge_multi_island_vertical creates separate outer and inner contours."""
        # Create outer contour (large rectangle) - clockwise for TrueType outer
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 200.0),
                Point(0.0, 200.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        # Create two vertically-stacked inner contours - counter-clockwise for holes
        inner1 = Contour(  # Top island
            points=[
                Point(25.0, 125.0),
                Point(75.0, 125.0),
                Point(75.0, 175.0),
                Point(25.0, 175.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )
        inner2 = Contour(  # Bottom island
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        result = merge_multi_island_vertical(outer, [inner1, inner2], bridge_width=20.0)

        # Should create 6 contours: 2 outer (CW) + 4 inner (CCW) for left/right pieces
        # Left: 1 outer + 2 inner portions; Right: 1 outer + 2 inner portions
        assert len(result) >= 4, f"Should create at least 4 contours, got {len(result)}"

        # Verify we have CW (outer) and CCW (inner) contours
        cw_count = sum(1 for c in result if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(1 for c in result if c.direction == WindingDirection.COUNTER_CLOCKWISE)
        assert cw_count == 2, f"Expected 2 CW (outer) contours, got {cw_count}"
        assert ccw_count >= 2, f"Expected at least 2 CCW (inner) contours, got {ccw_count}"

    def test_has_spanning_obstruction_allows_structural_bars(self) -> None:
        """Test that has_spanning_obstruction allows structural bars (same winding as outer)."""
        # Outer contour (large rectangle)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 200.0),
                Point(0.0, 200.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        # Two vertically-stacked inner contours
        inner1 = Contour(  # Top island
            points=[
                Point(25.0, 125.0),
                Point(75.0, 125.0),
                Point(75.0, 175.0),
                Point(25.0, 175.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )
        inner2 = Contour(  # Bottom island
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        # Horizontal bar contour in the gap between islands (like Theta)
        bar = Contour(
            points=[
                Point(30.0, 95.0),
                Point(70.0, 95.0),
                Point(70.0, 105.0),
                Point(30.0, 105.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        inners = [inner1, inner2]
        all_contours = [outer, inner1, inner2, bar]

        # Bridge would be around center X (50) with width 20 -> bridge_left=40, bridge_right=60
        # Horizontal bars with same winding as outer are "structural" and NOT obstructions
        # They get split along with everything else for proper stencil cutting
        result = has_spanning_obstruction(
            outer, inners, all_contours, bridge_left_x=40.0, bridge_right_x=60.0
        )

        assert result is False, "Structural bars (same winding as outer) should not be obstructions"

    def test_has_spanning_obstruction_no_obstruction(self) -> None:
        """Test that has_spanning_obstruction returns False when path is clear."""
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 200.0),
                Point(0.0, 200.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        inner1 = Contour(
            points=[
                Point(25.0, 125.0),
                Point(75.0, 125.0),
                Point(75.0, 175.0),
                Point(25.0, 175.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )
        inner2 = Contour(
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        inners = [inner1, inner2]
        all_contours = [outer, inner1, inner2]  # No bar - path is clear

        result = has_spanning_obstruction(
            outer, inners, all_contours, bridge_left_x=40.0, bridge_right_x=60.0
        )

        assert result is False, "Should not detect obstruction when path is clear"

    def test_merge_multi_island_vertical_splits_structural_bar(self) -> None:
        """Test that merge_multi_island_vertical splits structural bars along with everything else."""
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 200.0),
                Point(0.0, 200.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        inner1 = Contour(
            points=[
                Point(25.0, 125.0),
                Point(75.0, 125.0),
                Point(75.0, 175.0),
                Point(25.0, 175.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )
        inner2 = Contour(
            points=[
                Point(25.0, 25.0),
                Point(75.0, 25.0),
                Point(75.0, 75.0),
                Point(25.0, 75.0),
            ],
            direction=WindingDirection.COUNTER_CLOCKWISE,
        )

        # Horizontal bar in gap - will be split into LEFT and RIGHT pieces
        bar = Contour(
            points=[
                Point(30.0, 95.0),
                Point(70.0, 95.0),
                Point(70.0, 105.0),
                Point(30.0, 105.0),
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        inners = [inner1, inner2]
        all_contours = [outer, inner1, inner2, bar]

        result = merge_multi_island_vertical(
            outer, inners, bridge_width=20.0, all_contours=all_contours
        )

        # Should create LEFT and RIGHT pieces for outer, inners, and bar
        # Minimum: 2 outer pieces + 4 inner pieces = 6, plus bar pieces
        assert len(result) >= 6, f"Should create spanning bridge pieces, got {len(result)}"
        assert result[0] != outer, "Should not return original outer unchanged"


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

        # Should have 4 contours (2 outer CW + 2 inner CCW for left/right pieces)
        assert len(transformed.contours) == 4

        # All contours should have valid geometry
        for contour in transformed.contours:
            assert len(contour.points) >= 3

        # Verify correct winding distribution (2 CW outer + 2 CCW inner)
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE)
        assert cw_count == 2, f"Expected 2 CW (outer) contours, got {cw_count}"
        assert ccw_count == 2, f"Expected 2 CCW (inner) contours, got {ccw_count}"

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

        # Should have 4 contours (2 outer CW + 2 inner CCW for left/right pieces)
        assert len(transformed.contours) == 4

        # All contours should have valid geometry
        for contour in transformed.contours:
            assert len(contour.points) >= 3

        # Verify correct winding distribution
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE)
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
        config = BridgeConfig(width_percent=30.0, min_bridges=1)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator, bridge_config=config)

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
                Point(70.0, 110.0),   # bottom-left
                Point(70.0, 160.0),   # top-left
                Point(130.0, 160.0),  # top-right
                Point(130.0, 110.0),  # bottom-right
            ],
            direction=WindingDirection.CLOCKWISE,
        )

        # Create bottom bowl of 8 (CW - filled area inside the digit cutout)
        # CW order: bottom-left → top-left → top-right → bottom-right
        bottom_bowl = Contour(
            points=[
                Point(70.0, 40.0),    # bottom-left
                Point(70.0, 90.0),    # top-left
                Point(130.0, 90.0),   # top-right
                Point(130.0, 40.0),   # bottom-right
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
            assert len(contour.points) >= 3, (
                f"Contour has only {len(contour.points)} points"
            )

        # Verify we have both CW and CCW contours (indicating proper bridging)
        cw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.CLOCKWISE)
        ccw_count = sum(1 for c in transformed.contours if c.direction == WindingDirection.COUNTER_CLOCKWISE)
        assert cw_count >= 2, f"Expected at least 2 CW contours, got {cw_count}"
        assert ccw_count >= 2, f"Expected at least 2 CCW contours, got {ccw_count}"
