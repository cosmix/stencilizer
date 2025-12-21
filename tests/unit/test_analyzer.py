"""Unit tests for glyph analysis engine.

Tests cover:
- Contour classification by winding direction
- Containment detection between contours
- Island identification
- Edge cases (empty glyphs, degenerate contours)
"""

from stencilizer.core.analyzer import GlyphAnalyzer, get_island_glyphs
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point


class TestContourClassification:
    """Tests for classifying contours by winding direction."""

    def test_outer_contour_detection_cw(self):
        """Clockwise contour should be classified as outer (TrueType convention)."""
        # Create a square with CW winding (TrueType outer)
        points = [
            Point(0.0, 0.0),
            Point(0.0, 100.0),
            Point(100.0, 100.0),
            Point(100.0, 0.0),
        ]
        contour = Contour(points=points)

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[contour])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.outer_contours == [0]
        assert hierarchy.inner_contours == []
        assert hierarchy.islands == []

    def test_inner_contour_detection_ccw(self):
        """Counter-clockwise contour should be classified as inner (TrueType convention)."""
        # Create a square with CCW winding (TrueType inner/hole)
        points = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        contour = Contour(points=points)

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[contour])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.outer_contours == []
        assert hierarchy.inner_contours == [0]
        assert hierarchy.islands == []

    def test_mixed_contours(self):
        """Glyph with both outer and inner contours."""
        # Outer contour (CW - TrueType convention)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(200.0, 200.0),
                Point(200.0, 0.0),
            ]
        )

        # Inner contour (CCW - TrueType convention for holes)
        inner = Contour(
            points=[
                Point(50.0, 50.0),
                Point(150.0, 50.0),
                Point(150.0, 150.0),
                Point(50.0, 150.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer, inner])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.outer_contours == [0]
        assert hierarchy.inner_contours == [1]


class TestContainmentDetection:
    """Tests for detecting containment relationships."""

    def test_containment_detection(self):
        """Inner contour inside outer contour should be detected."""
        # Outer contour (CW - TrueType convention)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(200.0, 200.0),
                Point(200.0, 0.0),
            ]
        )

        # Inner contour (CCW - TrueType convention) fully inside outer
        inner = Contour(
            points=[
                Point(50.0, 50.0),
                Point(150.0, 50.0),
                Point(150.0, 150.0),
                Point(50.0, 150.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer, inner])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert 1 in hierarchy.containment
        assert hierarchy.containment[1] == 0

    def test_no_containment_separate_contours(self):
        """Separate contours should not show containment."""
        # First outer contour (CW - TrueType)
        outer1 = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 100.0),
                Point(100.0, 100.0),
                Point(100.0, 0.0),
            ]
        )

        # Second outer contour (CW - TrueType) - separate
        outer2 = Contour(
            points=[
                Point(200.0, 0.0),
                Point(200.0, 100.0),
                Point(300.0, 100.0),
                Point(300.0, 0.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=400, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer1, outer2])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert len(hierarchy.containment) == 0


class TestIslandIdentification:
    """Tests for identifying islands (fully enclosed inner contours)."""

    def test_island_identification(self):
        """Inner contour fully enclosed in outer should be an island."""
        # Outer contour (CW - TrueType)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(200.0, 200.0),
                Point(200.0, 0.0),
            ]
        )

        # Inner contour (CCW - TrueType) fully inside outer
        inner = Contour(
            points=[
                Point(50.0, 50.0),
                Point(150.0, 50.0),
                Point(150.0, 150.0),
                Point(50.0, 150.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer, inner])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.islands == [1]

    def test_multiple_islands(self):
        """Multiple inner contours should all be detected as islands."""
        # Outer contour (CW - TrueType)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(300.0, 200.0),
                Point(300.0, 0.0),
            ]
        )

        # First inner contour (CCW - TrueType)
        inner1 = Contour(
            points=[
                Point(50.0, 50.0),
                Point(100.0, 50.0),
                Point(100.0, 150.0),
                Point(50.0, 150.0),
            ]
        )

        # Second inner contour (CCW - TrueType)
        inner2 = Contour(
            points=[
                Point(200.0, 50.0),
                Point(250.0, 50.0),
                Point(250.0, 150.0),
                Point(200.0, 150.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=300, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer, inner1, inner2])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert sorted(hierarchy.islands) == [1, 2]

    def test_no_islands_simple_glyph(self):
        """Simple glyph with no islands should return empty islands list."""
        # Single outer contour (CW - TrueType)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 100.0),
                Point(100.0, 100.0),
                Point(100.0, 0.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[outer])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.islands == []


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_glyph(self):
        """Empty glyph should return empty hierarchy."""
        metadata = GlyphMetadata(
            name="space", unicode=32, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.outer_contours == []
        assert hierarchy.inner_contours == []
        assert hierarchy.containment == {}
        assert hierarchy.islands == []

    def test_degenerate_contour(self):
        """Contour with area zero should be skipped."""
        # Degenerate contour (all points collinear)
        degenerate = Contour(
            points=[
                Point(0.0, 0.0),
                Point(50.0, 0.0),
                Point(100.0, 0.0),
            ]
        )

        metadata = GlyphMetadata(
            name="test", unicode=None, advance_width=200, left_side_bearing=0
        )
        glyph = Glyph(metadata=metadata, contours=[degenerate])

        analyzer = GlyphAnalyzer()
        hierarchy = analyzer.analyze(glyph)

        assert hierarchy.outer_contours == []
        assert hierarchy.inner_contours == []


class TestGetIslandGlyphs:
    """Tests for the get_island_glyphs utility function."""

    def test_filter_glyphs_with_islands(self):
        """Should return only glyphs that contain islands."""
        # Glyph with island - outer is CW, inner is CCW (TrueType)
        outer = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 200.0),
                Point(200.0, 200.0),
                Point(200.0, 0.0),
            ]
        )
        inner = Contour(
            points=[
                Point(50.0, 50.0),
                Point(150.0, 50.0),
                Point(150.0, 150.0),
                Point(50.0, 150.0),
            ]
        )
        glyph_with_island = Glyph(
            metadata=GlyphMetadata(
                name="O", unicode=79, advance_width=200, left_side_bearing=0
            ),
            contours=[outer, inner],
        )

        # Glyph without island - single CW contour (TrueType outer)
        simple = Contour(
            points=[
                Point(0.0, 0.0),
                Point(0.0, 100.0),
                Point(100.0, 100.0),
                Point(100.0, 0.0),
            ]
        )
        glyph_without_island = Glyph(
            metadata=GlyphMetadata(
                name="L", unicode=76, advance_width=100, left_side_bearing=0
            ),
            contours=[simple],
        )

        glyphs = [glyph_with_island, glyph_without_island]
        result = get_island_glyphs(glyphs)

        assert len(result) == 1
        assert result[0].name == "O"

    def test_empty_list(self):
        """Should handle empty glyph list."""
        result = get_island_glyphs([])
        assert result == []
