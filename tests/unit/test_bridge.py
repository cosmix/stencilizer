"""Unit tests for bridge placement algorithm."""

import pytest

from stencilizer.config import BridgeConfig, BridgePosition
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.domain import BridgeSpec, Contour, Glyph, GlyphMetadata, Point, WindingDirection


class TestBridgePlacer:
    """Tests for BridgePlacer class."""

    @pytest.fixture
    def config(self) -> BridgeConfig:
        """Create default bridge config."""
        return BridgeConfig()

    @pytest.fixture
    def placer(self, config: BridgeConfig) -> BridgePlacer:
        """Create bridge placer."""
        return BridgePlacer(config)

    @pytest.fixture
    def simple_square_island(self) -> Contour:
        """Create a simple square island (inner contour)."""
        points = [
            Point(100.0, 100.0),
            Point(200.0, 100.0),
            Point(200.0, 200.0),
            Point(100.0, 200.0),
        ]
        return Contour(points=points, direction=WindingDirection.CLOCKWISE)

    @pytest.fixture
    def simple_square_outer(self) -> Contour:
        """Create a simple square outer contour."""
        points = [
            Point(0.0, 0.0),
            Point(300.0, 0.0),
            Point(300.0, 300.0),
            Point(0.0, 300.0),
        ]
        return Contour(points=points, direction=WindingDirection.COUNTER_CLOCKWISE)

    def test_find_candidates_returns_expected_count(
        self, placer: BridgePlacer, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test that find_candidates returns candidates based on available points."""
        # Square has 4 points, so sampling 12 will return all 4 points
        candidates = placer.find_candidates(simple_square_island, simple_square_outer, 0, 1, 12)

        # Should return 4 candidates (one per existing point)
        assert len(candidates) == 4

    def test_find_candidates_creates_valid_specs(
        self, placer: BridgePlacer, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test that candidates have valid bridge specs."""
        candidates = placer.find_candidates(simple_square_island, simple_square_outer, 0, 1, 8)

        for spec in candidates:
            # Check indices are set correctly
            assert spec.inner_contour_idx == 0
            assert spec.outer_contour_idx == 1

            # Check points are valid
            assert isinstance(spec.inner_point, Point)
            assert isinstance(spec.outer_point, Point)

            # Initial width should be 0 (set during generation)
            assert spec.width == 0.0

            # Initial score should be 0 (set during scoring)
            assert spec.score == 0.0

    def test_score_candidate_longer_is_better(
        self, placer: BridgePlacer, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test that longer bridges score higher (ensures full stroke cut-through)."""
        # Create mock glyph
        glyph = Glyph(
            metadata=GlyphMetadata(name="test", unicode=None, advance_width=1000, left_side_bearing=0),
            contours=[simple_square_island, simple_square_outer],
        )

        # Create two bridge specs with different lengths
        short_spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),  # Bottom of island
            outer_point=Point(150.0, 50.0),  # 50 units away
            width=0.0,
        )

        long_spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),  # Bottom of island
            outer_point=Point(150.0, 0.0),  # 100 units away
            width=0.0,
        )

        short_score = placer.score_candidate(short_spec, glyph, 1000)
        long_score = placer.score_candidate(long_spec, glyph, 1000)

        # Longer bridges score higher to ensure full stroke cut-through
        assert long_score > short_score

    def test_score_candidate_zero_length_bridge(
        self, placer: BridgePlacer, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test that zero-length bridges get very poor score."""
        glyph = Glyph(
            metadata=GlyphMetadata(name="test", unicode=None, advance_width=1000, left_side_bearing=0),
            contours=[simple_square_island, simple_square_outer],
        )

        # Create degenerate bridge (same start and end point)
        degenerate_spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 100.0),  # Same point
            width=0.0,
        )

        score = placer.score_candidate(degenerate_spec, glyph, 1000)
        assert score < 0  # Should be negative/very poor

    def test_score_position_auto_preference(
        self, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test position scoring with AUTO preference."""
        config = BridgeConfig(position_preference=BridgePosition.AUTO)
        placer = BridgePlacer(config)
        glyph = Glyph(
            metadata=GlyphMetadata(name="test", unicode=None, advance_width=1000, left_side_bearing=0),
            contours=[simple_square_island, simple_square_outer],
        )

        # Bridge going downward (from inner to outer) - this is a "bottom" bridge
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),  # Below inner - going down
            width=0.0,
        )

        score = placer._score_position(spec, glyph)
        # AUTO mode prefers bottom positions - should score high (close to 1.0)
        assert score >= 0.7

    def test_score_position_top_preference(
        self, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test position scoring with TOP preference."""
        config = BridgeConfig(position_preference=BridgePosition.TOP)
        placer = BridgePlacer(config)
        glyph = Glyph(
            metadata=GlyphMetadata(name="test", unicode=None, advance_width=1000, left_side_bearing=0),
            contours=[simple_square_island, simple_square_outer],
        )

        # Upward bridge (should score high)
        upward_spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 200.0),
            outer_point=Point(150.0, 250.0),  # Goes up
            width=0.0,
        )

        # Downward bridge (should score low)
        downward_spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),  # Goes down
            width=0.0,
        )

        upward_score = placer._score_position(upward_spec, glyph)
        downward_score = placer._score_position(downward_spec, glyph)

        assert upward_score > downward_score

    def test_select_bridges_returns_minimum_count(
        self, placer: BridgePlacer, simple_square_island: Contour, simple_square_outer: Contour
    ) -> None:
        """Test that select_bridges returns at least min_count bridges when possible."""
        # Create candidates and score them
        candidates = placer.find_candidates(simple_square_island, simple_square_outer, 0, 1, 12)

        glyph = Glyph(
            metadata=GlyphMetadata(name="test", unicode=None, advance_width=1000, left_side_bearing=0),
            contours=[simple_square_island, simple_square_outer],
        )

        # Score all candidates
        for candidate in candidates:
            candidate.score = placer.score_candidate(candidate, glyph, 1000)

        # Select bridges with lower min_spacing_factor to allow more bridges
        selected = placer.select_bridges(candidates, min_count=2, min_spacing_factor=0.05)

        assert len(selected) >= 2

    def test_select_bridges_picks_best_first(self, placer: BridgePlacer) -> None:
        """Test that select_bridges picks the best candidate first."""
        # Create candidates with known scores
        candidates = [
            BridgeSpec(
                inner_contour_idx=0,
                outer_contour_idx=1,
                inner_point=Point(100.0, 150.0),
                outer_point=Point(50.0, 150.0),
                width=0.0,
                score=0.5,
            ),
            BridgeSpec(
                inner_contour_idx=0,
                outer_contour_idx=1,
                inner_point=Point(150.0, 100.0),
                outer_point=Point(150.0, 50.0),
                width=0.0,
                score=0.9,  # Best score
            ),
            BridgeSpec(
                inner_contour_idx=0,
                outer_contour_idx=1,
                inner_point=Point(200.0, 150.0),
                outer_point=Point(250.0, 150.0),
                width=0.0,
                score=0.3,
            ),
        ]

        selected = placer.select_bridges(candidates, min_count=1)

        assert len(selected) >= 1
        assert selected[0].score == 0.9

    def test_select_bridges_empty_candidates(self, placer: BridgePlacer) -> None:
        """Test that select_bridges handles empty candidate list."""
        selected = placer.select_bridges([], min_count=1)
        assert selected == []

    def test_is_far_enough_spacing(self, placer: BridgePlacer) -> None:
        """Test spacing check between bridges."""
        selected = [
            BridgeSpec(
                inner_contour_idx=0,
                outer_contour_idx=1,
                inner_point=Point(100.0, 100.0),
                outer_point=Point(50.0, 50.0),
                width=0.0,
                score=0.8,
            )
        ]

        # Candidate very close to selected bridge
        close_candidate = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(105.0, 105.0),  # Only 5 units away
            outer_point=Point(55.0, 55.0),
            width=0.0,
            score=0.7,
        )

        # Candidate far from selected bridge
        far_candidate = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(200.0, 200.0),  # Far away
            outer_point=Point(250.0, 250.0),
            width=0.0,
            score=0.7,
        )

        # Close candidate should not be far enough
        assert not placer._is_far_enough(close_candidate, selected, 0.15)

        # Far candidate should be far enough
        assert placer._is_far_enough(far_candidate, selected, 0.15)


class TestBridgeGenerator:
    """Tests for BridgeGenerator class."""

    @pytest.fixture
    def config(self) -> BridgeConfig:
        """Create bridge config with known width."""
        return BridgeConfig(width_percent=50.0)  # 50% of stroke width

    @pytest.fixture
    def generator(self, config: BridgeConfig) -> BridgeGenerator:
        """Create bridge generator."""
        return BridgeGenerator(config)

    def test_generate_geometry_creates_four_vertices(self, generator: BridgeGenerator) -> None:
        """Test that generate_geometry creates 4 vertices."""
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        assert len(geometry.vertices) == 4

    def test_generate_geometry_calculates_width(self, generator: BridgeGenerator) -> None:
        """Test that bridge width is calculated from config."""
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),  # 50 units apart (stroke width)
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        # Expected width: 50% of 50 (stroke width) = 25
        expected_width = 25.0
        assert geometry.spec.width == expected_width

    def test_generate_geometry_perpendicular_to_bridge(self, generator: BridgeGenerator) -> None:
        """Test that bridge vertices are perpendicular to bridge direction."""
        # Vertical bridge (going down)
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),  # 50 units apart
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        # For vertical bridge, perpendicular should be horizontal
        # Check that x-coordinates vary (horizontal spread)
        x_coords = [v.x for v in geometry.vertices]
        assert len(set(x_coords)) > 1  # Should have different x values

        # Check that vertices are symmetric around bridge center
        v1, v2, v3, v4 = geometry.vertices

        # v1 and v2 should be at inner side (with inset)
        # With inset_percent=2% and stroke_width=50, inset = 1.0
        # inner_inset_y = 100 + (-1)*1.0 = 99.0 (moved toward outer)
        expected_inner_y = 99.0
        assert abs(v1.y - expected_inner_y) < 1e-6
        assert abs(v2.y - expected_inner_y) < 1e-6

        # v3 and v4 should be at outer side (with inset)
        # outer_inset_y = 50 - (-1)*1.0 = 51.0 (moved toward inner)
        expected_outer_y = 51.0
        assert abs(v3.y - expected_outer_y) < 1e-6
        assert abs(v4.y - expected_outer_y) < 1e-6

    def test_generate_geometry_horizontal_bridge(self, generator: BridgeGenerator) -> None:
        """Test geometry generation for horizontal bridge."""
        # Horizontal bridge (going right)
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(100.0, 150.0),
            outer_point=Point(50.0, 150.0),  # Going left
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        # For horizontal bridge, perpendicular should be vertical
        # Check that y-coordinates vary (vertical spread)
        y_coords = [v.y for v in geometry.vertices]
        assert len(set(y_coords)) > 1  # Should have different y values

    def test_generate_geometry_diagonal_bridge(self, generator: BridgeGenerator) -> None:
        """Test geometry generation for diagonal bridge."""
        # Diagonal bridge (45 degrees)
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(100.0, 100.0),
            outer_point=Point(150.0, 150.0),
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        # All four vertices should be distinct
        vertices_set = set((v.x, v.y) for v in geometry.vertices)
        assert len(vertices_set) == 4

    def test_generate_geometry_zero_length_raises_error(self, generator: BridgeGenerator) -> None:
        """Test that zero-length bridge raises error."""
        # Degenerate bridge (same start and end)
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(100.0, 100.0),
            outer_point=Point(100.0, 100.0),  # Same point
            width=0.0,
        )

        with pytest.raises(ValueError, match="zero length"):
            generator.generate_geometry(spec, upm=1000)

    def test_generate_geometry_preserves_spec(self, generator: BridgeGenerator) -> None:
        """Test that generated geometry preserves original spec data."""
        spec = BridgeSpec(
            inner_contour_idx=5,
            outer_contour_idx=7,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),
            width=0.0,
            score=0.85,
        )

        geometry = generator.generate_geometry(spec, upm=1000)

        # Check that spec data is preserved
        assert geometry.spec.inner_contour_idx == 5
        assert geometry.spec.outer_contour_idx == 7
        assert geometry.spec.score == 0.85
        assert geometry.spec.inner_point == spec.inner_point
        assert geometry.spec.outer_point == spec.outer_point

    def test_generate_geometry_as_contour(self, generator: BridgeGenerator) -> None:
        """Test converting bridge geometry to contour."""
        spec = BridgeSpec(
            inner_contour_idx=0,
            outer_contour_idx=1,
            inner_point=Point(150.0, 100.0),
            outer_point=Point(150.0, 50.0),
            width=0.0,
        )

        geometry = generator.generate_geometry(spec, upm=1000)
        contour = geometry.as_contour()

        # Contour should have 4 points
        assert len(contour.points) == 4

        # Contour should be counter-clockwise
        assert contour.direction == WindingDirection.COUNTER_CLOCKWISE

        # Points should match vertices
        for i, vertex in enumerate(geometry.vertices):
            assert contour.points[i] == vertex
