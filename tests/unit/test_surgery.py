"""Unit tests for contour surgery module.

Tests cover:
- Creating bridge hole contours
- Glyph transformation with bridge holes
"""

from stencilizer.core.multi_island import has_spanning_obstruction, merge_multi_island_vertical
from stencilizer.core.surgery import ContourMerger
from stencilizer.domain import (
    Contour,
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
            assert contour.direction in (
                WindingDirection.CLOCKWISE,
                WindingDirection.COUNTER_CLOCKWISE,
            )

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
