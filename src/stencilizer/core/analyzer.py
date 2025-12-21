"""Glyph analysis engine for detecting islands and contour hierarchy.

This module analyzes glyphs to identify:
- Outer contours (counter-clockwise winding)
- Inner contours (clockwise winding)
- Containment relationships between contours
- Islands (fully enclosed inner contours)

The analysis uses signed area calculation to determine winding direction
and point-in-polygon tests to establish containment relationships.
"""

from dataclasses import dataclass

from stencilizer.core.geometry import point_in_polygon, signed_area
from stencilizer.domain import Contour, Glyph


@dataclass
class ContourHierarchy:
    """Hierarchical classification of contours in a glyph.

    This structure captures the relationship between outer and inner contours,
    including which inner contours are fully contained within which outer
    contours (islands).

    Attributes:
        outer_contours: Indices of contours with counter-clockwise winding
        inner_contours: Indices of contours with clockwise winding
        containment: Maps inner contour index to the outer contour index that
            contains it. Only includes inner contours that are fully enclosed.
        islands: Indices of inner contours that are fully enclosed within
            exactly one outer contour
    """

    outer_contours: list[int]
    inner_contours: list[int]
    containment: dict[int, int]
    islands: list[int]

    def has_islands(self) -> bool:
        """Check if this hierarchy contains any islands.

        Returns:
            True if there are one or more islands
        """
        return len(self.islands) > 0

    def get_islands(self) -> list[int]:
        """Get list of island indices.

        Returns:
            List of contour indices that are islands
        """
        return self.islands


class GlyphAnalyzer:
    """Analyzes glyphs to detect islands and contour hierarchy.

    This analyzer uses geometric properties to classify contours:
    - Winding direction (via signed area calculation)
    - Point containment (via ray casting algorithm)

    The analyzer is stateless and safe for use in parallel processing.
    """

    def analyze(self, glyph: Glyph) -> ContourHierarchy:
        """Analyze a glyph to determine its contour hierarchy.

        Process:
        1. Classify contours by winding direction (outer vs inner)
        2. Test containment relationships between inner and outer contours
        3. Identify islands (inner contours fully enclosed in one outer)

        Args:
            glyph: The glyph to analyze

        Returns:
            ContourHierarchy containing classification and relationships
        """
        if not glyph.contours:
            return ContourHierarchy(
                outer_contours=[],
                inner_contours=[],
                containment={},
                islands=[],
            )

        outer_contours: list[int] = []
        inner_contours: list[int] = []

        # Classify contours by winding direction
        # TrueType convention: outer contours are clockwise (negative area),
        # inner contours (holes) are counter-clockwise (positive area)
        for idx, contour in enumerate(glyph.contours):
            area = signed_area(contour.points)
            if area < 0:
                # Negative area = clockwise = outer contour (TrueType convention)
                outer_contours.append(idx)
            elif area > 0:
                # Positive area = counter-clockwise = inner contour (hole)
                inner_contours.append(idx)
            # Skip degenerate contours (area == 0)

        # Determine containment relationships
        containment: dict[int, int] = {}
        islands: list[int] = []

        for inner_idx in inner_contours:
            inner_contour = glyph.contours[inner_idx]
            containing_outer = self._find_containing_outer(
                inner_contour, glyph.contours, outer_contours
            )

            if containing_outer is not None:
                containment[inner_idx] = containing_outer

                # Check if this inner contour is an island
                # (all points inside exactly one outer contour)
                if self._is_island(inner_contour, glyph.contours[containing_outer]):
                    islands.append(inner_idx)

        return ContourHierarchy(
            outer_contours=outer_contours,
            inner_contours=inner_contours,
            containment=containment,
            islands=islands,
        )

    def _find_containing_outer(
        self,
        inner_contour: Contour,
        all_contours: list[Contour],
        outer_indices: list[int],
    ) -> int | None:
        """Find the outer contour that contains an inner contour.

        Tests the first point of the inner contour against each outer contour.
        Returns the first outer contour that contains the test point.

        Args:
            inner_contour: The inner contour to test
            all_contours: All contours in the glyph
            outer_indices: Indices of outer contours to test against

        Returns:
            Index of containing outer contour, or None if not contained
        """
        if not inner_contour.points:
            return None

        # Use first point as representative test point
        test_point = inner_contour.points[0]

        for outer_idx in outer_indices:
            outer_contour = all_contours[outer_idx]
            if point_in_polygon(test_point, outer_contour.points):
                return outer_idx

        return None

    def _is_island(self, inner_contour: Contour, outer_contour: Contour) -> bool:
        """Check if an inner contour is fully enclosed within an outer contour.

        An island is defined as an inner contour where ALL points are inside
        the outer contour. This is a strict definition to ensure the inner
        contour is truly isolated.

        Args:
            inner_contour: The inner contour to test
            outer_contour: The outer contour to test against

        Returns:
            True if all points of inner contour are inside outer contour
        """
        # Test all points to ensure complete containment
        for point in inner_contour.points:
            if not point_in_polygon(point, outer_contour.points):
                return False

        return True


def get_island_glyphs(glyphs: list[Glyph]) -> list[Glyph]:
    """Filter glyphs to those containing islands.

    This is a convenience function for quickly identifying glyphs that
    require bridge processing.

    Args:
        glyphs: List of glyphs to filter

    Returns:
        List of glyphs that contain at least one island
    """
    analyzer = GlyphAnalyzer()
    result: list[Glyph] = []

    for glyph in glyphs:
        hierarchy = analyzer.analyze(glyph)
        if hierarchy.islands:
            result.append(glyph)

    return result
