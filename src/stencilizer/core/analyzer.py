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
class ContourNode:
    """A node in the contour nesting tree.

    Attributes:
        index: Index of this contour in the glyph's contour list
        is_outer: True if CW (outer/filled), False if CCW (hole)
        parent: Index of parent contour (None if root)
        children: Indices of child contours
        depth: Nesting depth (0 for top-level)
    """

    index: int
    is_outer: bool
    parent: int | None
    children: list[int]
    depth: int


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
        nesting_tree: Complete nesting tree of all contours (ContourNode for each)
        nested_outers: CW contours inside CCW holes (need special bridging)
    """

    outer_contours: list[int]
    inner_contours: list[int]
    containment: dict[int, int]
    islands: list[int]
    nesting_tree: dict[int, ContourNode] | None = None
    nested_outers: list[int] | None = None

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
        2. Build complete nesting tree of all contours
        3. Identify islands and nested outers

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
                nesting_tree={},
                nested_outers=[],
            )

        outer_contours: list[int] = []
        inner_contours: list[int] = []
        contour_is_outer: dict[int, bool] = {}

        # Classify contours by winding direction
        # TrueType convention: outer contours are clockwise (negative area),
        # inner contours (holes) are counter-clockwise (positive area)
        for idx, contour in enumerate(glyph.contours):
            area = signed_area(contour.points)
            if area < 0:
                # Negative area = clockwise = outer contour (TrueType convention)
                outer_contours.append(idx)
                contour_is_outer[idx] = True
            elif area > 0:
                # Positive area = counter-clockwise = inner contour (hole)
                inner_contours.append(idx)
                contour_is_outer[idx] = False
            # Skip degenerate contours (area == 0)

        # Build complete nesting tree
        nesting_tree, nested_outers = self._build_nesting_tree(
            glyph.contours, contour_is_outer
        )

        # Determine containment relationships (for backward compatibility)
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
            nesting_tree=nesting_tree,
            nested_outers=nested_outers,
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

    def _build_nesting_tree(
        self,
        contours: list[Contour],
        contour_is_outer: dict[int, bool],
    ) -> tuple[dict[int, ContourNode], list[int]]:
        """Build complete nesting tree of all contours.

        For each contour, finds its immediate parent (smallest contour that
        contains it), regardless of winding direction. This allows detecting:
        - CCW holes inside CW outers (normal islands)
        - CW outers inside CCW holes (nested islands like R inside ®)

        Args:
            contours: All contours in the glyph
            contour_is_outer: Map of contour index to whether it's CW (outer)

        Returns:
            Tuple of (nesting_tree, nested_outers) where:
            - nesting_tree: Dict mapping contour index to ContourNode
            - nested_outers: List of CW contours inside CCW holes
        """
        n = len(contours)
        if n == 0:
            return {}, []

        # Calculate bounding box areas for each contour (used for sorting)
        bbox_areas: dict[int, float] = {}
        for idx, contour in enumerate(contours):
            if idx not in contour_is_outer:
                continue
            bbox = contour.bounding_box()
            bbox_areas[idx] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # For each contour, find its immediate parent (smallest containing contour)
        parent_map: dict[int, int | None] = {}

        for idx in contour_is_outer:
            contour = contours[idx]
            if not contour.points:
                parent_map[idx] = None
                continue

            # Test point for containment
            test_point = contour.points[0]

            # Find all contours that contain this one
            candidates: list[int] = []
            for other_idx in contour_is_outer:
                if other_idx == idx:
                    continue
                other_contour = contours[other_idx]
                if point_in_polygon(test_point, other_contour.points):
                    candidates.append(other_idx)

            if not candidates:
                parent_map[idx] = None
            else:
                # Choose the smallest containing contour as parent
                parent_map[idx] = min(candidates, key=lambda i: bbox_areas.get(i, float('inf')))

        # Build tree nodes with depth calculation
        def get_depth(idx: int, memo: dict[int, int]) -> int:
            if idx in memo:
                return memo[idx]
            parent = parent_map.get(idx)
            if parent is None:
                memo[idx] = 0
            else:
                memo[idx] = get_depth(parent, memo) + 1
            return memo[idx]

        depth_memo: dict[int, int] = {}
        nesting_tree: dict[int, ContourNode] = {}

        for idx in contour_is_outer:
            depth = get_depth(idx, depth_memo)
            nesting_tree[idx] = ContourNode(
                index=idx,
                is_outer=contour_is_outer[idx],
                parent=parent_map.get(idx),
                children=[],
                depth=depth,
            )

        # Populate children lists
        for idx, node in nesting_tree.items():
            if node.parent is not None and node.parent in nesting_tree:
                nesting_tree[node.parent].children.append(idx)

        # Identify nested outers: CW contours inside CCW holes
        nested_outers: list[int] = []
        for idx, node in nesting_tree.items():
            if node.is_outer and node.parent is not None:
                parent_node = nesting_tree.get(node.parent)
                if parent_node and not parent_node.is_outer:
                    # This is a CW contour inside a CCW hole
                    nested_outers.append(idx)

        return nesting_tree, nested_outers


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
