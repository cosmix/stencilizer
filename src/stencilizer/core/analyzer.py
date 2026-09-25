"""Glyph analysis engine for detecting islands and contour hierarchy."""

from dataclasses import dataclass

from stencilizer.core.curve import curve_tolerance, flatten_contour
from stencilizer.core.geometry import point_in_polygon, signed_area
from stencilizer.domain import Contour, Glyph, Point


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

    def analyze(self, glyph: Glyph, upm: int = 1000) -> ContourHierarchy:
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

        contours = [flatten_contour(c, curve_tolerance(upm)) for c in glyph.contours]
        outer_contours, inner_contours, contour_is_outer = self._classify_contours(contours)

        # Build complete nesting tree
        nesting_tree, nested_outers = self._build_nesting_tree(contours, contour_is_outer)

        containment, islands = self._find_islands(nesting_tree, inner_contours)

        return ContourHierarchy(
            outer_contours=outer_contours,
            inner_contours=inner_contours,
            containment=containment,
            islands=islands,
            nesting_tree=nesting_tree,
            nested_outers=nested_outers,
        )

    @staticmethod
    def _classify_contours(
        contours: list[Contour],
    ) -> tuple[list[int], list[int], dict[int, bool]]:
        outer_contours: list[int] = []
        inner_contours: list[int] = []
        contour_is_outer: dict[int, bool] = {}

        # TrueType convention: negative area is clockwise (outer).
        for idx, contour in enumerate(contours):
            area = signed_area(contour.points)
            if area < 0:
                outer_contours.append(idx)
                contour_is_outer[idx] = True
            elif area > 0:
                inner_contours.append(idx)
                contour_is_outer[idx] = False

        return outer_contours, inner_contours, contour_is_outer

    @staticmethod
    def _find_islands(
        nesting_tree: dict[int, ContourNode], inner_contours: list[int]
    ) -> tuple[dict[int, int], list[int]]:
        """Use the established nesting tree to locate each hole's nearest outer."""
        containment: dict[int, int] = {}
        islands: list[int] = []
        for inner_idx in inner_contours:
            parent = nesting_tree[inner_idx].parent
            while parent is not None and not nesting_tree[parent].is_outer:
                parent = nesting_tree[parent].parent
            if parent is not None:
                containment[inner_idx] = parent
                islands.append(inner_idx)
        return containment, islands

    def _build_nesting_tree(
        self,
        contours: list[Contour],
        contour_is_outer: dict[int, bool],
    ) -> tuple[dict[int, ContourNode], list[int]]:
        """Build immediate-parent relationships and identify outers nested inside holes."""
        n = len(contours)
        if n == 0:
            return {}, []

        parent_map = self._find_parents(contours, contour_is_outer)

        depth_memo: dict[int, int] = {}
        nesting_tree: dict[int, ContourNode] = {}

        for idx in contour_is_outer:
            depth = _depth(idx, parent_map, depth_memo, set())
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
                    nested_outers.append(idx)

        return nesting_tree, nested_outers

    @staticmethod
    def _find_parents(
        contours: list[Contour], contour_is_outer: dict[int, bool]
    ) -> dict[int, int | None]:
        parent_map: dict[int, int | None] = {}
        for idx in contour_is_outer:
            contour = contours[idx]
            candidates = [
                other
                for other in contour_is_outer
                if other != idx and _strictly_contains(contours[other], contour)
            ]
            parent_map[idx] = (
                min(candidates, key=lambda i: abs(signed_area(contours[i].points)))
                if candidates
                else None
            )

        return parent_map


def _depth(
    idx: int,
    parents: dict[int, int | None],
    memo: dict[int, int],
    visiting: set[int],
) -> int:
    """Calculate nesting depth, severing a malformed parent cycle."""
    if idx in memo:
        return memo[idx]
    if idx in visiting:
        parents[idx] = None
        memo[idx] = 0
        return 0
    visiting.add(idx)
    parent = parents.get(idx)
    if idx not in memo:
        memo[idx] = 0 if parent is None else _depth(parent, parents, memo, visiting) + 1
    visiting.remove(idx)
    return memo[idx]


def _strictly_contains(outer: Contour, inner: Contour) -> bool:
    """Require a nested bbox, one interior vertex, and no edge contact."""
    if len(inner.points) < 3 or len(outer.points) < 3:
        return False
    ax, ay, bx, by = outer.bounding_box()
    ix, iy, jx, jy = inner.bounding_box()
    if not (ax < ix and ay < iy and jx < bx and jy < by):
        return False
    if not point_in_polygon(inner.points[0], outer.points):
        return False
    for p1, p2 in zip(inner.points, inner.points[1:] + inner.points[:1], strict=True):
        for q1, q2 in zip(outer.points, outer.points[1:] + outer.points[:1], strict=True):
            if _edges_meet(p1, p2, q1, q2):
                return False
    return True


def _edges_meet(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Test crossing or contact, rejecting disjoint edge boxes first."""
    if (
        max(p1.x, p2.x) < min(q1.x, q2.x)
        or max(q1.x, q2.x) < min(p1.x, p2.x)
        or max(p1.y, p2.y) < min(q1.y, q2.y)
        or max(q1.y, q2.y) < min(p1.y, p2.y)
    ):
        return False

    def turn(a: Point, b: Point, c: Point) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    a, b = turn(p1, p2, q1), turn(p1, p2, q2)
    c, d = turn(q1, q2, p1), turn(q1, q2, p2)
    return a * b <= 0 and c * d <= 0


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
