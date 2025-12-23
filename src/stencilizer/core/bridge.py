"""Bridge placement algorithm for connecting islands to outer contours.

This module implements the core bridge placement logic:
- Finding optimal bridge positions between inner and outer contours
- Scoring candidates based on length, perpendicularity, and position
- Generating bridge geometry as rectangular connectors

Key classes:
- BridgePlacer: Finds and scores bridge candidates
- BridgeGenerator: Creates bridge geometry from specifications
"""

import math

from stencilizer.config import BridgeConfig, BridgePosition
from stencilizer.core.geometry import nearest_point_on_contour, perpendicular_direction
from stencilizer.domain import BridgeGeometry, BridgeSpec, Contour, Glyph, Point


class BridgePlacer:
    """Finds optimal bridge positions to connect islands to outer contours.

    Uses a multi-stage approach:
    1. Sample candidate points around island perimeter
    2. Find nearest point on outer contour for each candidate
    3. Score each candidate based on length, perpendicularity, and position
    4. Select best candidates ensuring minimum spacing
    """

    def __init__(self, config: BridgeConfig) -> None:
        """Initialize bridge placer with configuration.

        Args:
            config: Bridge configuration with width, position, etc.
        """
        self.config = config

    def find_candidates(
        self, island: Contour, outer: Contour, island_idx: int, outer_idx: int, count: int
    ) -> list[BridgeSpec]:
        """Find candidate bridge positions connecting island to outer contour.

        Args:
            island: Inner contour (island/hole) to connect
            outer: Outer contour to connect to
            island_idx: Index of island in glyph's contour list
            outer_idx: Index of outer contour in glyph's contour list
            count: Number of candidate points to sample around island

        Returns:
            List of bridge specifications with endpoints and initial width
        """
        candidates: list[BridgeSpec] = []

        # Sample points around island perimeter
        sampled_points = island.sample_points(count)

        for inner_point in sampled_points:
            # Find nearest point on outer contour
            outer_point, _ = nearest_point_on_contour(inner_point, outer)

            # Create bridge spec with estimated width (will be refined later)
            spec = BridgeSpec(
                inner_contour_idx=island_idx,
                outer_contour_idx=outer_idx,
                inner_point=inner_point,
                outer_point=outer_point,
                width=0.0,  # Width will be set during generation
                score=0.0,  # Score will be calculated separately
            )

            candidates.append(spec)

        return candidates

    def score_candidate(self, spec: BridgeSpec, glyph: Glyph, upm: int) -> float:
        """Score a bridge candidate based on multiple criteria.

        Scoring factors:
        - Bridge length (LONGER is better - ensures full stroke cut-through)
        - Position preference (e.g., prefer top/bottom/left/right)

        Args:
            spec: Bridge specification to score
            glyph: Glyph containing the contours
            upm: Units per em (font UPM)

        Returns:
            Score value (higher is better)
        """
        # Calculate bridge length (stroke width at this location)
        dx = spec.outer_point.x - spec.inner_point.x
        dy = spec.outer_point.y - spec.inner_point.y
        length = math.hypot(dx, dy)

        # Handle zero-length bridge (degenerate case)
        if length < 1e-10:
            return -1000.0  # Very poor score

        # Length score: LONGER is better (ensures full stroke cut-through)
        # Normalized so typical stroke widths (5-15% of UPM) get good scores
        normalized_length = length / upm
        length_score = min(normalized_length * 10.0, 1.0)  # Cap at 1.0

        # Position preference score
        position_score = self._score_position(spec, glyph)

        # Combine scores with weights - position more important for placement,
        # but length ensures we pick thick stroke sections
        total_score = length_score * 0.4 + position_score * 0.6

        return total_score

    def _score_position(self, spec: BridgeSpec, glyph: Glyph) -> float:  # noqa: ARG002
        """Score based on position preference.

        Args:
            spec: Bridge specification
            glyph: Glyph containing the contours (unused but kept for API consistency)

        Returns:
            Position score between 0.0 and 1.0
        """
        # Calculate bridge direction
        dx = spec.outer_point.x - spec.inner_point.x
        dy = spec.outer_point.y - spec.inner_point.y
        angle = math.atan2(dy, dx)

        if self.config.position_preference == BridgePosition.AUTO:
            # AUTO mode: prefer bottom, then sides, avoid top
            # Bottom (angle near -pi/2): score 1.0
            # Sides (angle near 0 or pi): score 0.7
            # Top (angle near pi/2): score 0.2
            if angle < -math.pi / 4:  # Bottom half
                # Closer to -pi/2 is better
                bottom_factor = abs(angle + math.pi / 2) / (math.pi / 4)
                return 1.0 - 0.3 * min(bottom_factor, 1.0)
            elif angle > math.pi / 4:  # Top half
                # Penalize top positions
                top_factor = abs(angle - math.pi / 2) / (math.pi / 4)
                return 0.2 + 0.3 * min(top_factor, 1.0)
            else:  # Sides
                # Good but not as good as bottom
                return 0.7

        # Score based on preference
        if self.config.position_preference == BridgePosition.TOP:
            # Prefer upward bridges (angle near pi/2)
            target_angle = math.pi / 2
        elif self.config.position_preference == BridgePosition.BOTTOM:
            # Prefer downward bridges (angle near -pi/2)
            target_angle = -math.pi / 2
        elif self.config.position_preference == BridgePosition.LEFT:
            # Prefer leftward bridges (angle near pi)
            target_angle = math.pi
        elif self.config.position_preference == BridgePosition.RIGHT:
            # Prefer rightward bridges (angle near 0)
            target_angle = 0.0
        elif self.config.position_preference == BridgePosition.TOP_BOTTOM:
            # Prefer vertical bridges (up or down)
            # Use minimum distance from either pi/2 or -pi/2
            target_angle = math.pi / 2 if abs(angle - math.pi / 2) < abs(angle + math.pi / 2) else -math.pi / 2
        else:
            return 0.5  # Neutral for unknown preferences

        # Calculate angular difference
        angle_diff = abs(angle - target_angle)
        # Normalize to [0, pi]
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # Convert to score: 0 diff = 1.0, pi diff = 0.0
        position_score = 1.0 - (angle_diff / math.pi)

        return position_score

    def select_bridges(
        self, candidates: list[BridgeSpec], min_count: int, min_spacing_factor: float = 0.15
    ) -> list[BridgeSpec]:
        """Select best bridges from candidates using greedy algorithm.

        Ensures selected bridges are spaced apart to avoid clustering.

        Args:
            candidates: List of scored bridge candidates
            min_count: Minimum number of bridges to select
            min_spacing_factor: Minimum spacing as fraction of island perimeter

        Returns:
            List of selected bridge specifications
        """
        if not candidates:
            return []

        # Sort candidates by score (descending)
        sorted_candidates = sorted(candidates, key=lambda spec: spec.score, reverse=True)

        selected: list[BridgeSpec] = []

        # Always select the best candidate first
        if sorted_candidates:
            selected.append(sorted_candidates[0])

        # Greedily select additional candidates
        for candidate in sorted_candidates[1:]:
            if len(selected) >= min_count:
                # We have enough bridges
                break

            # Check if candidate is far enough from all selected bridges
            if self._is_far_enough(candidate, selected, min_spacing_factor):
                selected.append(candidate)

        return selected

    def _is_far_enough(
        self, candidate: BridgeSpec, selected: list[BridgeSpec], min_spacing_factor: float
    ) -> bool:
        """Check if candidate is far enough from all selected bridges.

        Args:
            candidate: Candidate to check
            selected: Already selected bridges
            min_spacing_factor: Minimum spacing factor

        Returns:
            True if candidate is far enough from all selected bridges
        """
        # Calculate minimum distance threshold
        # Use Euclidean distance between inner points
        for bridge in selected:
            dx = candidate.inner_point.x - bridge.inner_point.x
            dy = candidate.inner_point.y - bridge.inner_point.y
            distance = math.hypot(dx, dy)

            # Calculate average bridge length for threshold
            dx_cand = candidate.outer_point.x - candidate.inner_point.x
            dy_cand = candidate.outer_point.y - candidate.inner_point.y
            cand_length = math.hypot(dx_cand, dy_cand)

            dx_bridge = bridge.outer_point.x - bridge.inner_point.x
            dy_bridge = bridge.outer_point.y - bridge.inner_point.y
            bridge_length = math.hypot(dx_bridge, dy_bridge)

            avg_length = (cand_length + bridge_length) / 2.0

            # Use average bridge length scaled by spacing factor
            min_distance = avg_length * min_spacing_factor

            if distance < min_distance:
                return False

        return True


class BridgeGenerator:
    """Generates bridge geometry from specifications.

    Creates rectangular bridge shapes that connect inner and outer contours.
    """

    def __init__(self, config: BridgeConfig, reference_stroke_width: float | None = None) -> None:
        """Initialize bridge generator with configuration.

        Args:
            config: Bridge configuration with width percentage, etc.
            reference_stroke_width: Fixed stroke width to use for all bridges.
                If None, uses local stroke width (inconsistent).
        """
        self.config = config
        self.reference_stroke_width = reference_stroke_width

    def generate_geometry(self, spec: BridgeSpec, upm: int) -> BridgeGeometry:  # noqa: ARG002
        """Generate bridge geometry as a rectangular connector.

        Creates a 4-corner rectangle perpendicular to the bridge direction.
        The bridge is inset slightly from the contours to avoid precision issues.

        Args:
            spec: Bridge specification with endpoints
            upm: Units per em (font UPM) - unused, kept for API compatibility

        Returns:
            Bridge geometry with 4 vertices forming the bridge rectangle:
            - v1, v2: inner side of bridge (on/near inner contour)
            - v3, v4: outer side of bridge (on/near outer contour)

        Raises:
            ValueError: If bridge has zero length
        """
        # Calculate stroke width (distance between inner and outer points)
        dx = spec.outer_point.x - spec.inner_point.x
        dy = spec.outer_point.y - spec.inner_point.y
        stroke_width = math.hypot(dx, dy)

        if stroke_width < 1e-6:
            raise ValueError("Bridge has zero length")

        # Bridge width: use reference stroke width for consistency, or local if not set
        base_stroke = self.reference_stroke_width if self.reference_stroke_width else stroke_width
        width = (self.config.width_percent / 100.0) * base_stroke

        # Calculate inset distance (pulls endpoints slightly inward from contours)
        inset = (self.config.inset_percent / 100.0) * stroke_width

        # Normalize direction vector
        nx = dx / stroke_width
        ny = dy / stroke_width

        # Get perpendicular direction
        px, py = perpendicular_direction(spec.inner_point, spec.outer_point)

        # Calculate half-width offsets
        half_width = width / 2.0
        offset_x = px * half_width
        offset_y = py * half_width

        # Calculate inset endpoints (pulled slightly inward from the actual contour points)
        # Inner point moves toward outer (along bridge direction)
        inner_inset_x = spec.inner_point.x + nx * inset
        inner_inset_y = spec.inner_point.y + ny * inset
        # Outer point moves toward inner (opposite to bridge direction)
        outer_inset_x = spec.outer_point.x - nx * inset
        outer_inset_y = spec.outer_point.y - ny * inset

        # Create 4 vertices:
        # v1, v2 are on the inner side (closer to inner contour)
        # v3, v4 are on the outer side (closer to outer contour)
        v1 = Point(inner_inset_x - offset_x, inner_inset_y - offset_y)
        v2 = Point(inner_inset_x + offset_x, inner_inset_y + offset_y)
        v3 = Point(outer_inset_x + offset_x, outer_inset_y + offset_y)
        v4 = Point(outer_inset_x - offset_x, outer_inset_y - offset_y)

        # Create updated spec with actual width
        updated_spec = BridgeSpec(
            inner_contour_idx=spec.inner_contour_idx,
            outer_contour_idx=spec.outer_contour_idx,
            inner_point=spec.inner_point,
            outer_point=spec.outer_point,
            width=width,
            score=spec.score,
        )

        return BridgeGeometry(vertices=(v1, v2, v3, v4), spec=updated_spec)
