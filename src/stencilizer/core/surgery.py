"""Contour surgery module for creating stencil bridges.

This module implements contour surgery by merging outer and inner contours
with bridge connections. Instead of adding hole contours (which create
artifacts in empty space), we modify the outer contour to have notches
that connect to the inner contour at bridge positions.

Key classes:
- GlyphTransformer: Transforms glyphs with bridge gaps

The actual bridge creation logic has been split into separate modules:
- geometry.py: Basic geometry utilities
- vertical_bridge.py: Vertical bridge (TOP/BOTTOM gaps)
- horizontal_bridge.py: Horizontal bridge (LEFT/RIGHT gaps)
- multi_island.py: Multi-island handling
- merger.py: ContourMerger class
"""

from typing import TYPE_CHECKING

from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.merger import ContourMerger
from stencilizer.core.multi_island import merge_multi_island_vertical
from stencilizer.domain import Contour, Glyph

if TYPE_CHECKING:
    from stencilizer.config.settings import BridgeConfig


__all__ = ["ContourMerger", "GlyphTransformer"]


class GlyphTransformer:
    """Transforms glyphs by merging contours with bridge gaps.

    For each island (inner contour), merges it with its parent outer contour
    to create new contours with bridge gaps built in.
    """

    def __init__(
        self,
        analyzer: GlyphAnalyzer,
        placer: BridgePlacer,
        generator: BridgeGenerator,
        merger: ContourMerger | None = None,
        bridge_config: "BridgeConfig | None" = None,
    ) -> None:
        """Initialize glyph transformer with required services."""
        self.analyzer = analyzer
        self.placer = placer
        self.generator = generator
        self.merger = merger if merger is not None else ContourMerger()
        self.bridge_config = bridge_config

    def transform(self, glyph: Glyph, upm: int = 1000) -> Glyph:
        """Transform a glyph by merging contours with bridge gaps.

        Args:
            glyph: Glyph to transform
            upm: Units per em for the font

        Returns:
            New glyph with bridge gaps built into contours
        """
        hierarchy = self.analyzer.analyze(glyph)

        if not hierarchy.islands:
            return glyph

        # Calculate bridge width from config
        reference_stroke = upm * 0.1
        bridge_width = (self.placer.config.width_percent / 100.0) * reference_stroke

        # Group islands by their parent outer contour
        islands_by_parent: dict[int, list[int]] = {}
        for island_idx in hierarchy.islands:
            parent_idx = hierarchy.containment.get(island_idx)
            if parent_idx is not None:
                if parent_idx not in islands_by_parent:
                    islands_by_parent[parent_idx] = []
                islands_by_parent[parent_idx].append(island_idx)

        # Track which contours have been processed
        processed_indices: set[int] = set()
        new_contours: list[Contour] = []

        # Create mapping from contour identity to index for tracking nested contours
        contour_to_idx = {id(c): i for i, c in enumerate(glyph.contours)}

        # Process each parent's islands sequentially
        for parent_idx, island_indices in islands_by_parent.items():
            if parent_idx in processed_indices:
                continue

            # Sort islands by Y position (top-to-bottom) for consistent merge order
            island_indices_sorted = sorted(
                island_indices,
                key=lambda idx: -glyph.contours[idx].bounding_box()[3],  # -max_y (top first)
            )

            # Check if islands are vertically stacked or horizontally arranged
            is_vertically_stacked = False
            is_horizontally_arranged = False
            if len(island_indices_sorted) > 1:
                bboxes = [glyph.contours[idx].bounding_box() for idx in island_indices_sorted]

                # Check for vertical stacking (one above another)
                for i in range(len(bboxes) - 1):
                    upper_center_y = (bboxes[i][1] + bboxes[i][3]) / 2
                    lower_max_y = bboxes[i + 1][3]
                    if lower_max_y < upper_center_y:
                        is_vertically_stacked = True
                        break

                # Check for horizontal arrangement (side by side at similar Y)
                if not is_vertically_stacked:
                    bboxes_by_x = sorted(bboxes, key=lambda b: b[0])
                    for i in range(len(bboxes_by_x) - 1):
                        left_center_x = (bboxes_by_x[i][0] + bboxes_by_x[i][2]) / 2
                        right_min_x = bboxes_by_x[i + 1][0]
                        if right_min_x > left_center_x:
                            y_overlap = (
                                min(bboxes_by_x[i][3], bboxes_by_x[i + 1][3]) -
                                max(bboxes_by_x[i][1], bboxes_by_x[i + 1][1])
                            )
                            if y_overlap > 0:
                                is_horizontally_arranged = True
                                break

            # For vertically-stacked multi-island glyphs (like 8, B)
            if is_vertically_stacked and len(island_indices_sorted) > 1:
                inners = [glyph.contours[idx] for idx in island_indices_sorted]
                outer = glyph.contours[parent_idx]

                use_spanning = self.bridge_config.use_spanning_bridges if self.bridge_config else True

                if use_spanning:
                    result = merge_multi_island_vertical(
                        outer, inners, bridge_width, all_contours=glyph.contours
                    )
                    if len(result) >= 4 and result[0] != outer:
                        new_contours.extend(result)
                        processed_indices.add(parent_idx)
                        processed_indices.update(island_indices_sorted)
                    else:
                        use_spanning = False

                if not use_spanning:
                    current_pieces: list[Contour] = [glyph.contours[parent_idx]]

                    for island_idx in island_indices_sorted:
                        if island_idx in processed_indices:
                            continue

                        inner = glyph.contours[island_idx]
                        inner_bbox = inner.bounding_box()
                        inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                        inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                        containing_piece_idx = None
                        for i, piece in enumerate(current_pieces):
                            if piece.contains_point(inner_center_x, inner_center_y):
                                containing_piece_idx = i
                                break

                        if containing_piece_idx is not None:
                            piece = current_pieces[containing_piece_idx]
                            processed_nested: list[Contour] = []
                            result = self.merger.merge_contours_with_bridges(
                                inner, piece, bridge_width, force_horizontal=True,
                                all_contours=glyph.contours,
                                processed_nested=processed_nested
                            )
                            if len(result) >= 1 and result != [piece, inner]:
                                current_pieces = (
                                    current_pieces[:containing_piece_idx]
                                    + result
                                    + current_pieces[containing_piece_idx + 1:]
                                )
                                processed_indices.add(island_idx)
                                for c in processed_nested:
                                    idx = contour_to_idx.get(id(c))
                                    if idx is not None:
                                        processed_indices.add(idx)

                    new_contours.extend(current_pieces)
                    processed_indices.add(parent_idx)
            elif is_horizontally_arranged and len(island_indices_sorted) > 1:
                # For horizontally-arranged multi-island glyphs (like Phi)
                inners = [glyph.contours[idx] for idx in island_indices]
                outer = glyph.contours[parent_idx]

                use_spanning = self.bridge_config.use_spanning_bridges if self.bridge_config else True

                if use_spanning:
                    # Try spanning horizontal bridges
                    from .horizontal_multi_island import merge_multi_island_horizontal
                    result = merge_multi_island_horizontal(
                        outer, inners, bridge_width, all_contours=glyph.contours
                    )
                    if len(result) >= 4 and result[0] != outer:
                        new_contours.extend(result)
                        processed_indices.add(parent_idx)
                        processed_indices.update(island_indices)
                        continue

                # Fallback to individual processing
                current_pieces = [glyph.contours[parent_idx]]

                island_indices_by_x = sorted(
                    island_indices,
                    key=lambda idx: glyph.contours[idx].bounding_box()[0],
                )

                for island_idx in island_indices_by_x:
                    if island_idx in processed_indices:
                        continue

                    inner = glyph.contours[island_idx]
                    inner_bbox = inner.bounding_box()
                    inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                    inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                    containing_piece_idx = None
                    for i, piece in enumerate(current_pieces):
                        if piece.contains_point(inner_center_x, inner_center_y):
                            containing_piece_idx = i
                            break

                    if containing_piece_idx is not None:
                        piece = current_pieces[containing_piece_idx]
                        horiz_nested: list[Contour] = []
                        result = self.merger.merge_contours_with_bridges(
                            inner, piece, bridge_width, force_vertical=True,
                            all_contours=glyph.contours,
                            processed_nested=horiz_nested
                        )
                        if len(result) >= 1 and result != [piece, inner]:
                            current_pieces = (
                                current_pieces[:containing_piece_idx]
                                + result
                                + current_pieces[containing_piece_idx + 1:]
                            )
                            processed_indices.add(island_idx)
                            for c in horiz_nested:
                                idx = contour_to_idx.get(id(c))
                                if idx is not None:
                                    processed_indices.add(idx)

                new_contours.extend(current_pieces)
                processed_indices.add(parent_idx)
            else:
                # Single island - process normally
                current_pieces = [glyph.contours[parent_idx]]

                for island_idx in island_indices_sorted:
                    if island_idx in processed_indices:
                        continue

                    inner = glyph.contours[island_idx]
                    inner_bbox = inner.bounding_box()
                    inner_center_x = (inner_bbox[0] + inner_bbox[2]) / 2
                    inner_center_y = (inner_bbox[1] + inner_bbox[3]) / 2

                    # Find which current piece contains this island
                    containing_piece_idx = None
                    for i, piece in enumerate(current_pieces):
                        if piece.contains_point(inner_center_x, inner_center_y):
                            containing_piece_idx = i
                            break

                    if containing_piece_idx is None:
                        continue

                    # Merge this island with the containing piece
                    outer = current_pieces[containing_piece_idx]
                    single_nested: list[Contour] = []
                    merged = self.merger.merge_contours_with_bridges(
                        inner=inner,
                        outer=outer,
                        bridge_width=bridge_width,
                        all_contours=glyph.contours,
                        processed_nested=single_nested,
                    )

                    if len(merged) >= 1 and merged != [outer, inner]:
                        current_pieces = (
                            current_pieces[:containing_piece_idx]
                            + merged
                            + current_pieces[containing_piece_idx + 1:]
                        )
                        processed_indices.add(island_idx)
                        for c in single_nested:
                            idx = contour_to_idx.get(id(c))
                            if idx is not None:
                                processed_indices.add(idx)

                new_contours.extend(current_pieces)
                processed_indices.add(parent_idx)

        # Add any unprocessed contours (non-islands, failed merges, orphans)
        for i, contour in enumerate(glyph.contours):
            if i not in processed_indices:
                new_contours.append(contour)

        return Glyph(metadata=glyph.metadata, contours=new_contours)
