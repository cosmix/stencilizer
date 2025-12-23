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

        # Build set of "protected" contours: nested outers WITH CHILDREN and their descendants
        # Only protect nested outers that have their own holes (like R in ® or P in ℗).
        # Childless nested outers (like numbers in circled numbers) should NOT be protected -
        # they need to be split during outer structure bridging to create proper bridges.
        protected_indices: set[int] = set()
        if hierarchy.nested_outers and hierarchy.nesting_tree:
            def add_descendants(idx: int) -> None:
                protected_indices.add(idx)
                if hierarchy.nesting_tree is not None:
                    node = hierarchy.nesting_tree.get(idx)
                    if node:
                        for child_idx in node.children:
                            add_descendants(child_idx)
            for nested_outer_idx in hierarchy.nested_outers:
                node = hierarchy.nesting_tree.get(nested_outer_idx)
                # Only protect if this nested outer has children (holes inside it)
                if node and node.children:
                    add_descendants(nested_outer_idx)

        # Group islands by their parent outer contour
        # Skip parents that are nested outers (they're handled separately)
        islands_by_parent: dict[int, list[int]] = {}
        for island_idx in hierarchy.islands:
            parent_idx = hierarchy.containment.get(island_idx)
            if parent_idx is not None:
                # Skip if parent is protected (nested outer or descendant)
                if parent_idx in protected_indices:
                    continue
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
            # Compare X separation vs Y separation to determine primary arrangement
            is_vertically_stacked = False
            is_horizontally_arranged = False
            if len(island_indices_sorted) > 1:
                bboxes = [glyph.contours[idx].bounding_box() for idx in island_indices_sorted]

                # Calculate gap between islands in X and Y
                # For X: sort by X and find gap between consecutive islands
                # For Y: sort by Y and find gap between consecutive islands
                bboxes_by_x = sorted(bboxes, key=lambda b: b[0])
                bboxes_by_y = sorted(bboxes, key=lambda b: b[1])

                max_x_gap = 0
                for i in range(len(bboxes_by_x) - 1):
                    gap = bboxes_by_x[i + 1][0] - bboxes_by_x[i][2]  # next_min_x - curr_max_x
                    if gap > max_x_gap:
                        max_x_gap = gap

                max_y_gap = 0
                for i in range(len(bboxes_by_y) - 1):
                    gap = bboxes_by_y[i + 1][1] - bboxes_by_y[i][3]  # next_min_y - curr_max_y
                    if gap > max_y_gap:
                        max_y_gap = gap

                # Determine arrangement based on which gap is larger
                # If Y gap > X gap, islands are vertically stacked (use vertical bridges)
                # If X gap > Y gap, islands are horizontally arranged (use horizontal bridges)
                if max_y_gap > max_x_gap and max_y_gap > 0:
                    is_vertically_stacked = True
                elif max_x_gap > max_y_gap and max_x_gap > 0:
                    is_horizontally_arranged = True
                else:
                    # Fallback: check overlap patterns
                    # Significant Y overlap = horizontally arranged
                    # Significant X overlap = vertically stacked
                    y_overlap = min(b[3] for b in bboxes) - max(b[1] for b in bboxes)
                    x_overlap = min(b[2] for b in bboxes) - max(b[0] for b in bboxes)
                    if x_overlap > y_overlap:
                        is_vertically_stacked = True
                    else:
                        is_horizontally_arranged = True

            # For vertically-stacked multi-island glyphs (like 8, B, Θ)
            if is_vertically_stacked and len(island_indices_sorted) > 1:
                inners = [glyph.contours[idx] for idx in island_indices_sorted]
                outer = glyph.contours[parent_idx]

                use_spanning = self.bridge_config.use_spanning_bridges if self.bridge_config else True

                if use_spanning:
                    multi_nested: list[Contour] = []
                    # Pass ALL contours (not filtered) so nested structures can be found and split
                    # The processed_nested tracking prevents double-processing
                    result = merge_multi_island_vertical(
                        outer, inners, bridge_width, all_contours=glyph.contours,
                        processed_nested=multi_nested
                    )
                    if len(result) >= 4 and result[0] != outer:
                        new_contours.extend(result)
                        processed_indices.add(parent_idx)
                        processed_indices.update(island_indices_sorted)
                        # Track any nested contours that were processed
                        for c in multi_nested:
                            idx = contour_to_idx.get(id(c))
                            if idx is not None:
                                processed_indices.add(idx)
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
                            # Pass ALL contours so nested structures can be found and split
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
                    horiz_multi_nested: list[Contour] = []
                    # Pass ALL contours so nested structures can be found and split
                    result = merge_multi_island_horizontal(
                        outer, inners, bridge_width, all_contours=glyph.contours,
                        processed_nested=horiz_multi_nested
                    )
                    if len(result) >= 4 and result[0] != outer:
                        new_contours.extend(result)
                        processed_indices.add(parent_idx)
                        processed_indices.update(island_indices)
                        # Track any nested contours that were processed
                        for c in horiz_multi_nested:
                            idx = contour_to_idx.get(id(c))
                            if idx is not None:
                                processed_indices.add(idx)
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
                        # Pass ALL contours so nested structures can be found and split
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
                    # Pass ALL contours so nested structures can be found and split
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

        # Process nested outers (CW contours inside CCW holes like R in ®)
        if hierarchy.nested_outers:
            for nested_outer_idx in hierarchy.nested_outers:
                if nested_outer_idx in processed_indices:
                    continue

                nested_outer = glyph.contours[nested_outer_idx]
                nested_outer_bbox = nested_outer.bounding_box()

                # Find the parent CCW hole this nested outer is inside
                parent_idx = hierarchy.nesting_tree[nested_outer_idx].parent if hierarchy.nesting_tree else None
                if parent_idx is None:
                    continue

                # Check if nested outer spans a bridge gap in the processed contours.
                # Find gaps between adjacent CW (filled) pieces to detect bridge gaps.
                cw_pieces = [(i, c.bounding_box()) for i, c in enumerate(new_contours)
                             if c.signed_area() < 0]  # CW = filled

                bridge_gap_x_min = None
                bridge_gap_x_max = None
                if len(cw_pieces) >= 2:
                    # Sort by X position and find gaps
                    cw_pieces.sort(key=lambda p: p[1][0])
                    for i in range(len(cw_pieces) - 1):
                        curr_max_x = cw_pieces[i][1][2]
                        next_min_x = cw_pieces[i + 1][1][0]
                        if next_min_x > curr_max_x + 10:  # Gap > 10 units
                            # Check if nested outer spans this gap
                            if (nested_outer_bbox[0] < curr_max_x and
                                nested_outer_bbox[2] > next_min_x):
                                bridge_gap_x_min = curr_max_x
                                bridge_gap_x_max = next_min_x
                                break

                # First, process any islands (CCW holes) of this nested outer
                nested_children = []
                if hierarchy.nesting_tree:
                    node = hierarchy.nesting_tree.get(nested_outer_idx)
                    if node:
                        for child_idx in node.children:
                            child_node = hierarchy.nesting_tree.get(child_idx)
                            if child_node and not child_node.is_outer:
                                nested_children.append(child_idx)

                if nested_children:
                    # Check if we need to split vertically first (to match outer gap)
                    if bridge_gap_x_min is not None and bridge_gap_x_max is not None:
                        # Split nested outer and children vertically to match gap
                        from .vertical_bridge import create_vertical_bridge_contours
                        gap_center_x = (bridge_gap_x_min + bridge_gap_x_max) / 2
                        gap_half_width = (bridge_gap_x_max - bridge_gap_x_min) / 2

                        for child_idx in nested_children:
                            if child_idx in processed_indices:
                                continue
                            child = glyph.contours[child_idx]
                            child_bbox = child.bounding_box()

                            # Check if child spans the bridge gap
                            child_spans_gap = (child_bbox[0] < bridge_gap_x_min and
                                               child_bbox[2] > bridge_gap_x_max)

                            if child_spans_gap:
                                # Child spans gap - split vertically to match outer structure
                                vert_result = create_vertical_bridge_contours(
                                    child, nested_outer, gap_center_x, gap_half_width,
                                    child_bbox[1], child_bbox[3],
                                    all_contours=glyph.contours, processed_nested=None
                                )
                                if vert_result != [nested_outer, child]:
                                    new_contours.extend(vert_result)
                                    processed_indices.add(nested_outer_idx)
                                    processed_indices.add(child_idx)
                                    continue

                            # Child doesn't span gap OR vertical bridging failed
                            # Fall back to normal bridge processing
                            nested_processed: list[Contour] = []
                            merged = self.merger.merge_contours_with_bridges(
                                inner=child,
                                outer=nested_outer,
                                bridge_width=bridge_width,
                                all_contours=glyph.contours,
                                processed_nested=nested_processed,
                            )
                            if len(merged) >= 1 and merged != [nested_outer, child]:
                                new_contours.extend(merged)
                                processed_indices.add(nested_outer_idx)
                                processed_indices.add(child_idx)
                                for c in nested_processed:
                                    idx = contour_to_idx.get(id(c))
                                    if idx is not None:
                                        processed_indices.add(idx)
                            else:
                                # Last resort: add unchanged (should rarely happen)
                                new_contours.append(nested_outer)
                                new_contours.append(child)
                                processed_indices.add(nested_outer_idx)
                                processed_indices.add(child_idx)
                    else:
                        # No gap to match - use normal bridge processing
                        for child_idx in nested_children:
                            if child_idx in processed_indices:
                                continue
                            child = glyph.contours[child_idx]
                            nested_processed: list[Contour] = []
                            merged = self.merger.merge_contours_with_bridges(
                                inner=child,
                                outer=nested_outer,
                                bridge_width=bridge_width,
                                all_contours=glyph.contours,
                                processed_nested=nested_processed,
                            )
                            if len(merged) >= 1 and merged != [nested_outer, child]:
                                # The nested outer was split/bridged
                                new_contours.extend(merged)
                                processed_indices.add(nested_outer_idx)
                                processed_indices.add(child_idx)
                                for c in nested_processed:
                                    idx = contour_to_idx.get(id(c))
                                    if idx is not None:
                                        processed_indices.add(idx)
                            else:
                                # Couldn't bridge - add both as-is
                                new_contours.append(nested_outer)
                                new_contours.append(child)
                                processed_indices.add(nested_outer_idx)
                                processed_indices.add(child_idx)
                else:
                    # Nested outer with no children - just add it
                    # (It should have been split during outer structure bridging
                    # if it wasn't protected. If we reach here, it was protected
                    # but has no children to process.)
                    new_contours.append(nested_outer)
                    processed_indices.add(nested_outer_idx)

        # Add any unprocessed contours (non-islands, failed merges, orphans)
        for i, contour in enumerate(glyph.contours):
            if i not in processed_indices:
                new_contours.append(contour)

        return Glyph(metadata=glyph.metadata, contours=new_contours)
