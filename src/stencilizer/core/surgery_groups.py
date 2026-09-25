"""Group and process ordinary island contours."""

from stencilizer.config.settings import BridgeDirection
from stencilizer.core.horizontal_multi_island import merge_multi_island_horizontal
from stencilizer.core.multi_island import merge_multi_island_vertical
from stencilizer.core.surgery_context import SurgeryContext
from stencilizer.domain import Contour


def protected_indices(ctx: SurgeryContext) -> set[int]:
    """Protect nested outers that have their own holes and descendants."""
    hierarchy = ctx.hierarchy
    protected: set[int] = set()
    if not hierarchy.nested_outers or not hierarchy.nesting_tree:
        return protected

    def add_descendants(idx: int) -> None:
        protected.add(idx)
        node = hierarchy.nesting_tree.get(idx) if hierarchy.nesting_tree else None
        if node:
            for child_idx in node.children:
                add_descendants(child_idx)

    for idx in hierarchy.nested_outers:
        node = hierarchy.nesting_tree.get(idx)
        if node and node.children:
            add_descendants(idx)
    return protected


def group_islands(ctx: SurgeryContext) -> dict[int, list[int]]:
    """Group bridgeable islands by their unprotected parent."""
    protected = protected_indices(ctx)
    groups: dict[int, list[int]] = {}
    for island_idx in ctx.hierarchy.islands:
        parent_idx = ctx.hierarchy.containment.get(island_idx)
        if parent_idx is not None and parent_idx not in protected:
            if parent_idx not in groups:
                groups[parent_idx] = []
            groups[parent_idx].append(island_idx)
    return groups


def arrangement(ctx: SurgeryContext, indices: list[int]) -> str:
    """Choose the dominant separation axis using the original gap tests."""
    if len(indices) <= 1:
        return "single"
    bboxes = [ctx.glyph.contours[idx].bounding_box() for idx in indices]
    by_x = sorted(bboxes, key=lambda b: b[0])
    by_y = sorted(bboxes, key=lambda b: b[1])
    max_x_gap = 0.0
    for i in range(len(by_x) - 1):
        gap = by_x[i + 1][0] - by_x[i][2]
        if gap > max_x_gap:
            max_x_gap = gap
    max_y_gap = 0.0
    for i in range(len(by_y) - 1):
        gap = by_y[i + 1][1] - by_y[i][3]
        if gap > max_y_gap:
            max_y_gap = gap
    if max_y_gap > max_x_gap and max_y_gap > 0:
        return "vertical"
    if max_x_gap > max_y_gap and max_x_gap > 0:
        return "horizontal"
    y_overlap = min(b[3] for b in bboxes) - max(b[1] for b in bboxes)
    x_overlap = min(b[2] for b in bboxes) - max(b[0] for b in bboxes)
    return "vertical" if x_overlap > y_overlap else "horizontal"


def _spanning(ctx: SurgeryContext, parent_idx: int, indices: list[int], axis: str) -> bool:
    outer = ctx.glyph.contours[parent_idx]
    inners = [ctx.glyph.contours[idx] for idx in indices]
    nested: list[Contour] = []
    kwargs = {
        "edge_margin": ctx.geometry.scaled("island_edge_margin", ctx.upm),
        "epsilon": ctx.geometry.get_line_epsilon(ctx.upm),
        "connection_tolerance": ctx.geometry.get_contour_gap(ctx.upm),
        "duplicate_tolerance": ctx.geometry.get_point_dedup_tolerance(ctx.upm),
    }
    if axis == "vertical":
        result = merge_multi_island_vertical(
            outer, inners, ctx.bridge_width, ctx.glyph.contours, nested, **kwargs
        )
    else:
        result = merge_multi_island_horizontal(
            outer, inners, ctx.bridge_width, ctx.glyph.contours, nested, **kwargs
        )
    if len(result) < 4 or result[0] == outer:
        return False
    ctx.contours.extend(result)
    ctx.processed.add(parent_idx)
    ctx.processed.update(indices)
    ctx.record_bridge(*indices)
    ctx.track_nested(nested)
    return True


def _containing_piece(pieces: list[Contour], inner: Contour) -> int | None:
    bbox = inner.bounding_box()
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    for i, piece in enumerate(pieces):
        if piece.contains_point(center_x, center_y):
            return i
    return None


def _sequential(ctx: SurgeryContext, parent_idx: int, indices: list[int], axis: str) -> None:
    pieces = [ctx.glyph.contours[parent_idx]]
    changed = False
    for island_idx in indices:
        if island_idx in ctx.processed:
            continue
        inner = ctx.glyph.contours[island_idx]
        piece_idx = _containing_piece(pieces, inner)
        if piece_idx is None:
            continue
        piece = pieces[piece_idx]
        nested: list[Contour] = []
        result = ctx.merge(
            inner,
            piece,
            nested,
            force_horizontal=axis == "vertical",
            force_vertical=axis == "horizontal",
        )
        if len(result) >= 1 and result != [piece, inner]:
            pieces = pieces[:piece_idx] + result + pieces[piece_idx + 1 :]
            ctx.processed.add(island_idx)
            ctx.record_bridge(island_idx)
            changed = True
            ctx.track_nested(nested)
    if changed:
        ctx.contours.extend(pieces)
        ctx.processed.add(parent_idx)


def _spanning_allowed(ctx: SurgeryContext, axis: str) -> bool:
    """Span along ``axis`` when the glyph's explicit direction matches it, else follow the setting."""
    if ctx.direction is BridgeDirection.AUTO:
        return ctx.use_spanning
    return ctx.direction.value == axis


def process_groups(ctx: SurgeryContext) -> None:
    """Process each parent in insertion order and islands in axis order."""
    for parent_idx, indices in group_islands(ctx).items():
        if parent_idx in ctx.processed:
            continue
        sorted_y = sorted(indices, key=lambda idx: -ctx.glyph.contours[idx].bounding_box()[3])
        axis = arrangement(ctx, sorted_y)
        if axis == "horizontal":
            if _spanning_allowed(ctx, axis) and _spanning(ctx, parent_idx, indices, axis):
                continue
            by_x = sorted(indices, key=lambda idx: ctx.glyph.contours[idx].bounding_box()[0])
            _sequential(ctx, parent_idx, by_x, axis)
        elif axis == "vertical":
            if _spanning_allowed(ctx, axis) and _spanning(ctx, parent_idx, sorted_y, axis):
                continue
            _sequential(ctx, parent_idx, sorted_y, axis)
        else:
            _sequential(ctx, parent_idx, sorted_y, axis)
