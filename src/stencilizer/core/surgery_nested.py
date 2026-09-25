"""Bridge nested filled contours and inverted islands."""

from stencilizer.core.surgery_context import SurgeryContext
from stencilizer.core.vertical_bridge import create_vertical_bridge_contours
from stencilizer.domain import Contour


def find_containing_hole(
    nested_outer_bbox: tuple[float, float, float, float], contours: list[Contour]
) -> tuple[Contour | None, int | None]:
    """Find the first CCW hole containing the nested outer bounding box."""
    for i, contour in enumerate(contours):
        if contour.signed_area() > 0:
            bbox = contour.bounding_box()
            if (
                bbox[0] < nested_outer_bbox[0]
                and bbox[2] > nested_outer_bbox[2]
                and bbox[1] < nested_outer_bbox[1]
                and bbox[3] > nested_outer_bbox[3]
            ):
                return contour, i
    return None, None


def _bridge_gap(
    ctx: SurgeryContext, bbox: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    pieces = [(i, c.bounding_box()) for i, c in enumerate(ctx.contours) if c.signed_area() < 0]
    if len(pieces) < 2:
        return None
    pieces.sort(key=lambda p: p[1][0])
    min_gap = ctx.geometry.scaled("nested_outer_gap", ctx.upm)
    for i in range(len(pieces) - 1):
        curr_max_x = pieces[i][1][2]
        next_min_x = pieces[i + 1][1][0]
        if next_min_x > curr_max_x + min_gap and bbox[0] < curr_max_x and bbox[2] > next_min_x:
            return curr_max_x, next_min_x
    return None


def _nested_children(ctx: SurgeryContext, outer_idx: int) -> list[int]:
    children: list[int] = []
    tree = ctx.hierarchy.nesting_tree
    if tree:
        node = tree.get(outer_idx)
        if node:
            for child_idx in node.children:
                child_node = tree.get(child_idx)
                if child_node and not child_node.is_outer:
                    children.append(child_idx)
    return children


def _split_child(
    ctx: SurgeryContext, outer_idx: int, child_idx: int, gap: tuple[float, float]
) -> bool:
    outer = ctx.glyph.contours[outer_idx]
    child = ctx.glyph.contours[child_idx]
    bbox = child.bounding_box()
    if not (bbox[0] < gap[0] and bbox[2] > gap[1]):
        return False
    center_x = (gap[0] + gap[1]) / 2
    half_width = (gap[1] - gap[0]) / 2
    result = create_vertical_bridge_contours(
        child,
        outer,
        center_x,
        half_width,
        bbox[1],
        bbox[3],
        all_contours=ctx.glyph.contours,
        processed_nested=None,
        epsilon=ctx.geometry.get_line_epsilon(ctx.upm),
        bridge_tolerance=ctx.geometry.get_contour_gap(ctx.upm),
        point_tolerance=ctx.geometry.get_point_dedup_tolerance(ctx.upm),
    )
    if result == [outer, child]:
        return False
    ctx.contours.extend(result)
    ctx.processed.add(outer_idx)
    ctx.processed.add(child_idx)
    ctx.record_bridge(child_idx)
    ctx.bridged.add(outer_idx)
    return True


def _merge_child(ctx: SurgeryContext, outer_idx: int, child_idx: int) -> None:
    outer = ctx.glyph.contours[outer_idx]
    child = ctx.glyph.contours[child_idx]
    nested: list[Contour] = []
    merged = ctx.merge(child, outer, nested)
    if len(merged) >= 1 and merged != [outer, child]:
        ctx.contours.extend(merged)
        ctx.track_nested(nested)
        ctx.processed.add(outer_idx)
        ctx.processed.add(child_idx)
        ctx.record_bridge(child_idx)
        ctx.bridged.add(outer_idx)


def _process_children(
    ctx: SurgeryContext,
    outer_idx: int,
    children: list[int],
    gap: tuple[float, float] | None,
) -> None:
    for child_idx in children:
        if child_idx in ctx.processed:
            continue
        if gap is not None and _split_child(ctx, outer_idx, child_idx, gap):
            continue
        _merge_child(ctx, outer_idx, child_idx)


def _process_inverted(
    ctx: SurgeryContext,
    outer_idx: int,
    bbox: tuple[float, float, float, float],
) -> None:
    outer = ctx.glyph.contours[outer_idx]
    hole, hole_idx = find_containing_hole(bbox, ctx.contours)
    if hole is None or hole_idx is None:
        return
    nested: list[Contour] = []
    merged = ctx.merge(outer, hole, nested)
    if len(merged) >= 1 and merged != [hole, outer]:
        ctx.contours = ctx.contours[:hole_idx] + merged + ctx.contours[hole_idx + 1 :]
        ctx.track_nested(nested)
        ctx.processed.add(outer_idx)
        ctx.record_bridge(outer_idx)


def process_nested(ctx: SurgeryContext) -> None:
    """Process nested outers after primary island groups."""
    for outer_idx in ctx.hierarchy.nested_outers or []:
        if outer_idx in ctx.processed:
            continue
        bbox = ctx.glyph.contours[outer_idx].bounding_box()
        tree = ctx.hierarchy.nesting_tree
        parent_idx = tree[outer_idx].parent if tree else None
        if parent_idx is None:
            continue
        gap = _bridge_gap(ctx, bbox)
        children = _nested_children(ctx, outer_idx)
        if children:
            _process_children(ctx, outer_idx, children, gap)
        else:
            _process_inverted(ctx, outer_idx, bbox)
