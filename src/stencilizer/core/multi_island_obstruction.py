"""Axis-aware obstruction checks for spanning multi-island bridges."""

from stencilizer.core.axis import Axis
from stencilizer.core.geometry import signed_area
from stencilizer.domain import Contour


def classify_obstruction_axis(
    contour: Contour,
    outer: Contour,
    axis: Axis,
    *,
    gap_lower: float | None = None,
    gap_upper: float | None = None,
    bridge_lower: float | None = None,
    bridge_upper: float | None = None,
    edge_margin: float = 20.0,
) -> str:
    """Classify a candidate obstruction, retaining the vertical gap rule."""
    contour_area = signed_area(contour.points)
    outer_area = signed_area(outer.points)
    same_winding = (contour_area < 0) == (outer_area < 0)
    if same_winding:
        bbox = contour.bounding_box()
        outer_bbox = outer.bounding_box()
        if (
            axis.is_x
            and bridge_lower is not None
            and bridge_upper is not None
            and gap_lower is not None
            and gap_upper is not None
        ):
            spans_bridge = axis.bbox_lo(bbox) < bridge_lower and axis.bbox_hi(bbox) > bridge_upper
            in_gap_region = axis.cross_lo(bbox) <= gap_upper and axis.cross_hi(bbox) >= gap_lower
            if spans_bridge and in_gap_region:
                return "structural"
        if (
            axis.bbox_lo(bbox) <= axis.bbox_lo(outer_bbox) + edge_margin
            and axis.bbox_hi(bbox) >= axis.bbox_hi(outer_bbox) - edge_margin
        ):
            return "structural"
    if not same_winding:
        return "island"
    return "unknown"


def has_spanning_obstruction_axis(
    outer: Contour,
    inners: list[Contour],
    all_contours: list[Contour],
    bridge_lower: float,
    bridge_upper: float,
    axis: Axis,
    *,
    edge_margin: float = 20.0,
) -> bool:
    """Check bridge-space contours between consecutive islands."""
    if len(inners) < 2:
        return False
    sorted_inners = sorted(inners, key=lambda c: axis.cross_lo(c.bounding_box()))
    for i in range(len(sorted_inners) - 1):
        lower_bbox = sorted_inners[i].bounding_box()
        upper_bbox = sorted_inners[i + 1].bounding_box()
        gap_lower = axis.cross_hi(lower_bbox)
        gap_upper = axis.cross_lo(upper_bbox)
        for contour in all_contours:
            if contour is outer or contour in inners:
                continue
            bbox = contour.bounding_box()
            if axis.bbox_lo(bbox) <= bridge_upper and axis.bbox_hi(bbox) >= bridge_lower:
                overlaps_gap = axis.cross_lo(bbox) <= gap_lower and axis.cross_hi(bbox) >= gap_upper
                inside_gap = axis.cross_lo(bbox) >= gap_lower and axis.cross_hi(bbox) <= gap_upper
                if overlaps_gap or inside_gap:
                    kind = classify_obstruction_axis(
                        contour,
                        outer,
                        axis,
                        gap_lower=gap_lower,
                        gap_upper=gap_upper,
                        bridge_lower=bridge_lower,
                        bridge_upper=bridge_upper,
                        edge_margin=edge_margin,
                    )
                    if kind != "structural":
                        return True
    return False
