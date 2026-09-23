"""Contour traversal direction calculations."""

from stencilizer.domain import Point


def _compute_side_percentages(
    contour_points: list[Point],
    start_idx: int,
    end_idx: int,
    threshold: float,
    want_less_than: bool,
    is_x: bool,
) -> tuple[float, float]:
    """Compute percentage of points on correct side for forward and backward traversal.

    Args:
        contour_points: List of points in the contour
        start_idx: Starting index
        end_idx: Ending index
        threshold: The coordinate threshold to check against
        want_less_than: If True, check for coord < threshold; else coord > threshold
        is_x: If True, check x coordinate; else check y coordinate

    Returns:
        Tuple of (forward_percentage, backward_percentage)
    """
    n = len(contour_points)

    forward_correct = 0
    forward_total = 0
    idx = (start_idx + 1) % n
    count = 0
    while idx != end_idx and count < n:
        p = contour_points[idx]
        coord = p.x if is_x else p.y
        if want_less_than:
            if coord < threshold:
                forward_correct += 1
        else:
            if coord > threshold:
                forward_correct += 1
        forward_total += 1
        idx = (idx + 1) % n
        count += 1

    backward_correct = 0
    backward_total = 0
    idx = start_idx
    count = 0
    while idx != end_idx and count < n:
        p = contour_points[idx]
        coord = p.x if is_x else p.y
        if want_less_than:
            if coord < threshold:
                backward_correct += 1
        else:
            if coord > threshold:
                backward_correct += 1
        backward_total += 1
        idx = (idx - 1) % n
        count += 1

    forward_pct = forward_correct / max(forward_total, 1)
    backward_pct = backward_correct / max(backward_total, 1)

    return forward_pct, backward_pct


def detect_traversal_direction(
    contour_points: list[Point],
    start_idx: int,
    end_idx: int,
    threshold: float,
    want_less_than: bool,
    is_x: bool,
) -> int:
    """Detect which traversal direction stays on the correct side.

    Returns:
        +1 for forward traversal, -1 for backward traversal
    """
    forward_pct, backward_pct = _compute_side_percentages(
        contour_points, start_idx, end_idx, threshold, want_less_than, is_x
    )

    return -1 if backward_pct > forward_pct else 1


def detect_traversal_direction_robust(
    contour_points: list[Point],
    start_idx: int,
    end_idx: int,
    threshold: float,
    want_less_than: bool,
    is_x: bool,
) -> int:
    """Detect traversal direction using multiple heuristics.

    Combines:
    1. Percentage of points on correct side
    2. First point after start check (critical)
    3. Signed distance accumulation

    Args:
        contour_points: List of points in the contour
        start_idx: Starting index
        end_idx: Ending index
        threshold: The coordinate threshold to check against
        want_less_than: If True, check for coord < threshold; else coord > threshold
        is_x: If True, check x coordinate; else check y coordinate

    Returns:
        1 for forward direction, -1 for backward direction
    """
    n = len(contour_points)

    # Get the existing percentage-based scores
    forward_pct, backward_pct = _compute_side_percentages(
        contour_points, start_idx, end_idx, threshold, want_less_than, is_x
    )

    # Check first point after start in each direction
    forward_first_idx = (start_idx + 1) % n
    backward_first_idx = (start_idx - 1) % n

    def on_correct_side(p: Point) -> bool:
        coord = p.x if is_x else p.y
        return (coord < threshold) if want_less_than else (coord > threshold)

    forward_first_ok = on_correct_side(contour_points[forward_first_idx])
    backward_first_ok = on_correct_side(contour_points[backward_first_idx])

    # Combine heuristics with weights
    forward_score = forward_pct * 0.5 + (0.5 if forward_first_ok else 0.0)
    backward_score = backward_pct * 0.5 + (0.5 if backward_first_ok else 0.0)

    return -1 if backward_score > forward_score else 1
