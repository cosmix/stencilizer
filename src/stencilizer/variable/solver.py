"""Solve per-support deltas from masters sampled at each support's peak."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from stencilizer.exceptions import VariationDataError

if TYPE_CHECKING:
    from stencilizer.variable.model import Support

Coords = list[tuple[float, float]]

_PIVOT_EPSILON = 1e-12


def solve_deltas(
    supports: "Sequence[Support]",
    default: Sequence[tuple[float, float]],
    masters: Sequence[Sequence[tuple[float, float]]],
    *,
    glyph_name: str = "<unknown>",
) -> list[Coords]:
    """Solve ``M . D = masters - default`` where ``M[j][k] = supports[k].scalar(peak_j)``.

    Raises VariationDataError when the supports are linearly dependent (for example two
    supports sharing a peak, which OpenType allows).
    """
    count = len(supports)
    if count == 0:
        return []
    peaks = [support.peak() for support in supports]
    matrix = [[supports[k].scalar(peaks[j]) for k in range(count)] for j in range(count)]
    # One right-hand-side row per master: x0, y0, x1, y1, ... of (master - default).
    rhs = [
        [v for (mx, my), (dx, dy) in zip(master, default, strict=True) for v in (mx - dx, my - dy)]
        for master in masters
    ]
    _eliminate(matrix, rhs, glyph_name)
    return [[(row[i], row[i + 1]) for i in range(0, len(row), 2)] for row in rhs]


def _eliminate(matrix: list[list[float]], rhs: list[list[float]], glyph_name: str) -> None:
    """Gauss-Jordan elimination with partial pivoting, in place; ``rhs`` becomes the solution."""
    size = len(matrix)
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) <= _PIVOT_EPSILON:
            raise VariationDataError(glyph_name, "singular variation supports")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        divisor = matrix[col][col]
        matrix[col] = [v / divisor for v in matrix[col]]
        rhs[col] = [v / divisor for v in rhs[col]]
        for row in range(size):
            factor = matrix[row][col]
            if row == col or factor == 0.0:
                continue
            matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col], strict=True)]
            rhs[row] = [a - factor * b for a, b in zip(rhs[row], rhs[col], strict=True)]
