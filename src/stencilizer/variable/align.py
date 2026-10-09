"""Put every point of a bridge line on that line in the default surgery output.

Core surgery keeps an input vertex in place of a cut point near it, so a cut edge can
lean off its bridge line. Two cut edges that must coincide then leave a hairline of ink
that closes the counter, and rounding widens it when the vertex and the line round to
different integers.
"""

from stencilizer.config import GeometryConfig
from stencilizer.domain.glyph import Glyph
from stencilizer.variable.model import glyph_coordinates
from stencilizer.variable.replay import SurgeryMap, replay

# Surgery output sits on its input edges to within replay's 1e-4 mapping distance.
_MAPPING_SLACK = 1e-3


def snap_distance(geometry: GeometryConfig, upm: int) -> float:
    """How far off a bridge line core surgery may keep an input vertex in place of a cut.

    Surgery skips a cut point within its contour gap of the line, and drops one within its
    point dedup tolerance of the vertex before it.
    """
    return max(geometry.get_contour_gap(upm), geometry.get_point_dedup_tolerance(upm))


def align_to_lines(
    smap: SurgeryMap, input_default: Glyph, output_default: Glyph, *, snap: float
) -> Glyph | None:
    """The default output replayed through ``smap``, with every line member on its line.

    The default is replayed like any master, so every member of a line gets the same
    coordinate on the line axis there and in every master: the members round alike and
    share their deltas. None when the replay fails or moves a point more than ``snap`` on
    either axis, as when a member's own edges miss its line and a distant edge is used.
    """
    glyph = replay(smap, input_default, output_default, input_default)
    if glyph is None:
        return None
    limit = snap + _MAPPING_SLACK
    moved = zip(glyph_coordinates(glyph), glyph_coordinates(output_default), strict=True)
    if any(abs(gx - ox) > limit or abs(gy - oy) > limit for (gx, gy), (ox, oy) in moved):
        return None
    return glyph
