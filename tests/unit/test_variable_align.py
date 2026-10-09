"""Bridge-line members share their line coordinate in the stored default and every master.

Core surgery may keep an input vertex up to its contour gap off a bridge line in place of
a cut point. Left there, the counter's cut edge leans off the outer piece's cut edge and a
hairline of ink closes the counter, touching the outside at one point at most.
"""

from collections import Counter
from pathlib import Path

import pytest

from stencilizer.config import GeometryConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.variable.align import align_to_lines, snap_distance
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.replay import (
    BridgeLine,
    EdgePoint,
    LineMember,
    SurgeryMap,
    Vertex,
    map_surgery,
    replay,
    slot_values,
)
from stencilizer.variable.rounding import round_variable_glyph
from tests.font_helpers import CANTARELL, INTER
from tests.unit._variable_cases import glyph_from_outlines, read_variable


def test_snap_distance_scales_with_the_font() -> None:
    geometry = GeometryConfig()
    assert snap_distance(geometry, 1000) == pytest.approx(1.0)
    assert snap_distance(geometry, 2048) == pytest.approx(2.048)


# Inter's o, B and eight keep a vertex 0.6 to 0.8 units off a line at 2048 UPM; Cantarell's
# six keeps one 0.18 units off.
@pytest.mark.parametrize(
    ("path", "char"), [(INTER, "o"), (INTER, "B"), (INTER, "8"), (CANTARELL, "6")]
)
def test_line_members_share_their_coordinate_once_rounded(path: Path, char: str) -> None:
    vg, upm = read_variable(path, char)
    merged = remove_overlaps_compatible(flatten_compatible(vg, upm))
    assert merged is not None
    outcome = GlyphTransformer(GlyphAnalyzer()).transform_with_outcome(merged.default, upm=upm)
    assert outcome.bridge_count >= 1
    snap = snap_distance(GeometryConfig(), upm)
    smap = map_surgery(merged.default, outcome.glyph, snap)
    assert smap is not None
    default = align_to_lines(smap, merged.default, outcome.glyph, snap=snap)
    assert default is not None
    replayed = [replay(smap, merged.default, default, master) for master in merged.masters]
    masters = [glyph for glyph in replayed if glyph is not None]
    assert len(masters) == len(replayed)
    rounded = round_variable_glyph(
        VariableGlyph(default, merged.supports, tuple(masters), vg.axis_tags, vg.cff2)
    )
    for line in smap.lines:
        # A cut edge meets its line at both ends.
        assert min(Counter(member.slot[0] for member in line.members).values()) >= 2
        slots = [member.slot for member in line.members]
        for glyph in (rounded.default, *rounded.masters):
            assert len(set(slot_values(glyph, slots, line.axis))) == 1


def test_default_moved_beyond_the_snap_distance_gives_no_alignment() -> None:
    # The vertex (100, 100) is a member of the line y = 50; its own edge from (100, 100)
    # to (100, 0) crosses the line 50 units below it, so aligning moves it that far.
    source = glyph_from_outlines([(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)])
    output = glyph_from_outlines(
        [(0.0, 0.0), (0.0, 50.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)]
    )
    row = (Vertex(0), EdgePoint(0, 1, 0.5), Vertex(1), Vertex(2), Vertex(3))
    line = BridgeLine(
        1,
        50.0,
        (LineMember((0, 1), (0, 0), cut=True), LineMember((0, 3), (1, 2), cut=False)),
    )
    smap = SurgeryMap((row,), (line,))
    assert align_to_lines(smap, source, output, snap=1.0) is None
    aligned = align_to_lines(smap, source, output, snap=60.0)
    assert aligned is not None
    assert (aligned.contours[0].points[3].x, aligned.contours[0].points[3].y) == (100.0, 50.0)
