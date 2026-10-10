"""Unit tests for bridge integrity in ``validate`` and the transform's use of its verdict."""

from collections.abc import Sequence

import pytest

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.domain.glyph import Glyph
from stencilizer.variable import transform
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.replay import BridgeLine, SurgeryMap, map_surgery, replay
from stencilizer.variable.validate import validate
from tests.font_helpers import UBUNTU
from tests.unit._variable_cases import ABOVE, BELOW, STEM, UPM, WGHT, Outline
from tests.unit._variable_cases import glyph_from_outlines as _glyph
from tests.unit._variable_cases import read_variable as _read


def _stem_lines() -> Sequence[BridgeLine]:
    smap = map_surgery(_glyph(STEM), _glyph(BELOW, ABOVE))
    assert smap is not None
    assert len(smap.lines) == 2
    return smap.lines


def _replayed(below: Outline, above: Outline) -> VariableGlyph:
    """The bridged stem with its wght master at the given pieces."""
    return VariableGlyph(_glyph(BELOW, ABOVE), (WGHT,), (_glyph(below, above),), ("wght",))


def test_intact_bridge_in_every_master_validates() -> None:
    grown = [[(x * 1.1, y * 1.1) for x, y in piece] for piece in (BELOW, ABOVE)]
    assert validate(_replayed(*grown), UPM, allowed_islands=0, lines=_stem_lines())


@pytest.mark.parametrize(
    ("below", "above"),
    [
        # The facing lines meet: BELOW's top rises to ABOVE's bottom.
        ([(0.0, 0.0), (0.0, 200.0), (100.0, 200.0), (100.0, 0.0)], ABOVE),
        # The facing lines swap order: BELOW's top rises past ABOVE's bottom.
        ([(0.0, 0.0), (0.0, 250.0), (100.0, 250.0), (100.0, 0.0)], ABOVE),
        # ABOVE collapses to zero area while both lines keep their places.
        (BELOW, [(0.0, 200.0), (0.0, 200.0), (100.0, 200.0), (100.0, 200.0), (100.0, 200.0)]),
        # ABOVE turns over (mirrored in x) while both lines keep their places.
        (BELOW, [(100.0 - x, y) for x, y in ABOVE]),
    ],
    ids=["lines-close", "lines-swap", "piece-collapses", "piece-turns-over"],
)
def test_broken_bridge_in_a_master_fails_validation(below: Outline, above: Outline) -> None:
    vg = _replayed(below, above)
    # The island count alone accepts the master; only the bridge-line check rejects it.
    assert validate(vg, UPM, allowed_islands=0)
    assert not validate(vg, UPM, allowed_islands=0, lines=_stem_lines())


def test_bridged_returns_none_when_replay_succeeds_but_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vg, upm = _read(UBUNTU, "o")
    merged = remove_overlaps_compatible(flatten_compatible(vg, upm))
    assert merged is not None
    bridge, geometry = BridgeConfig(), GeometryConfig()
    assert transform._bridged(vg, merged, bridge, geometry, upm) is not None

    replayed: list[Glyph | None] = []

    def spy(smap: SurgeryMap, source: Glyph, output: Glyph, master: Glyph) -> Glyph | None:
        replayed.append(replay(smap, source, output, master))
        return replayed[-1]

    monkeypatch.setattr(transform, "replay", spy)
    monkeypatch.setattr(transform, "validate", lambda *_args, **_kwargs: False)
    assert transform._bridged(vg, merged, bridge, geometry, upm) is None
    assert len(replayed) == len(merged.masters)
    assert all(glyph is not None for glyph in replayed)
