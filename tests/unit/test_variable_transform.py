"""Unit tests for the unchanged-glyph fallback of the variable transform pipeline.

A glyph whose surgery cannot be mapped, replayed on every master, or validated is
written unchanged with its default-master islands counted; it is never half-replayed.
"""

from collections.abc import Callable

import pytest

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable import transform
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph, glyph_coordinates
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.replay import SurgeryMap, replay
from stencilizer.variable.solver import Coords
from tests.font_helpers import UBUNTU, island_count
from tests.unit._variable_cases import read_variable as _read


def _coordinates(vg: VariableGlyph) -> list[Coords]:
    return [glyph_coordinates(glyph) for glyph in (vg.default, *vg.masters)]


def _merged_islands(vg: VariableGlyph, upm: int) -> int:
    merged = remove_overlaps_compatible(flatten_compatible(vg, upm))
    assert merged is not None
    return island_count(merged.default, upm)


def _assert_unchanged(
    outcome: transform.VariableOutcome, vg: VariableGlyph, before: list[Coords], islands: int
) -> None:
    assert outcome.glyph is vg
    assert _coordinates(outcome.glyph) == before
    assert (outcome.bridge_count, outcome.unbridged_count) == (0, islands)


def _returning(value: object, calls: list[str], name: str) -> Callable[..., object]:
    def stub(*_args: object, **_kwargs: object) -> object:
        calls.append(name)
        return value

    return stub


def test_unpatched_pipeline_bridges_the_fixture_glyph() -> None:
    vg, upm = _read(UBUNTU, "o")
    assert _merged_islands(vg, upm) == 1
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert outcome.bridge_count >= 1
    assert outcome.glyph is not vg


@pytest.mark.parametrize(
    ("name", "value"), [("map_surgery", None), ("replay", None), ("validate", False)]
)
def test_failed_step_returns_input_with_merged_islands_counted(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    vg, upm = _read(UBUNTU, "o")
    islands = _merged_islands(vg, upm)
    before = _coordinates(vg)
    calls: list[str] = []
    monkeypatch.setattr(transform, name, _returning(value, calls, name))
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert calls
    _assert_unchanged(outcome, vg, before, islands)


def test_replay_failing_in_the_last_master_writes_no_replayed_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vg, upm = _read(UBUNTU, "o")
    before = _coordinates(vg)
    results: list[Glyph | None] = []

    def fail_last(
        smap: SurgeryMap, input_default: Glyph, output_default: Glyph, master: Glyph
    ) -> Glyph | None:
        last = len(results) == len(vg.masters) - 1
        results.append(None if last else replay(smap, input_default, output_default, master))
        return results[-1]

    monkeypatch.setattr(transform, "replay", fail_last)
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    assert len(results) == len(vg.masters)
    assert all(glyph is not None for glyph in results[:-1])
    _assert_unchanged(outcome, vg, before, _merged_islands(vg, upm))


@pytest.mark.parametrize("name", ["map_surgery", "round_variable_glyph", "validate"])
def test_variation_error_after_overlap_removal_returns_input(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    vg, upm = _read(UBUNTU, "o")
    islands = _merged_islands(vg, upm)
    before = _coordinates(vg)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise VariationDataError(vg.name, "no solution")

    monkeypatch.setattr(transform, name, fail)
    outcome = transform.transform_variable_glyph(vg, BridgeConfig(), GeometryConfig(), upm)
    _assert_unchanged(outcome, vg, before, islands)


def test_worker_reports_no_bridges_when_replay_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    vg, upm = _read(UBUNTU, "o")
    calls: list[str] = []
    monkeypatch.setattr(transform, "replay", _returning(None, calls, "replay"))
    config = BridgeConfig().model_dump()
    result = transform.process_variable_glyph(vg.to_dict(), config, upm, None)
    assert calls
    assert "error" not in result
    assert result["bridges_added"] == 0
    assert result["unbridged_count"] == 1
    assert result["glyph"] == vg.to_dict()
