"""Qt-free variable session behavior: outcome cache, survey storage and threading."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.core import FontProcessor
from stencilizer.gui import variable_session
from stencilizer.gui.session import FontSession
from stencilizer.gui.variable_session import OutcomeKey, VariableOutcomeCache
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.reader import read_variable_glyph
from stencilizer.variable.transform import VariableOutcome, transform_variable_glyph

FIXTURES = Path(__file__).parent.parent / "fixtures" / "variable"
INTER = FIXTURES / "Inter-VF-subset.ttf"


def _outcome() -> VariableOutcome:
    glyph = read_variable_glyph(TTFont(INTER), "o")
    assert glyph is not None
    return VariableOutcome(glyph, 1, 0)


def _key(name: str, bridge: str = "b", geometry: str = "g") -> OutcomeKey:
    return (name, bridge, geometry, BridgeDirection.AUTO)


def test_cache_keeps_most_recently_used() -> None:
    cache = VariableOutcomeCache(max_entries=4)
    names = [f"g{i}" for i in range(6)]
    for name in names[:4]:
        cache.outcome(_key(name), _outcome)
    cache.outcome(_key("g0"), _outcome)  # refresh g0
    for name in names[4:]:
        cache.outcome(_key(name), _outcome)
    assert [key[0] for key in cache.outcome_keys] == ["g3", "g0", "g4", "g5"]


def test_cache_clears_on_other_parameters() -> None:
    cache = VariableOutcomeCache(max_entries=4)
    cache.outcome(_key("a"), _outcome)
    cache.bridge_count(_key("b"), _outcome)
    cache.outcome(_key("a", bridge="other"), _outcome)
    assert [key[0] for key in cache.outcome_keys] == ["a"]
    assert cache.counts == {}


def test_survey_stores_only_integers(processor: FontProcessor) -> None:
    session = FontSession.open(INTER, processor)
    assert session.variable is not None
    session.unbridged(BridgeConfig(), GeometryConfig())
    counts = session.variable._cache.counts
    assert counts
    assert all(type(count) is int for count in counts.values())
    assert session.variable._cache.outcome_keys == []


def _counting_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    calls: list[str] = []
    real = transform_variable_glyph

    def counted(vg: VariableGlyph, *args: object) -> VariableOutcome:
        calls.append(vg.name)
        return real(vg, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(variable_session, "transform_variable_glyph", counted)
    return calls


def test_geometry_change_misses_cache(
    processor: FontProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = FontSession.open(INTER, processor)
    calls = _counting_transform(monkeypatch)
    bridge = BridgeConfig()
    session.preview("o", bridge, GeometryConfig())
    session.preview("o", bridge, GeometryConfig(), location={"wght": 700})
    assert calls == ["o"]
    session.preview("o", bridge, GeometryConfig(point_dedup_tolerance=0.9))
    assert calls == ["o", "o"]


def test_preview_and_survey_in_parallel(processor: FontProcessor) -> None:
    session = FontSession.open(INTER, processor)
    bridge, geometry = BridgeConfig(), GeometryConfig()
    with ThreadPoolExecutor(max_workers=2) as pool:
        preview_future = pool.submit(lambda: session.preview("o", bridge, geometry).bridges_added)
        survey_future = pool.submit(session.unbridged, bridge, geometry)
        preview_count = preview_future.result()
        unbridged = survey_future.result()
    assert preview_count == session.preview("o", bridge, geometry).bridges_added
    assert ("o" in unbridged) == (preview_count == 0)


def test_cold_preview_time_per_island_glyph(processor: FontProcessor) -> None:
    session = FontSession.open(INTER, processor)
    worst = 0.0
    for glyph in session.island_glyphs:
        started = time.perf_counter()
        session.preview(glyph.name, BridgeConfig(), GeometryConfig())
        worst = max(worst, time.perf_counter() - started)
    print(f"worst cold preview: {worst * 1000:.1f} ms")
    assert worst < 5.0


def test_zero_bridge_preview_returns_unchanged_outline(
    processor: FontProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = FontSession.open(INTER, processor)
    monkeypatch.setattr(
        variable_session,
        "transform_variable_glyph",
        lambda vg, *_args: VariableOutcome(vg, 0, 0),
    )
    result = session.preview("o", BridgeConfig(), GeometryConfig(), location={"wght": 700})
    assert result.error is None
    assert result.bridges_added == 0
    assert result.stenciled is not None
    assert result.stenciled == result.original


def test_transform_failure_becomes_preview_error_and_is_not_cached(
    processor: FontProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = FontSession.open(INTER, processor)
    assert session.variable is not None

    def failing(_vg: VariableGlyph, *_args: object) -> VariableOutcome:
        raise ValueError("boom")

    monkeypatch.setattr(variable_session, "transform_variable_glyph", failing)
    result = session.preview("o", BridgeConfig(), GeometryConfig())
    assert result.stenciled is None
    assert result.bridges_added == 0
    assert result.error == "boom"
    assert session.variable._cache.outcome_keys == []
