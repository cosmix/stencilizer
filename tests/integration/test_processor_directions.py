"""Integration tests for processor bridge counts and per-glyph directions."""

import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from stencilizer.config import BridgeConfig, LoggingConfig, StencilizerSettings
from stencilizer.config.settings import BridgeDirection
from stencilizer.core.processor import FontProcessor, _islands_bridged, process_glyph
from stencilizer.domain import Glyph
from stencilizer.io import FontReader
from tests.integration.conftest import FIXTURES_DIR


@pytest.fixture(autouse=True)
def spawn_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start processor workers with the same spawn context as the application."""
    monkeypatch.setattr(
        "stencilizer.core.processor.ProcessPoolExecutor",
        functools.partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    )


def _processor(tmp_path: Path) -> FontProcessor:
    """Create a processor whose logs remain in the test directory."""
    settings = StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "p.log"))
    return FontProcessor(settings)


def _glyph(reader: FontReader, name: str) -> Glyph:
    """Return a required fixture glyph."""
    glyph = reader.get_glyph(name)
    assert glyph is not None
    return glyph


def _has_spanning_contour(glyph: Glyph, axis: int, centre: float) -> bool:
    """Return whether a contour spans a centre coordinate on one axis."""
    return any(
        contour.bounding_box()[axis] < centre < contour.bounding_box()[axis + 2]
        for contour in glyph.contours
    )


def test_duplicate_islands_are_counted_once_each() -> None:
    """Match each island against one ``after`` entry instead of set membership."""
    island_a = {"points": [{"x": 0.0, "y": 0.0}], "direction": "CW"}
    island_b = {"points": [{"x": 1.0, "y": 1.0}], "direction": "CCW"}

    assert _islands_bridged([island_a, island_a], [island_a]) == 1
    assert _islands_bridged([island_a, island_b], [island_a, island_b]) == 0
    assert _islands_bridged([island_a, island_b], []) == 2


def test_unbridgeable_glyph_reports_zero_bridges() -> None:
    """Report bridges only for islands that the transformer changed."""
    reader = FontReader(FIXTURES_DIR / "Roboto-Regular.ttf")
    reader.load()
    try:
        expected_counts = {"four": 0, "AE": 0, "O": 1, "B": 2, "eight": 2}
        for name, expected_count in expected_counts.items():
            glyph = _glyph(reader, name)
            result = process_glyph(
                glyph.to_dict(), BridgeConfig().model_dump(), reader.units_per_em
            )
            assert "error" not in result
            assert result["bridges_added"] == expected_count
            if expected_count == 0:
                assert Glyph.from_dict(result["glyph"]).contours == glyph.contours
    finally:
        reader.close()


def test_process_applies_per_glyph_directions(tmp_path: Path) -> None:
    """Apply an override only to its named glyph when saving a font."""
    input_path = FIXTURES_DIR / "Roboto-Regular.ttf"
    directed_path = tmp_path / "directed.ttf"
    plain_path = tmp_path / "plain.ttf"
    processor = _processor(tmp_path)

    directed_stats = processor.process(
        input_path,
        directed_path,
        max_workers=1,
        directions={"O": BridgeDirection.HORIZONTAL},
    )
    plain_stats = processor.process(input_path, plain_path, max_workers=1)

    assert directed_stats.error_count == 0
    assert plain_stats.error_count == 0

    directed_reader = FontReader(directed_path)
    plain_reader = FontReader(plain_path)
    directed_reader.load()
    plain_reader.load()
    try:
        directed_o = _glyph(directed_reader, "O")
        plain_o = _glyph(plain_reader, "O")
        directed_d = _glyph(directed_reader, "D")
        plain_d = _glyph(plain_reader, "D")

        assert not _has_spanning_contour(directed_o, 1, 728)
        assert not _has_spanning_contour(plain_o, 0, 703.5)
        assert directed_d.to_dict() == plain_d.to_dict()
    finally:
        directed_reader.close()
        plain_reader.close()
