"""Tests for variable-font classification and processing."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.misc.textTools import Tag  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.tables._f_v_a_r import Axis, table__f_v_a_r  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor, GlyphAnalyzer
from stencilizer.exceptions import VariationDataError
from stencilizer.io import FontReader
from stencilizer.variable import processing
from stencilizer.variable.processing import classify_variable_glyphs, variable_island_counts
from stencilizer.variable.transform import transform_variable_glyph

FIXTURES = Path(__file__).parent.parent / "fixtures"
UBUNTU = FIXTURES / "variable" / "Ubuntu-VF-subset.ttf"
INTER = FIXTURES / "variable" / "Inter-VF-subset.ttf"
ROBOTO = FIXTURES / "Roboto-Regular.ttf"


def _processor(tmp_path: Path) -> FontProcessor:
    return FontProcessor(StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "log.txt")))


def _gvar_tuples(font: TTFont, name: str) -> list[Any]:
    return [(v.axes, list(v.coordinates)) for v in font["gvar"].variations.get(name, [])]


def _glyph_bytes(font: TTFont, name: str) -> bytes:
    data: bytes = font["glyf"][name].compile(font["glyf"])
    return data


def test_classify_lists_overlap_built_counter_and_skips_plain_glyph(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    with FontReader(INTER) as reader:
        classification = processor.classify_glyphs(reader)
        counts = dict(variable_island_counts(reader))
    names = [glyph.name for glyph in classification.glyphs_to_process]
    assert "P" in names
    assert classification.skipped_reasons["l"] == "no islands"
    assert counts["P"] == 1
    assert "l" not in counts


def test_variation_data_error_skips_one_glyph_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = processing.read_variable_glyph

    def failing_read(font: TTFont, name: str) -> Any:
        if name == "o":
            raise VariationDataError(name, "unreadable")
        return original(font, name)

    monkeypatch.setattr(processing, "read_variable_glyph", failing_read)
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        classification, variable = classify_variable_glyphs(processor, reader)
    assert classification.skipped_reasons["o"] == "unsupported variation data"
    assert classification.unsupported_islands["o"] == 1
    assert "o" not in variable
    assert variable
    assert [g.name for g in classification.glyphs_to_process] == list(variable)


def test_flatten_failure_keeps_glyph_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = processing.flatten_compatible

    def failing_flatten(vg: Any, upm: int) -> Any:
        if vg.name == "o":
            raise VariationDataError(vg.name, "cannot flatten")
        return original(vg, upm)

    monkeypatch.setattr(processing, "flatten_compatible", failing_flatten)
    with FontReader(UBUNTU) as reader:
        classification, variable = classify_variable_glyphs(_processor(tmp_path), reader)
    assert "o" in variable
    assert "o" in [glyph.name for glyph in classification.glyphs_to_process]


def test_process_writes_valid_variable_font_with_stats(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    out = tmp_path / "out.ttf"
    with FontReader(UBUNTU) as reader:
        classification, variable = classify_variable_glyphs(processor, reader)
        upm = reader.units_per_em
    config = processor.config
    expected_unbridged = sum(classification.unsupported_islands.values())
    for vg in variable.values():
        expected_unbridged += transform_variable_glyph(
            vg, config.bridge, config.geometry, upm
        ).unbridged_count
    stats = processor.process(UBUNTU, out, max_workers=2, classification=classification)
    output = TTFont(out)
    assert "fvar" in output
    assert "gvar" in output
    assert stats.bridges_added > 0
    assert stats.processed_count == len(classification.glyphs_to_process)
    assert stats.skipped_count == classification.skipped_count
    assert stats.error_count == 0
    assert stats.unbridged_count == expected_unbridged
    assert stats.duration_seconds > 0


def test_process_fvar_only_font_writes_no_gvar(tmp_path: Path) -> None:
    font = TTFont(ROBOTO)
    axis = Axis()
    axis.axisTag = Tag("wght")
    axis.minValue, axis.defaultValue, axis.maxValue = 100.0, 400.0, 900.0
    axis.axisNameID = 256
    fvar = table__f_v_a_r()
    fvar.axes = [axis]
    fvar.instances = []
    font["fvar"] = fvar
    source = tmp_path / "fvar-only.ttf"
    font.save(source)
    out = tmp_path / "out.ttf"
    stats = _processor(tmp_path).process(source, out, max_workers=2)
    output = TTFont(out)
    assert stats.error_count == 0
    assert stats.bridges_added > 0
    assert "gvar" not in output
    assert _glyph_bytes(output, "O") != _glyph_bytes(TTFont(source), "O")


def test_progress_callback_total_matches_selected(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    calls: list[tuple[int, int, str, bool]] = []
    with FontReader(UBUNTU) as reader:
        expected = len(processor.classify_glyphs(reader).glyphs_to_process)
    processor.process(
        UBUNTU,
        tmp_path / "out.ttf",
        max_workers=2,
        progress_callback=lambda *args: calls.append(args),
    )
    assert len(calls) == expected
    assert {total for _, total, _, _ in calls} == {expected}
    assert all(success for _, _, _, success in calls)


def test_given_classification_is_not_analyzed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        classification = processor.classify_glyphs(reader)
    calls: list[str] = []
    original = GlyphAnalyzer.analyze

    def counting(self: GlyphAnalyzer, glyph: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(glyph.name)
        return original(self, glyph, *args, **kwargs)

    monkeypatch.setattr(GlyphAnalyzer, "analyze", counting)
    processor.process(UBUNTU, tmp_path / "out.ttf", max_workers=2, classification=classification)
    assert calls == []


def test_glyph_without_bridges_is_not_rewritten(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        _, variable = classify_variable_glyphs(processor, reader)
        upm = reader.units_per_em
    config = processor.config
    unbridged = [
        name
        for name, vg in variable.items()
        if transform_variable_glyph(vg, config.bridge, config.geometry, upm).bridge_count == 0
    ]
    assert unbridged
    out = tmp_path / "out.ttf"
    processor.process(UBUNTU, out, max_workers=2)
    source, output = TTFont(UBUNTU), TTFont(out)
    for name in unbridged:
        assert _glyph_bytes(output, name) == _glyph_bytes(source, name)
        assert _gvar_tuples(output, name) == _gvar_tuples(source, name)
