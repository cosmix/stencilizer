"""Tests for variable-font classification and processing."""

from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor, GlyphAnalyzer
from stencilizer.exceptions import VariationDataError
from stencilizer.io import FontReader
from stencilizer.variable import processing
from stencilizer.variable.processing import classify_variable_glyphs, variable_island_counts
from stencilizer.variable.transform import transform_variable_glyph
from tests.font_helpers import INTER, UBUNTU, fvar_only_roboto


def _processor(tmp_path: Path) -> FontProcessor:
    return FontProcessor(StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "log.txt")))


def _gvar_tuples(font: TTFont, name: str) -> list[Any]:
    return [(v.axes, list(v.coordinates)) for v in font["gvar"].variations.get(name, [])]


def _glyph_bytes(font: TTFont, name: str) -> bytes:
    data: bytes = font["glyf"][name].compile(font["glyf"])
    return data


def _failing_read(failing: str) -> Any:
    """Wrap ``read_variable_glyph`` so reading the glyph ``failing`` raises VariationDataError."""
    original = processing.read_variable_glyph

    def failing_read(font: TTFont, name: str, *args: Any) -> Any:
        if name == failing:
            raise VariationDataError(name, "unreadable")
        return original(font, name, *args)

    return failing_read


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
    monkeypatch.setattr(processing, "read_variable_glyph", _failing_read("o"))
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
    source = tmp_path / "fvar-only.ttf"
    fvar_only_roboto().save(source)
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


def test_unsupported_skip_log_carries_the_error_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(processing, "read_variable_glyph", _failing_read("o"))
    processor = _processor(tmp_path)
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        processor.processing_logger,
        "log_glyph_skipped",
        lambda name, reason: logged.append((name, reason)),
    )
    with FontReader(UBUNTU) as reader:
        classification, _ = classify_variable_glyphs(processor, reader)
    assert ("o", "unsupported variation data: unreadable") in logged
    assert classification.skipped_reasons["o"] == "unsupported variation data"


def test_font_wide_unicode_map_is_built_once_per_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = processing.read_variable_glyph
    maps: list[Any] = []

    def recording_read(font: TTFont, name: str, unicode_by_name: Any = None) -> Any:
        maps.append(unicode_by_name)
        return original(font, name, unicode_by_name)

    monkeypatch.setattr(processing, "read_variable_glyph", recording_read)
    processor = _processor(tmp_path)
    cmap_reads: list[str] = []
    with FontReader(UBUNTU) as reader:
        best_cmap = reader.font.getBestCmap

        def counting_cmap() -> Any:
            cmap_reads.append("getBestCmap")
            return best_cmap()

        monkeypatch.setattr(reader.font, "getBestCmap", counting_cmap)
        classification, _ = classify_variable_glyphs(processor, reader)
        glyph_count = len(reader.font.getGlyphOrder())
        classified = len(maps)
        processing._selected_glyphs(processor, reader, classification)
        variable_island_counts(reader)
        font_wide = reader.unicode_by_name
    assert classified == glyph_count
    assert len(maps) > classified
    assert font_wide
    assert all(unicode_map is font_wide for unicode_map in maps)
    assert cmap_reads == ["getBestCmap"]


def test_selected_glyph_that_fails_to_read_is_skipped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        classification = processor.classify_glyphs(reader)
        monkeypatch.setattr(processing, "read_variable_glyph", _failing_read("o"))
        selected, variable = processing._selected_glyphs(processor, reader, classification)
    assert "o" not in variable
    assert variable
    assert selected.skipped_reasons["o"] == "unsupported variation data"
    assert selected.unsupported_islands["o"] == 1
    assert selected.skipped_count == classification.skipped_count + 1
    assert "o" not in [glyph.name for glyph in selected.glyphs_to_process]
    assert "o" in [glyph.name for glyph in classification.glyphs_to_process]
    assert "o" not in classification.skipped_reasons


def test_selected_glyph_that_fails_to_read_is_left_unchanged_in_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path)
    with FontReader(UBUNTU) as reader:
        classification, variable = classify_variable_glyphs(processor, reader)
        upm = reader.units_per_em
    config = processor.config
    expected_unbridged = sum(classification.unsupported_islands.values()) + 1
    for name, vg in variable.items():
        if name != "o":
            expected_unbridged += transform_variable_glyph(
                vg, config.bridge, config.geometry, upm
            ).unbridged_count
    monkeypatch.setattr(processing, "read_variable_glyph", _failing_read("o"))
    out = tmp_path / "out.ttf"
    stats = processor.process(UBUNTU, out, max_workers=2, classification=classification)
    assert stats.error_count == 0
    assert stats.skipped_count == classification.skipped_count + 1
    assert stats.processed_count == len(variable) - 1
    assert stats.unbridged_count == expected_unbridged
    source, output = TTFont(UBUNTU), TTFont(out)
    assert _glyph_bytes(output, "o") == _glyph_bytes(source, "o")
    assert _gvar_tuples(output, "o") == _gvar_tuples(source, "o")
