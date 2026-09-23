"""Golden checks for per-glyph and complete-font processing behavior."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from stencilizer.config.settings import StencilizerSettings
from stencilizer.core.processor import FontProcessor
from tests.regression._golden import (
    FIXTURES,
    GOLDEN_DIR,
    _record_font,
    assert_glyphs_close,
    load_font_glyphs,
    normalize_to_1000,
    run_process_glyph,
)

GENERATE_COMMAND = "uv run python -m tests.regression._golden"


def _load_golden(path: Path) -> Any:
    if not path.exists():
        pytest.skip(f"Missing golden file {path}; generate with: {GENERATE_COMMAND}")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _assert_recording_value(actual: object, expected: object, context: str) -> None:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not abs(actual - expected) <= 1e-6:
            raise AssertionError(f"{context}: actual={actual!r} expected={expected!r}")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{context}: length actual={len(actual)} expected={len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_recording_value(actual_item, expected_item, f"{context}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"{context}: actual={actual!r} expected={expected!r}")


@pytest.mark.parametrize("font_key", tuple(FIXTURES))
def test_island_glyphs_match_golden(font_key: str) -> None:
    golden = _load_golden(GOLDEN_DIR / f"{font_key}.json.gz")
    upm, native_glyphs = load_font_glyphs(font_key)
    expected_glyphs = golden["glyphs"]
    assert expected_glyphs, f"{font_key}: golden glyph set is empty"
    assert golden["source_upm"] == upm, f"{font_key}: source UPM changed"

    failures: list[str] = []
    for name, expected in sorted(expected_glyphs.items()):
        if name not in native_glyphs:
            failures.append(f"{name}: missing from source font")
            continue
        try:
            normalized = normalize_to_1000(native_glyphs[name], upm)
            actual = run_process_glyph(normalized, 1000, None)
            assert_glyphs_close(actual, expected, tol=1e-6, context=f"{font_key}/{name}")
        except AssertionError as error:
            failures.append(f"{name}: {error}")

    if failures:
        names = ", ".join(failure.split(":", 1)[0] for failure in failures[:20])
        pytest.fail(
            f"{font_key}: {len(failures)} glyphs differ: {names}\nFirst failure: {failures[0]}"
        )


def test_commitmono_full_pipeline_matches_golden(tmp_path: Path) -> None:
    expected: dict[str, Any] = _load_golden(GOLDEN_DIR / "commitmono_pipeline.json.gz")
    output_path = tmp_path / "out.otf"
    FontProcessor(StencilizerSettings()).process(FIXTURES["commitmono"], output_path)
    actual = _record_font(output_path)

    assert set(actual) == set(expected), (
        f"CommitMono glyph set differs: missing={sorted(set(expected) - set(actual))[:20]} "
        f"extra={sorted(set(actual) - set(expected))[:20]}"
    )
    failures: list[str] = []
    for name in sorted(expected):
        try:
            _assert_recording_value(actual[name], expected[name], name)
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    if failures:
        names = ", ".join(failure.split(":", 1)[0] for failure in failures[:20])
        pytest.fail(
            f"CommitMono pipeline: {len(failures)} glyphs differ: {names}\n"
            f"First failure: {failures[0]}"
        )
