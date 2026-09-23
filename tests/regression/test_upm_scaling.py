"""Processing a glyph at another UPM should only scale its output coordinates."""

import gzip
import json
from functools import lru_cache
from typing import cast

import pytest

from stencilizer.domain.contour import Contour, Point, PointType, WindingDirection
from stencilizer.domain.glyph import Glyph, GlyphMetadata
from tests.regression._golden import (
    GOLDEN_DIR,
    assert_glyphs_close,
    load_font_glyphs,
    normalize_to_1000,
    run_process_glyph,
    scale_glyph,
    serialize_glyph,
)


@lru_cache(maxsize=3)
def _golden_glyph_names(font_key: str) -> tuple[str, ...]:
    with gzip.open(GOLDEN_DIR / f"{font_key}.json.gz", "rt", encoding="utf-8") as stream:
        payload = cast("dict[str, object]", json.load(stream))
    glyphs = cast("dict[str, object]", payload["glyphs"])
    return tuple(glyphs)


@lru_cache(maxsize=3)
def _normalized_glyphs(font_key: str) -> dict[str, Glyph]:
    native_upm, native_glyphs = load_font_glyphs(font_key)
    wanted = set(_golden_glyph_names(font_key))
    return {
        name: normalize_to_1000(glyph, native_upm)
        for name, glyph in native_glyphs.items()
        if name in wanted
    }


@lru_cache(maxsize=6)
def _baseline_glyphs(
    font_key: str, reference_stroke_width: float | None
) -> tuple[dict[str, Glyph], dict[str, str]]:
    outputs: dict[str, Glyph] = {}
    errors: dict[str, str] = {}
    for name, glyph in _normalized_glyphs(font_key).items():
        try:
            outputs[name] = run_process_glyph(glyph, 1000, reference_stroke_width)
        except AssertionError as exc:
            errors[name] = _first_message(exc)
    return outputs, errors


def _first_message(exc: AssertionError) -> str:
    return next((line.strip() for line in str(exc).splitlines() if line.strip()), "AssertionError")


@pytest.mark.parametrize("font_key", ["roboto", "lato", "commitmono"])
@pytest.mark.parametrize("factor", [2.0, 0.5])
@pytest.mark.parametrize("reference_stroke_width", [None, 80.0])
def test_font_glyph_processing_scales_with_upm(
    font_key: str, factor: float, reference_stroke_width: float | None
) -> None:
    golden_path = GOLDEN_DIR / f"{font_key}.json.gz"
    if not golden_path.is_file():
        pytest.skip(f"golden file missing: {golden_path}")

    names = _golden_glyph_names(font_key)
    if not names:
        pytest.fail(f"{font_key}: golden glyph set is empty")
    normalized = _normalized_glyphs(font_key)
    baselines, baseline_errors = _baseline_glyphs(font_key, reference_stroke_width)
    failures: list[str] = []
    scaled_stroke_width = (
        None if reference_stroke_width is None else reference_stroke_width * factor
    )

    for name in names:
        if name not in normalized:
            failures.append(f"{name}: absent from source font")
            continue
        if name in baseline_errors:
            failures.append(f"{name}: baseline processing failed: {baseline_errors[name]}")
            continue
        try:
            scaled = run_process_glyph(
                scale_glyph(normalized[name], factor),
                int(1000 * factor),
                scaled_stroke_width,
            )
            assert_glyphs_close(
                scaled,
                serialize_glyph(baselines[name]),
                scale=factor,
                tol=1e-6,
                context=name,
            )
        except AssertionError as exc:
            failures.append(f"{name}: {_first_message(exc)}")

    if failures:
        details = "\n".join(failures[:20])
        pytest.fail(
            f"{len(failures)} of {len(names)} glyphs differ for "
            f"{font_key}, factor={factor}, stroke={reference_stroke_width}:\n{details}",
            pytrace=False,
        )


def _synthetic_o(*, reverse_winding: bool = False) -> Glyph:
    outer = Contour(
        points=[
            Point(0, 0, PointType.ON_CURVE),
            Point(600, 0, PointType.ON_CURVE),
            Point(600, 600, PointType.ON_CURVE),
            Point(0, 600, PointType.ON_CURVE),
        ],
        direction=WindingDirection.COUNTER_CLOCKWISE,
    )
    inner = Contour(
        points=[
            Point(100, 100, PointType.ON_CURVE),
            Point(100, 500, PointType.ON_CURVE),
            Point(500, 500, PointType.ON_CURVE),
            Point(500, 100, PointType.ON_CURVE),
        ],
        direction=WindingDirection.CLOCKWISE,
    )
    if reverse_winding:
        outer.points.reverse()
        outer.direction = WindingDirection.CLOCKWISE
        inner.points.reverse()
        inner.direction = WindingDirection.COUNTER_CLOCKWISE
    return Glyph(
        metadata=GlyphMetadata(
            name="synthetic_O", unicode=79, advance_width=600, left_side_bearing=0
        ),
        contours=[outer, inner],
    )


@lru_cache(maxsize=2)
def _synthetic_baseline(reverse_winding: bool) -> Glyph:
    return run_process_glyph(_synthetic_o(reverse_winding=reverse_winding), 1000, 80.0)


@pytest.mark.parametrize("factor", [2.0, 0.5, 4.0])
def test_synthetic_o_processing_scales_with_upm(factor: float) -> None:
    for reverse_winding in (False, True):
        glyph = _synthetic_o(reverse_winding=reverse_winding)
        baseline = _synthetic_baseline(reverse_winding)
        if reverse_winding and serialize_glyph(baseline) == serialize_glyph(glyph):
            pytest.fail("synthetic_O: reversed winding did not exercise bridge processing")

        scaled = run_process_glyph(scale_glyph(glyph, factor), int(1000 * factor), 80.0 * factor)
        assert_glyphs_close(
            scaled,
            serialize_glyph(baseline),
            scale=factor,
            tol=1e-6,
            context=f"synthetic_O factor={factor} reversed={reverse_winding}",
        )
