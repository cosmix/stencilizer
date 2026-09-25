"""Integration coverage for explicit glyph bridge directions."""

from pathlib import Path

import pytest

from stencilizer.config.settings import (
    BridgeConfig,
    BridgeDirection,
    GeometryConfig,
    LoggingConfig,
    StencilizerSettings,
)
from stencilizer.core.merger import ContourMerger
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point, WindingDirection
from stencilizer.io import FontReader
from tests.integration.conftest import FIXTURES_DIR

MergeCall = tuple[Contour, Contour, bool, bool]


def _load_glyph(font_name: str, glyph_name: str) -> tuple[Glyph, int]:
    """Load one glyph and its font's units per em from a fixture font."""
    reader = FontReader(FIXTURES_DIR / font_name)
    reader.load()
    try:
        glyph = reader.get_glyph(glyph_name)
        assert glyph is not None, f"{glyph_name} is missing from {font_name}"
        return glyph, reader.units_per_em
    finally:
        reader.close()


def _process(glyph: Glyph, bridge: BridgeConfig, upm: int) -> Glyph:
    """Process a glyph through the public worker entry point."""
    result = process_glyph(glyph.to_dict(), bridge.model_dump(), upm)
    assert "error" not in result, result
    return Glyph.from_dict(result["glyph"])


def _contours(glyph: Glyph) -> list[dict[str, object]]:
    """Return a glyph's serialized contours for exact output comparison."""
    return [contour.to_dict() for contour in glyph.contours]


def _center(glyph: Glyph) -> tuple[float, float]:
    """Return the centre of a glyph's input bounding box."""
    bounds = [contour.bounding_box() for contour in glyph.contours]
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def _spans(value: float, lower: float, upper: float) -> bool:
    """Return whether a contour range includes the given coordinate."""
    return lower <= value <= upper


def test_explicit_direction_splits_o_along_axis() -> None:
    """Explicit and automatic directions cut Roboto O along their expected axes."""
    glyph, upm = _load_glyph("Roboto-Regular.ttf", "O")
    center_x, center_y = _center(glyph)

    horizontal = _process(glyph, BridgeConfig(direction=BridgeDirection.HORIZONTAL), upm)
    assert len(horizontal.contours) == 4
    assert all(
        not _spans(center_y, contour.bounding_box()[1], contour.bounding_box()[3])
        for contour in horizontal.contours
    )

    for direction in (BridgeDirection.VERTICAL, BridgeDirection.AUTO):
        result = _process(glyph, BridgeConfig(direction=direction), upm)
        assert len(result.contours) == 4
        assert all(
            not _spans(center_x, contour.bounding_box()[0], contour.bounding_box()[2])
            for contour in result.contours
        )


def test_stacked_islands_follow_direction() -> None:
    """Explicit directions choose the matching spanning or sequential B and eight surgery."""
    for name in ("B", "eight"):
        glyph, upm = _load_glyph("Roboto-Regular.ttf", name)
        automatic_spanning = _process(glyph, BridgeConfig(use_spanning_bridges=True), upm)
        automatic_sequential = _process(glyph, BridgeConfig(use_spanning_bridges=False), upm)
        horizontal = _process(glyph, BridgeConfig(direction=BridgeDirection.HORIZONTAL), upm)
        vertical = _process(
            glyph,
            BridgeConfig(direction=BridgeDirection.VERTICAL, use_spanning_bridges=False),
            upm,
        )

        assert _contours(horizontal) == _contours(automatic_sequential)
        assert _contours(vertical) == _contours(automatic_spanning)
        assert _contours(horizontal) != _contours(automatic_spanning)


def test_every_island_glyph_survives_explicit_directions(tmp_path: Path) -> None:
    """Every classified island glyph processes without an error in either explicit direction."""
    settings = StencilizerSettings(logging=LoggingConfig(log_file=tmp_path / "p.log"))
    processor = FontProcessor(settings)
    for font_name in ("Roboto-Regular.ttf", "Lato-Black.ttf"):
        reader = FontReader(FIXTURES_DIR / font_name)
        reader.load()
        try:
            classification = processor.classify_glyphs(reader)
            for direction in (BridgeDirection.VERTICAL, BridgeDirection.HORIZONTAL):
                bridge = BridgeConfig(direction=direction)
                for glyph in classification.glyphs_to_process:
                    result = process_glyph(
                        glyph.to_dict(), bridge.model_dump(), reader.units_per_em
                    )
                    assert "error" not in result, f"{font_name} {glyph.name}: {result}"
        finally:
            reader.close()


def _build_filled_encircled_digit() -> tuple[Glyph, int]:
    """Build the filled-circle digit-8 fixture with two inverted (nested-outer) bowls.

    Mirrors tests/unit/test_glyph_transformer.py::
    test_transform_filled_encircled_digit_with_inverted_islands, copied here rather than
    imported so this integration test does not depend on a frozen unit test module.
    """
    outer_circle = Contour(
        points=[Point(0.0, 0.0), Point(0.0, 200.0), Point(200.0, 200.0), Point(200.0, 0.0)],
        direction=WindingDirection.CLOCKWISE,
    )
    digit_cutout = Contour(
        points=[Point(50.0, 20.0), Point(150.0, 20.0), Point(150.0, 180.0), Point(50.0, 180.0)],
        direction=WindingDirection.COUNTER_CLOCKWISE,
    )
    top_bowl = Contour(
        points=[Point(70.0, 110.0), Point(70.0, 160.0), Point(130.0, 160.0), Point(130.0, 110.0)],
        direction=WindingDirection.CLOCKWISE,
    )
    bottom_bowl = Contour(
        points=[Point(70.0, 40.0), Point(70.0, 90.0), Point(130.0, 90.0), Point(130.0, 40.0)],
        direction=WindingDirection.CLOCKWISE,
    )
    glyph = Glyph(
        metadata=GlyphMetadata(
            name="eight.circle", unicode=0x2467, advance_width=200, left_side_bearing=0
        ),
        contours=[outer_circle, digit_cutout, top_bowl, bottom_bowl],
    )
    return glyph, 1000


def _capture_merge_calls(monkeypatch: pytest.MonkeyPatch) -> list[MergeCall]:
    """Record each call to the merger, then delegate to the real implementation."""
    calls: list[MergeCall] = []
    original = ContourMerger.merge_contours_with_bridges

    def spy(
        self: ContourMerger,
        inner: Contour,
        outer: Contour,
        bridge_width: float,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        all_contours: list[Contour] | None = None,
        processed_nested: list[Contour] | None = None,
        *,
        upm: int = 1000,
        geometry: GeometryConfig | None = None,
    ) -> list[Contour]:
        calls.append((inner, outer, force_horizontal, force_vertical))
        return original(
            self,
            inner,
            outer,
            bridge_width,
            force_horizontal=force_horizontal,
            force_vertical=force_vertical,
            all_contours=all_contours,
            processed_nested=processed_nested,
            upm=upm,
            geometry=geometry,
        )

    monkeypatch.setattr(ContourMerger, "merge_contours_with_bridges", spy)
    return calls


def test_inverted_islands_follow_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit directions reach the merge that resolves a filled encircled digit's islands.

    For this fixture the outer circle's single hole is handled by the ordinary island-group
    path (surgery_groups._sequential), and that one call's bridge construction already accounts
    for the two inverted bowls via its ``all_contours``/``processed_nested`` bookkeeping: tracing
    ``ctx.processed`` shows both bowl indices are marked processed by this call alone, before
    ``surgery_nested.process_nested``/``_process_inverted`` ever runs, so no separate nested-path
    call reaches the merger for this glyph's geometry. The single call is identified below by its
    ``outer`` argument matching the glyph's outer circle, and it is this call's force flags that
    govern whether the bowls end up bridged, routed around, or passed through unchanged.
    """
    glyph, upm = _build_filled_encircled_digit()
    outer_bbox = glyph.contours[0].bounding_box()

    horizontal_calls = _capture_merge_calls(monkeypatch)
    horizontal = _process(glyph, BridgeConfig(direction=BridgeDirection.HORIZONTAL), upm)
    horizontal_outer_calls = [c for c in horizontal_calls if c[1].bounding_box() == outer_bbox]
    assert horizontal_outer_calls
    assert all(fh and not fv for _, _, fh, fv in horizontal_outer_calls)

    vertical_calls = _capture_merge_calls(monkeypatch)
    vertical = _process(glyph, BridgeConfig(direction=BridgeDirection.VERTICAL), upm)
    vertical_outer_calls = [c for c in vertical_calls if c[1].bounding_box() == outer_bbox]
    assert vertical_outer_calls
    assert all(fv and not fh for _, _, fh, fv in vertical_outer_calls)

    assert _contours(horizontal) != _contours(vertical)
