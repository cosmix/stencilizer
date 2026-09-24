"""Integration coverage for explicit glyph bridge directions."""

from pathlib import Path

from stencilizer.config.settings import (
    BridgeConfig,
    BridgeDirection,
    LoggingConfig,
    StencilizerSettings,
)
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.domain import Glyph
from stencilizer.io import FontReader
from tests.integration.conftest import FIXTURES_DIR


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
