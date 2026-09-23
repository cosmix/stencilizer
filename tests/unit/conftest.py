"""Shared processor test fixtures."""

import pytest

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.domain import Contour, Glyph, GlyphMetadata, Point, WindingDirection


@pytest.fixture
def sample_glyph_with_island() -> Glyph:
    """Create a sample glyph with one island."""
    # Outer contour (CW - TrueType convention)
    outer = Contour(
        points=[
            Point(0, 0),
            Point(0, 100),
            Point(100, 100),
            Point(100, 0),
        ],
        direction=WindingDirection.CLOCKWISE,
    )

    # Inner contour/island (CCW - TrueType convention)
    inner = Contour(
        points=[
            Point(25, 25),
            Point(75, 25),
            Point(75, 75),
            Point(25, 75),
        ],
        direction=WindingDirection.COUNTER_CLOCKWISE,
    )

    metadata = GlyphMetadata(
        name="O",
        unicode=ord("O"),
        advance_width=100,
        left_side_bearing=0,
    )

    return Glyph(metadata=metadata, contours=[outer, inner])


@pytest.fixture
def sample_glyph_no_island() -> Glyph:
    """Create a sample glyph without islands."""
    # Outer contour (CW - TrueType convention)
    outer = Contour(
        points=[
            Point(0, 0),
            Point(0, 100),
            Point(50, 100),
            Point(50, 0),
        ],
        direction=WindingDirection.CLOCKWISE,
    )

    metadata = GlyphMetadata(
        name="I",
        unicode=ord("I"),
        advance_width=50,
        left_side_bearing=0,
    )

    return Glyph(metadata=metadata, contours=[outer])


@pytest.fixture
def bridge_config() -> BridgeConfig:
    """Create test bridge configuration."""
    return BridgeConfig(width_percent=60.0)


@pytest.fixture
def settings(bridge_config: BridgeConfig) -> StencilizerSettings:
    """Create test stencilizer settings."""
    settings = StencilizerSettings()
    settings.bridge = bridge_config
    return settings
