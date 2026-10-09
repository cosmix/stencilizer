"""Configuration settings for Stencilizer."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class GeometryConfig(BaseModel):
    """Configuration for geometry operations with scale-relative tolerances.

    All tolerance values are specified at a reference UPM of 1000 and will be
    scaled proportionally for fonts with different UPM values.
    """

    reference_upm: int = Field(
        default=1000,
        description="Reference UPM for tolerance values",
    )
    point_dedup_tolerance: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Tolerance for point deduplication (at reference UPM)",
    )
    line_intersection_epsilon: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Epsilon for line intersection calculations (at reference UPM)",
    )
    min_contour_gap: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Minimum gap between contour points to consider distinct (at reference UPM)",
    )
    min_stroke: float = Field(default=20.0, ge=0)
    stroke_search_floor: float = Field(default=400.0, ge=0)
    bridge_length_floor: float = Field(default=400.0, ge=0)
    bottom_bridge_offset: float = Field(default=10.0, ge=0)
    nested_outer_gap: float = Field(default=10.0, ge=0)
    island_edge_margin: float = Field(default=20.0, ge=0)
    bar_min_gap: float = Field(default=100.0, ge=0)
    asymmetry_floor: float = Field(default=1.0, ge=0)

    def scale_tolerance(self, base_value: float, upm: int) -> float:
        """Scale a tolerance value for the given UPM.

        Args:
            base_value: The tolerance value at reference UPM
            upm: The actual UPM of the font

        Returns:
            Scaled tolerance value
        """
        return base_value * (upm / self.reference_upm)

    def scaled(self, field_name: str, upm: int) -> float:
        """Scale a named font-unit field for the given UPM."""
        return self.scale_tolerance(getattr(self, field_name), upm)

    def get_point_dedup_tolerance(self, upm: int) -> float:
        """Get point deduplication tolerance scaled for UPM."""
        return self.scale_tolerance(self.point_dedup_tolerance, upm)

    def get_line_epsilon(self, upm: int) -> float:
        """Get line intersection epsilon scaled for UPM."""
        return self.scale_tolerance(self.line_intersection_epsilon, upm)

    def get_contour_gap(self, upm: int) -> float:
        """Get minimum contour gap scaled for UPM."""
        return self.scale_tolerance(self.min_contour_gap, upm)


class BridgeDirection(StrEnum):
    """Which way a glyph's bridges cut its strokes."""

    AUTO = "auto"  # the analyzer's choice, as before this change
    VERTICAL = "vertical"  # bridge line at a fixed x: an O loses its top and bottom strokes
    HORIZONTAL = "horizontal"  # bridge line at a fixed y: an O loses its left and right strokes


class BridgeWidthScaling(StrEnum):
    """How a variable font's bridge gaps change from master to master."""

    FIXED = "fixed"  # the default master's gap in every master
    PROPORTIONAL = "proportional"  # each gap follows the thickness of the stroke it cuts


class BridgeConfig(BaseModel):
    """Configuration for bridge generation."""

    width_percent: float = Field(
        default=60.0,
        ge=30.0,
        le=110.0,
        description="Bridge width as a percentage of a reference stroke of 10% of the font UPM",
    )
    use_spanning_bridges: bool = Field(
        default=True,
        description="For vertically-stacked islands, use spanning vertical bridges instead of per-island horizontal bridges",
    )
    direction: BridgeDirection = Field(
        default=BridgeDirection.AUTO,
        description="Bridge direction for every island of the glyph (auto keeps the analyzer's choice)",
    )
    width_scaling: BridgeWidthScaling = Field(
        default=BridgeWidthScaling.FIXED,
        description="How a variable font's bridge gaps change across masters (fixed: the default master's gap everywhere; proportional: each gap follows the stroke it cuts)",
    )
    scaling_strength: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Proportional mode: how strongly gaps follow stroke thickness, 0 (fixed) to 100 (fully proportional)",
    )
    min_width_percent: float = Field(
        default=30.0,
        ge=10.0,
        le=110.0,
        description="Proportional mode: the smallest gap, as a percentage of the reference stroke, never above the default master's gap",
    )


class ProcessingConfig(BaseModel):
    """Configuration for font processing."""

    max_workers: int | None = Field(
        default=None,
        description="Max worker processes (None = auto)",
    )
    skip_composite: bool = Field(
        default=True,
        description="Skip composite glyphs",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_file: Path | None = Field(
        default=None,
        description="Path to log file",
    )
    log_level: str = Field(
        default="WARNING",
        description="Console log level",
    )
    file_log_level: str = Field(
        default="DEBUG",
        description="File log level (more verbose)",
    )


class StencilizerSettings(BaseModel):
    """Main application settings."""

    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def get_default_settings() -> StencilizerSettings:
    """Get default application settings."""
    return StencilizerSettings()
