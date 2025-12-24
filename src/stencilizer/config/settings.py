"""Configuration settings for Stencilizer."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class BridgePosition(str, Enum):
    """Preferred bridge position."""

    AUTO = "auto"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_BOTTOM = "top_bottom"


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
    bezier_flatten_tolerance: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Tolerance for Bezier curve flattening (at reference UPM)",
    )
    min_contour_gap: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Minimum gap between contour points to consider distinct (at reference UPM)",
    )

    def scale_tolerance(self, base_value: float, upm: int) -> float:
        """Scale a tolerance value for the given UPM.

        Args:
            base_value: The tolerance value at reference UPM
            upm: The actual UPM of the font

        Returns:
            Scaled tolerance value
        """
        return base_value * (upm / self.reference_upm)

    def get_point_dedup_tolerance(self, upm: int) -> float:
        """Get point deduplication tolerance scaled for UPM."""
        return self.scale_tolerance(self.point_dedup_tolerance, upm)

    def get_line_epsilon(self, upm: int) -> float:
        """Get line intersection epsilon scaled for UPM."""
        return self.scale_tolerance(self.line_intersection_epsilon, upm)

    def get_bezier_tolerance(self, upm: int) -> float:
        """Get Bezier flattening tolerance scaled for UPM."""
        return self.scale_tolerance(self.bezier_flatten_tolerance, upm)

    def get_contour_gap(self, upm: int) -> float:
        """Get minimum contour gap scaled for UPM."""
        return self.scale_tolerance(self.min_contour_gap, upm)


class BridgeConfig(BaseModel):
    """Configuration for bridge generation."""

    width_percent: float = Field(
        default=60.0,
        ge=30.0,
        le=110.0,
        description="Bridge width as percentage of stroke width",
    )
    inset_percent: float = Field(
        default=2.0,
        ge=0.0,
        le=25.0,
        description="How far bridge endpoints are inset from contours (prevents extending outside)",
    )
    min_bridges: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Minimum bridges per island",
    )
    position_preference: BridgePosition = Field(
        default=BridgePosition.AUTO,
        description="Preferred bridge position",
    )
    sample_count: int = Field(
        default=36,
        ge=8,
        le=72,
        description="Number of candidate points to sample per island",
    )
    use_spanning_bridges: bool = Field(
        default=True,
        description="For vertically-stacked islands, use spanning vertical bridges instead of per-island horizontal bridges",
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
