"""Configuration settings for Stencilizer."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BridgePosition(str, Enum):
    """Preferred bridge position."""

    AUTO = "auto"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_BOTTOM = "top_bottom"


class BridgeConfig(BaseModel):
    """Configuration for bridge generation."""

    width_percent: float = Field(
        default=235.0,
        ge=30.0,
        le=300.0,
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


class ProcessingConfig(BaseModel):
    """Configuration for font processing."""

    max_workers: Optional[int] = Field(
        default=None,
        description="Max worker processes (None = auto)",
    )
    skip_composite: bool = Field(
        default=True,
        description="Skip composite glyphs",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_file: Optional[Path] = Field(
        default=None,
        description="Path to log file",
    )
    log_level: str = Field(
        default="INFO",
        description="Console log level",
    )
    file_log_level: str = Field(
        default="DEBUG",
        description="File log level (more verbose)",
    )


class StencilizerSettings(BaseModel):
    """Main application settings."""

    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

def get_default_settings() -> StencilizerSettings:
    """Get default application settings."""
    return StencilizerSettings()
