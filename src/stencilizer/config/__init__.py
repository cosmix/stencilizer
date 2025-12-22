"""Configuration management for stencilizer.

This module provides configuration management using Pydantic models.
Configuration can be provided via CLI arguments or defaults.

Key classes:
- BridgeConfig: Bridge generation settings
- ProcessingConfig: Font processing settings
- LoggingConfig: Logging settings
- StencilizerSettings: Main application settings
"""

from stencilizer.config.settings import (
    BridgeConfig,
    BridgePosition,
    LoggingConfig,
    ProcessingConfig,
    StencilizerSettings,
    get_default_settings,
)

__all__ = [
    "BridgeConfig",
    "BridgePosition",
    "LoggingConfig",
    "ProcessingConfig",
    "StencilizerSettings",
    "get_default_settings",
]
