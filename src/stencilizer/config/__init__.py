"""Configuration management for stencilizer.

This module provides configuration management using Pydantic Settings.
Configuration can be provided via:

- CLI arguments (highest priority)
- Environment variables (STENCILIZER_ prefix)
- Default values

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
