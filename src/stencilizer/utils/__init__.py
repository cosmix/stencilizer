"""Utility functions for stencilizer.

This module provides utility functions including:

- Logging setup and configuration
- Geometric calculations (area, point-in-polygon, intersections)
- Progress reporting helpers
"""

from stencilizer.utils.logging import (
    ProcessingLogger,
    ProcessingStats,
    configure_logging,
)

__all__ = [
    "ProcessingLogger",
    "ProcessingStats",
    "configure_logging",
]
