"""Worker pool setup: workers always start with spawn, never fork.

The CLI progress bar and the GUI run threads in the parent process, and forking a
multi-threaded process can deadlock the children.
"""

import multiprocessing
from collections.abc import Mapping
from typing import Any

from stencilizer.config.settings import BridgeDirection
from stencilizer.utils.logging import worker_pool_options


def pool_options() -> dict[str, Any]:
    """Return the keyword arguments every ``ProcessPoolExecutor`` of the processor is built with."""
    return {"mp_context": multiprocessing.get_context("spawn"), **worker_pool_options()}


def config_for_glyph(
    config_dict: dict[str, Any],
    directions: Mapping[str, BridgeDirection] | None,
    name: str,
) -> dict[str, Any]:
    """Return the shared config or a glyph-specific direction override."""
    if directions is None or name not in directions:
        return config_dict
    return {**config_dict, "direction": directions[name]}
