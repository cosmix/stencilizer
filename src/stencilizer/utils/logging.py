"""Logging utilities for Stencilizer."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import structlog


@dataclass
class ProcessingStats:
    """Statistics from processing run."""

    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    bridges_added: int = 0
    unbridged_count: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    glyph_timings_ms: list[float] = field(default_factory=list)
    cancelled_count: int = 0
    was_cancelled: bool = False

    @property
    def duration_seconds(self) -> float:
        """Calculate processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def min_glyph_time_ms(self) -> float | None:
        """Minimum per-glyph processing time."""
        return min(self.glyph_timings_ms) if self.glyph_timings_ms else None

    @property
    def max_glyph_time_ms(self) -> float | None:
        """Maximum per-glyph processing time."""
        return max(self.glyph_timings_ms) if self.glyph_timings_ms else None

    @property
    def avg_glyph_time_ms(self) -> float | None:
        """Average per-glyph processing time."""
        if not self.glyph_timings_ms:
            return None
        return sum(self.glyph_timings_ms) / len(self.glyph_timings_ms)


LOGGER_NAME = "stencilizer"

# Picklable worker logging settings: log file path, file level, console level.
WorkerLogArgs = tuple[str, int, int]


def _attach_handlers(
    log_file: Path, file_level: int, console_level: int, *, delay: bool = False
) -> logging.Logger:
    """Replace the handlers on the ``stencilizer`` logger with a file and a console handler."""
    file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=delay)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    owned_logger = logging.getLogger(LOGGER_NAME)
    owned_logger.setLevel(logging.DEBUG)
    owned_logger.propagate = False
    for handler in owned_logger.handlers[:]:
        owned_logger.removeHandler(handler)
        handler.close()
    owned_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    owned_logger.addHandler(console_handler)
    return owned_logger


def worker_logging_initargs() -> WorkerLogArgs | None:
    """Describe the ``stencilizer`` logger's handlers so a spawned worker can rebuild them.

    Returns None when no file handler is attached (logging is not configured).
    """
    handlers = logging.getLogger(LOGGER_NAME).handlers
    file_handler = next((h for h in handlers if isinstance(h, logging.FileHandler)), None)
    if file_handler is None:
        return None
    console_levels = [
        h.level
        for h in handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    console_level = console_levels[0] if console_levels else logging.CRITICAL + 1
    return (file_handler.baseFilename, file_handler.level, console_level)


def init_worker_logging(log_file: str, file_level: int, console_level: int) -> None:
    """Process pool initializer: attach the parent's logging handlers inside a worker.

    A spawned worker starts with no handlers, so its debug records would be lost. The
    file opens lazily so a log path that has since disappeared cannot break the pool.
    """
    _attach_handlers(Path(log_file), file_level, console_level, delay=True)


def worker_pool_options() -> dict[str, Any]:
    """Return ``ProcessPoolExecutor`` keyword arguments that carry logging into workers."""
    initargs = worker_logging_initargs()
    if initargs is None:
        return {}
    return {"initializer": init_worker_logging, "initargs": initargs}


def configure_logging(
    log_file: Path | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    quiet: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Configure the stencilizer logger with file and console handlers."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"stencilizer_{timestamp}.log")

    owned_logger = _attach_handlers(
        log_file,
        getattr(logging, file_level.upper()),
        logging.ERROR if quiet else getattr(logging, console_level.upper()),
    )

    logger = structlog.wrap_logger(
        owned_logger,
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
    )
    logger.info("Logging initialized", log_file=str(log_file), level=file_level)

    return cast("structlog.stdlib.BoundLogger", logger)


class ProcessingLogger:
    """Logger for tracking processing progress and statistics."""

    def __init__(self, logger: structlog.stdlib.BoundLogger) -> None:
        self._logger = logger
        self._stats = ProcessingStats()

    def log_glyph_start(self, glyph_name: str) -> None:
        """Log start of glyph processing."""
        self._logger.debug("Processing glyph", glyph=glyph_name)

    def log_glyph_complete(
        self,
        glyph_name: str,
        bridges_added: int,
        duration_ms: float,
    ) -> None:
        """Log successful glyph processing."""
        self._logger.info(
            "Glyph processed",
            glyph=glyph_name,
            bridges=bridges_added,
            duration_ms=round(duration_ms, 2),
        )
        self._stats.processed_count += 1
        self._stats.bridges_added += bridges_added

    def log_glyph_skipped(self, glyph_name: str, reason: str) -> None:
        """Log skipped glyph."""
        self._logger.debug("Glyph skipped", glyph=glyph_name, reason=reason)
        self._stats.skipped_count += 1

    def log_glyph_error(
        self,
        glyph_name: str,
        error: Exception,
        traceback: str | None = None,
    ) -> None:
        """Log glyph processing error."""
        self._logger.error(
            "Glyph processing failed",
            glyph=glyph_name,
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback,
        )
        self._stats.error_count += 1
        self._stats.errors.append((glyph_name, str(error)))

    def log_bridge_placement(
        self,
        glyph_name: str,
        bridge_idx: int,
        position: str,
        score: float,
    ) -> None:
        """Log bridge placement details."""
        self._logger.debug(
            "Bridge placed",
            glyph=glyph_name,
            bridge_idx=bridge_idx,
            position=position,
            score=round(score, 2),
        )

    def log_contour_analysis(
        self,
        glyph_name: str,
        total_contours: int,
        outer_count: int,
        inner_count: int,
    ) -> None:
        """Log contour analysis results."""
        self._logger.debug(
            "Contour analysis",
            glyph=glyph_name,
            total=total_contours,
            outer=outer_count,
            inner=inner_count,
        )

    @property
    def stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self._stats
