"""Logging utilities for Stencilizer."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import structlog


@dataclass
class ProcessingStats:
    """Statistics from processing run."""

    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    bridges_added: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def configure_logging(
    log_file: Path | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    quiet: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Configure dual-output structured logging.

    Args:
        log_file: Path to log file (auto-generated if None)
        console_level: Logging level for console output
        file_level: Logging level for file output
        quiet: If True, suppress console output except errors

    Returns:
        Configured structlog logger
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"stencilizer_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(logging.Formatter("%(message)s"))

    structlog.configure(
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
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("stencilizer")
    logger.info("Logging initialized", log_file=str(log_file), level=file_level)

    return logger


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
