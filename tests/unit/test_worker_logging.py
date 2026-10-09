"""Spawned pool workers keep the parent's logging configuration."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from stencilizer.config import StencilizerSettings
from stencilizer.config.settings import LoggingConfig
from stencilizer.core.processor import FontProcessor
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats, configure_logging
from stencilizer.utils.logging import (
    init_worker_logging,
    worker_logging_initargs,
    worker_pool_options,
)
from tests.font_helpers import ROBOTO

WORKER_LOGGER = "stencilizer.core.multi_island_merge"


@pytest.fixture
def owned_logger() -> Iterator[logging.Logger]:
    """The ``stencilizer`` logger with no handlers, restored after the test."""
    logger = logging.getLogger("stencilizer")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = []
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers, logger.level, logger.propagate = saved


def test_initargs_are_none_without_a_file_handler(owned_logger: logging.Logger) -> None:
    owned_logger.addHandler(logging.StreamHandler())

    assert worker_logging_initargs() is None
    assert worker_pool_options() == {}


@pytest.mark.usefixtures("owned_logger")
def test_initargs_describe_the_configured_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"

    configure_logging(log_file=log_file, console_level="WARNING", file_level="INFO")

    initargs = worker_logging_initargs()
    assert initargs == (str(log_file), logging.INFO, logging.WARNING)
    options = worker_pool_options()
    assert options["initializer"] is init_worker_logging
    assert options["initargs"] == initargs


def test_worker_initializer_rebuilds_handlers_and_appends(
    owned_logger: logging.Logger, tmp_path: Path
) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("parent line\n", encoding="utf-8")

    init_worker_logging(str(log_file), logging.DEBUG, logging.ERROR)

    assert worker_logging_initargs() == (str(log_file), logging.DEBUG, logging.ERROR)
    assert owned_logger.level == logging.DEBUG
    assert owned_logger.propagate is False
    logging.getLogger(WORKER_LOGGER).debug("worker record")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "parent line"
    assert "DEBUG" in lines[1]
    assert lines[1].endswith(f"| {WORKER_LOGGER} | worker record")


@pytest.mark.usefixtures("owned_logger")
def test_worker_initializer_tolerates_a_missing_log_directory(tmp_path: Path) -> None:
    init_worker_logging(str(tmp_path / "gone" / "run.log"), logging.DEBUG, logging.CRITICAL)

    assert worker_logging_initargs() is not None
    assert not (tmp_path / "gone").exists()


@pytest.mark.usefixtures("owned_logger")
def test_worker_debug_records_reach_the_parent_log_file(tmp_path: Path) -> None:
    """A real spawned pool writes its debug records to the parent's log file."""
    log_file = tmp_path / "run.log"
    settings = StencilizerSettings(
        logging=LoggingConfig(log_file=log_file, log_level="ERROR", file_log_level="DEBUG")
    )
    processor = FontProcessor(settings, quiet=True)
    reader = FontReader(ROBOTO)
    reader.load()
    try:
        glyph = next(glyph for glyph in reader.iter_glyphs() if glyph.name == ".notdef")
        stats = ProcessingStats()

        processor._process_glyphs_parallel(
            glyphs=[glyph], upm=reader.units_per_em, max_workers=1, stats=stats
        )
    finally:
        reader.close()

    assert stats.error_count == 0
    text = log_file.read_text(encoding="utf-8")
    assert f"| {WORKER_LOGGER} |" in text
    assert "Multi-island merge failed: no common X overlap" in text
