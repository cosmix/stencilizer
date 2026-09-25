"""Tests for GUI font session saving edge cases not pinned to test_session.py."""

import shutil
from pathlib import Path

import pytest

from stencilizer.core import FontProcessor
from stencilizer.exceptions import FontSaveError
from stencilizer.gui.session import FontSession
from stencilizer.utils import ProcessingStats
from tests.gui.conftest import build_settings

pytestmark = pytest.mark.usefixtures("staging_root")


def _temporary_files(directory: Path) -> list[Path]:
    """Return save temporaries left behind in a directory."""
    return list(directory.glob(".*.tmp"))


def test_save_leaves_no_temporary_files(
    processor: FontProcessor, roboto_path: Path, tmp_path: Path
) -> None:
    """A successful save leaves only the published font, with ordinary file permissions."""
    session = FontSession.open(roboto_path, processor)
    output_dir = tmp_path / "saved"
    output_dir.mkdir()
    reference = tmp_path / "reference.bin"
    reference.write_bytes(b"")

    session.save(output_dir / "out.ttf", build_settings(processor))

    assert [path.name for path in output_dir.iterdir()] == ["out.ttf"]
    assert (output_dir / "out.ttf").stat().st_mode == reference.stat().st_mode


def test_save_reports_vanished_source(
    processor: FontProcessor, roboto_path: Path, tmp_path: Path
) -> None:
    """A source deleted since opening is a save error even when the output already exists."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)
    output_path = tmp_path / "out.ttf"
    output_path.write_bytes(b"previous save")
    source.unlink()

    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, build_settings(processor))
    assert output_path.read_bytes() == b"previous save"


def test_save_wraps_unexpected_errors(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-stencilizer failure while processing becomes a save error carrying its reason."""
    session = FontSession.open(roboto_path, processor)
    output_path = tmp_path / "out.ttf"

    def fail(**_kwargs: object) -> ProcessingStats:
        """Fail the way an exhausted disk would."""
        raise RuntimeError("disk full")

    monkeypatch.setattr(processor, "process", fail)
    with pytest.raises(FontSaveError, match="disk full"):
        session.save(output_path, build_settings(processor))

    assert not output_path.exists()
    assert _temporary_files(tmp_path) == []
