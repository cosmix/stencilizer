"""Tests for GUI font sessions without Qt widgets."""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification
from stencilizer.domain import Glyph
from stencilizer.exceptions import FontLoadError, FontSaveError
from stencilizer.gui.session import FontSession, source_digest, unsupported_reason
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats
from tests.font_helpers import CANTARELL
from tests.gui.conftest import build_settings

pytestmark = pytest.mark.usefixtures("staging_root")


def _saved_glyph(path: Path, name: str) -> Glyph:
    """Load one glyph from a saved font."""
    with FontReader(path) as reader:
        glyph = reader.get_glyph(name)
    assert glyph is not None
    return glyph


def _temporary_files(directory: Path) -> list[Path]:
    """Return save temporaries left behind in a directory."""
    return list(directory.glob(".*.tmp"))


def _write_then_replace_source(source: Path, replacement: bytes) -> Callable[..., ProcessingStats]:
    """Build a fake process that writes partial output, then replaces the source."""

    def process_and_change(**kwargs: object) -> ProcessingStats:
        """Write partial output then replace the source under processing."""
        output = kwargs["output_path"]
        assert isinstance(output, Path)
        output.write_bytes(b"partial")
        source.write_bytes(replacement)
        return ProcessingStats()

    return process_and_change


def _geometry() -> GeometryConfig:
    """Return the default geometry configuration for previews."""
    return GeometryConfig()


def test_open_rejects_unsupported_fonts(
    processor: FontProcessor,
    roboto_path: Path,
    commit_mono_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fonts without a supported outline table are rejected before classification."""
    calls: list[FontReader] = []

    def record_classification(reader: FontReader) -> GlyphClassification:
        """Record attempts to classify an unsupported font."""
        calls.append(reader)
        return GlyphClassification()

    monkeypatch.setattr(processor, "classify_glyphs", record_classification)

    outline_less = tmp_path / "outline-less.ttf"
    font = TTFont(roboto_path)
    del font["glyf"]
    del font["loca"]
    font.save(outline_less)
    with pytest.raises(FontLoadError, match="no supported outline table"):
        FontSession.open(outline_less, processor)

    assert unsupported_reason(TTFont()) is not None
    assert "glyf, CFF or CFF2" in (unsupported_reason(TTFont()) or "")
    assert unsupported_reason(TTFont(roboto_path)) is None
    assert unsupported_reason(TTFont(commit_mono_path)) is None
    assert calls == []


def test_open_variable_font_lists_axes(processor: FontProcessor, variable_font_path: Path) -> None:
    """A variable font opens with its axes."""
    session = FontSession.open(variable_font_path, processor)
    assert [(axis.tag, axis.name) for axis in session.axes] == [("wght", "wght")]


@pytest.mark.parametrize("cff2_font_path", [CANTARELL], ids=["variable"])
def test_open_accepts_cff2(processor: FontProcessor, cff2_font_path: Path) -> None:
    """A variable CFF2 font opens and lists its island glyphs.

    The static CFF2 case is covered by test_cff2_session_contracts.py.
    """
    session = FontSession.open(cff2_font_path, processor)

    assert "O" in {glyph.name for glyph in session.island_glyphs}
    assert unsupported_reason(TTFont(cff2_font_path)) is None


@pytest.mark.parametrize(
    ("input_path", "output_name", "island_count"),
    [("roboto", "out.ttf", 562), ("commit_mono", "out.otf", 467)],
)
def test_save_writes_stenciled_outlines(
    processor: FontProcessor,
    roboto_path: Path,
    commit_mono_path: Path,
    tmp_path: Path,
    outlines_match: Callable[[Glyph, Glyph], bool],
    input_path: str,
    output_name: str,
    island_count: int,
) -> None:
    """Saving writes all selected outlines and preserves preview geometry."""
    source = roboto_path if input_path == "roboto" else commit_mono_path
    session = FontSession.open(source, processor)
    output_path = tmp_path / output_name

    stats = session.save(output_path, build_settings(processor))

    assert stats.processed_count == island_count
    assert stats.error_count == 0
    family_name = TTFont(output_path)["name"].getDebugName(1)
    assert family_name is not None
    assert family_name.endswith(" Stenciled")
    saved_o = _saved_glyph(output_path, "O")
    preview = session.preview("O", BridgeConfig(), _geometry())
    assert len(saved_o.contours) == 4
    assert preview.stenciled is not None
    assert outlines_match(saved_o, preview.stenciled)


def test_save_uses_given_settings(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    outlines_match: Callable[[Glyph, Glyph], bool],
) -> None:
    """Each save applies its own bridge settings to its outlines."""
    session = FontSession.open(roboto_path, processor)
    narrow = BridgeConfig(width_percent=30.0)
    wide = BridgeConfig(width_percent=110.0)
    spanning = BridgeConfig(use_spanning_bridges=True)
    split = BridgeConfig(use_spanning_bridges=False)
    outputs = [("a.ttf", narrow), ("b.ttf", wide), ("c.ttf", spanning), ("d.ttf", split)]

    for filename, bridge in outputs:
        stats = session.save(tmp_path / filename, build_settings(processor, bridge))
        assert stats.error_count == 0
        assert stats.processed_count == 562

    saved_a = _saved_glyph(tmp_path / "a.ttf", "O")
    saved_b = _saved_glyph(tmp_path / "b.ttf", "O")
    saved_c = _saved_glyph(tmp_path / "c.ttf", "B")
    saved_d = _saved_glyph(tmp_path / "d.ttf", "B")
    narrow_preview = session.preview("O", narrow, _geometry())
    wide_preview = session.preview("O", wide, _geometry())
    spanning_preview = session.preview("B", spanning, _geometry())
    split_preview = session.preview("B", split, _geometry())

    assert saved_a.to_dict() != saved_b.to_dict()
    assert saved_c.to_dict() != saved_d.to_dict()
    assert narrow_preview.stenciled is not None
    assert wide_preview.stenciled is not None
    assert spanning_preview.stenciled is not None
    assert split_preview.stenciled is not None
    assert outlines_match(saved_a, narrow_preview.stenciled)
    assert outlines_match(saved_b, wide_preview.stenciled)
    assert outlines_match(saved_c, spanning_preview.stenciled)
    assert outlines_match(saved_d, split_preview.stenciled)


def test_save_refuses_changed_source(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving refuses a source changed before or during processing."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)
    replacement = Path("tests/fixtures/Lato-Black.ttf").read_bytes()
    source.write_bytes(replacement)
    output_path = tmp_path / "out.ttf"

    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, build_settings(processor))
    assert not output_path.exists()

    source.unlink()
    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, build_settings(processor))
    assert not output_path.exists()

    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)

    monkeypatch.setattr(processor, "process", _write_then_replace_source(source, replacement))
    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, build_settings(processor))
    assert not output_path.exists()


def test_save_refuses_input_path(
    processor: FontProcessor, roboto_path: Path, tmp_path: Path
) -> None:
    """Saving never overwrites the source through aliases or normalized paths."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)
    symlink = tmp_path / "source-link.ttf"
    hardlink = tmp_path / "source-hardlink.ttf"
    symlink.symlink_to(source)
    os.link(source, hardlink)
    original_bytes = source.read_bytes()

    for output_path in (source, symlink, hardlink, tmp_path / "sub" / ".." / source.name):
        with pytest.raises(FontSaveError, match="overwrite the input"):
            session.save(output_path, build_settings(processor))

    assert source.read_bytes() == original_bytes


def test_save_does_not_follow_output_swapped_to_input(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output path swapped for a link to the input during processing never writes the input."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)
    output_path = tmp_path / "out.ttf"
    real_process = processor.process

    def link_then_process(**kwargs: Any) -> ProcessingStats:
        """Plant a link from the output path to the input, then process for real."""
        output_path.symlink_to(source)
        return real_process(**kwargs)

    monkeypatch.setattr(processor, "process", link_then_process)
    stats = session.save(output_path, build_settings(processor))

    assert stats.error_count == 0
    assert source_digest(source) == session.source_sha256
    assert not output_path.is_symlink()
    assert len(_saved_glyph(output_path, "O").contours) == 4
    assert _temporary_files(tmp_path) == []


def test_failed_save_keeps_existing_output(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save that fails after writing leaves a file already at the output path untouched."""
    source = tmp_path / "source.ttf"
    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)
    output_path = tmp_path / "out.ttf"
    output_path.write_bytes(b"previous save")
    replacement = roboto_path.with_name("Lato-Black.ttf").read_bytes()

    monkeypatch.setattr(processor, "process", _write_then_replace_source(source, replacement))
    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, build_settings(processor))

    assert output_path.read_bytes() == b"previous save"
    assert _temporary_files(tmp_path) == []


def test_save_stages_outside_the_output_directory(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    staging_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The font is written in a private directory; the output directory sees only the result."""
    session = FontSession.open(roboto_path, processor)
    output_dir = tmp_path / "saved"
    output_dir.mkdir()
    real_process = processor.process
    staged: list[tuple[Path, int, list[str]]] = []

    def process_and_record(**kwargs: Any) -> ProcessingStats:
        """Process for real, then record the staging location and the output directory."""
        stats = real_process(**kwargs)
        path = kwargs["output_path"]
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        staged.append((path, mode, [entry.name for entry in output_dir.iterdir()]))
        return stats

    monkeypatch.setattr(processor, "process", process_and_record)
    session.save(output_dir / "out.ttf", build_settings(processor))

    [(path, mode, listing)] = staged
    assert not path.resolve().is_relative_to(output_dir.resolve())
    assert listing == []
    if os.name == "posix":
        assert mode == 0o700
    assert [entry.name for entry in output_dir.iterdir()] == ["out.ttf"]
    assert list(staging_root.iterdir()) == []


def test_save_refuses_directory_output(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output path naming an existing directory is refused before any processing."""
    session = FontSession.open(roboto_path, processor)
    output_path = tmp_path / "out.ttf"
    output_path.mkdir()
    calls: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> ProcessingStats:
        """Record an attempt to process."""
        calls.append(kwargs)
        return ProcessingStats()

    monkeypatch.setattr(processor, "process", record)
    with pytest.raises(FontSaveError, match="output is a directory"):
        session.save(output_path, build_settings(processor))

    assert calls == []
    assert output_path.is_dir()


def test_save_refuses_missing_output_folder(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output path whose parent folder is missing is refused before any processing."""
    session = FontSession.open(roboto_path, processor)
    output_path = tmp_path / "missing" / "out.ttf"
    calls: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> ProcessingStats:
        """Record an attempt to process."""
        calls.append(kwargs)
        return ProcessingStats()

    monkeypatch.setattr(processor, "process", record)
    with pytest.raises(FontSaveError, match="output folder does not exist"):
        session.save(output_path, build_settings(processor))

    assert calls == []
    assert not (tmp_path / "missing").exists()


def test_save_error_hides_internal_paths(
    processor: FontProcessor,
    roboto_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure names its cause without exposing staging or temporary paths."""
    session = FontSession.open(roboto_path, processor)
    staged: list[Path] = []

    def deny(**kwargs: Any) -> ProcessingStats:
        """Fail the way an unwritable staged file would, naming that file."""
        staged.append(kwargs["output_path"])
        raise PermissionError(13, "Permission denied", str(kwargs["output_path"]))

    monkeypatch.setattr(processor, "process", deny)
    with pytest.raises(FontSaveError, match="Permission denied") as caught:
        session.save(tmp_path / "out.ttf", build_settings(processor))

    message = str(caught.value)
    assert ".tmp" not in message
    assert staged[0].parent.name not in message
    assert not (tmp_path / "out.ttf").exists()
