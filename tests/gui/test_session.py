"""Tests for GUI font sessions without Qt widgets."""

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import (
    BridgeConfig,
    ProcessingConfig,
    StencilizerSettings,
)
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification
from stencilizer.domain import Glyph
from stencilizer.exceptions import FontLoadError, FontSaveError, GlyphNotFoundError
from stencilizer.gui.session import FontSession, unsupported_reason
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats


def _settings(processor: FontProcessor, bridge: BridgeConfig | None = None) -> StencilizerSettings:
    """Build serial-save settings while retaining the fixture logger."""
    return StencilizerSettings(
        bridge=bridge or BridgeConfig(),
        processing=ProcessingConfig(max_workers=1),
        logging=processor.config.logging,
    )


def _saved_glyph(path: Path, name: str) -> Glyph:
    """Load one glyph from a saved font."""
    with FontReader(path) as reader:
        glyph = reader.get_glyph(name)
    assert glyph is not None
    return glyph


def test_open_roboto(processor: FontProcessor, roboto_path: Path) -> None:
    """Opening Roboto exposes its measured metadata and island selection."""
    session = FontSession.open(roboto_path, processor)

    assert session.font_format == "TrueType"
    assert session.units_per_em == 2048
    assert session.ascender == 2146
    assert session.descender == -555
    assert len(session.island_glyphs) == 562
    assert session.glyph("O") is not None
    assert session.glyph("space") is None


def test_open_commit_mono_previews_first_glyph(
    processor: FontProcessor, commit_mono_path: Path
) -> None:
    """Opening the CFF fixture supports an in-process glyph preview."""
    session = FontSession.open(commit_mono_path, processor)

    assert session.font_format == "OpenType"
    assert len(session.island_glyphs) == 467
    result = session.preview(session.island_glyphs[0].name, BridgeConfig(), _geometry())
    assert result.stenciled is not None


def _geometry() -> GeometryConfig:
    """Return the default geometry configuration for previews."""
    return GeometryConfig()


def test_open_wraps_invalid_and_missing_files(processor: FontProcessor, tmp_path: Path) -> None:
    """Invalid and missing inputs become GUI load errors."""
    invalid_path = tmp_path / "invalid.ttf"
    invalid_path.write_bytes(b"not a font")

    with pytest.raises(FontLoadError):
        FontSession.open(invalid_path, processor)
    with pytest.raises(FontLoadError):
        FontSession.open(tmp_path / "missing.ttf", processor)


def test_open_rejects_unsupported_fonts(
    processor: FontProcessor,
    cff2_font_path: Path,
    variable_font_path: Path,
    roboto_path: Path,
    commit_mono_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported tables are rejected before glyph classification."""
    calls: list[FontReader] = []

    def record_classification(reader: FontReader) -> GlyphClassification:
        """Record attempts to classify an unsupported font."""
        calls.append(reader)
        return GlyphClassification()

    monkeypatch.setattr(processor, "classify_glyphs", record_classification)

    with pytest.raises(FontLoadError, match="CFF2 outlines are not supported"):
        FontSession.open(cff2_font_path, processor)
    with pytest.raises(FontLoadError, match="variable fonts \\(fvar table\\) are not supported"):
        FontSession.open(variable_font_path, processor)

    assert unsupported_reason(TTFont()) is not None
    assert "glyf or CFF" in (unsupported_reason(TTFont()) or "")
    assert unsupported_reason(TTFont(roboto_path)) is None
    assert unsupported_reason(TTFont(commit_mono_path)) is None
    assert calls == []


def test_open_pins_source_digest(processor: FontProcessor, roboto_path: Path) -> None:
    """Opening records the exact source revision used for previews and saves."""
    session = FontSession.open(roboto_path, processor)

    assert session.source_sha256 == hashlib.sha256(roboto_path.read_bytes()).hexdigest()


def test_preview_uses_parameters(processor: FontProcessor, roboto_path: Path) -> None:
    """Preview output changes with bridge width and spanning behavior."""
    session = FontSession.open(roboto_path, processor)
    default_o = session.preview("O", BridgeConfig(), _geometry())
    narrow_o = session.preview("O", BridgeConfig(width_percent=30.0), _geometry())
    wide_o = session.preview("O", BridgeConfig(width_percent=110.0), _geometry())
    spanning_b = session.preview("B", BridgeConfig(use_spanning_bridges=True), _geometry())
    split_b = session.preview("B", BridgeConfig(use_spanning_bridges=False), _geometry())

    assert default_o.bridges_added == 1
    assert default_o.error is None
    assert len(default_o.original.contours) == 2
    assert default_o.stenciled is not None
    assert len(default_o.stenciled.contours) == 4
    assert narrow_o.stenciled is not None
    assert wide_o.stenciled is not None
    assert spanning_b.stenciled is not None
    assert split_b.stenciled is not None
    assert narrow_o.stenciled.to_dict() != wide_o.stenciled.to_dict()
    assert spanning_b.stenciled.to_dict() != split_b.stenciled.to_dict()


def test_preview_rejects_unselected_glyph(processor: FontProcessor, roboto_path: Path) -> None:
    """A glyph outside the island selection cannot be previewed."""
    session = FontSession.open(roboto_path, processor)

    with pytest.raises(GlyphNotFoundError):
        session.preview("space", BridgeConfig(), _geometry())


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

    stats = session.save(output_path, _settings(processor))

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
        stats = session.save(tmp_path / filename, _settings(processor, bridge))
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
        session.save(output_path, _settings(processor))
    assert not output_path.exists()

    source.unlink()
    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, _settings(processor))
    assert not output_path.exists()

    shutil.copy(roboto_path, source)
    session = FontSession.open(source, processor)

    def process_and_change(**kwargs: object) -> ProcessingStats:
        """Write partial output then replace the source under processing."""
        output = kwargs["output_path"]
        assert isinstance(output, Path)
        output.write_bytes(b"partial")
        source.write_bytes(replacement)
        return ProcessingStats()

    monkeypatch.setattr(processor, "process", process_and_change)
    with pytest.raises(FontSaveError, match="changed on disk"):
        session.save(output_path, _settings(processor))
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
            session.save(output_path, _settings(processor))

    assert source.read_bytes() == original_bytes
