"""Font session state for the desktop GUI."""

import hashlib
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.config.settings import GeometryConfig
from stencilizer.core import FontProcessor, process_glyph
from stencilizer.core.processor import GlyphClassification, ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.exceptions import (
    FontLoadError,
    FontSaveError,
    GlyphNotFoundError,
    StencilizerError,
)
from stencilizer.io import FontReader
from stencilizer.utils import ProcessingStats


def source_digest(path: Path) -> str:
    """SHA-256 hex digest of the file's bytes (the pinned source revision)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(staged: Path, output_path: Path) -> None:
    """Write the staged font through a new, exclusively created sibling, then rename it in place."""
    temporary = output_path.with_name(f".{output_path.name}.{secrets.token_hex(8)}.tmp")
    data = staged.read_bytes()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_failure_reason(error: Exception) -> str:
    """User-facing reason for a failed save, without internal staging paths."""
    if isinstance(error, OSError) and error.strerror:
        return f"cannot write the output: {error.strerror}"
    return str(error)


def unsupported_reason(font: Any) -> str | None:
    """Why the core cannot stencilize this TTFont, or None when it can."""
    if "fvar" in font:
        return "variable fonts (fvar table) are not supported"
    if "CFF2" in font:
        return "CFF2 outlines are not supported"
    if "glyf" not in font and "CFF " not in font:
        return "no supported outline table (glyf or CFF)"
    return None


@dataclass(frozen=True)
class PreviewResult:
    """Outcome of stencilizing one glyph for preview."""

    glyph_name: str
    original: Glyph
    stenciled: Glyph | None
    bridges_added: int
    error: str | None
    duration_ms: float


@dataclass
class FontSession:
    """A loaded, classified font ready for preview and saving."""

    path: Path
    font_format: str
    units_per_em: int
    glyph_count: int
    ascender: int
    descender: int
    classification: GlyphClassification
    processor: FontProcessor
    source_sha256: str
    _glyph_index: dict[str, Glyph] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index selected glyphs by name for previews."""
        self._glyph_index = {glyph.name: glyph for glyph in self.island_glyphs}

    @classmethod
    def open(cls, path: Path, processor: FontProcessor) -> "FontSession":
        """Load a supported font and classify its island glyphs."""
        try:
            if path.exists() and not path.is_file():
                raise FontLoadError(str(path), "not a regular file")
            source_sha256 = source_digest(path)
            with FontReader(path) as reader:
                reason = unsupported_reason(reader.font)
                if reason is not None:
                    raise FontLoadError(str(path), reason)
                return cls(
                    path=path,
                    font_format=reader.format,
                    units_per_em=reader.units_per_em,
                    glyph_count=reader.glyph_count,
                    ascender=int(reader.font["hhea"].ascent),
                    descender=int(reader.font["hhea"].descent),
                    classification=processor.classify_glyphs(reader),
                    processor=processor,
                    source_sha256=source_sha256,
                )
        except FontLoadError:
            raise
        except Exception as error:
            raise FontLoadError(str(path), str(error)) from error

    @property
    def island_glyphs(self) -> list[Glyph]:
        """Return glyphs selected for stencilization."""
        return self.classification.glyphs_to_process

    def glyph(self, name: str) -> Glyph | None:
        """Return an island glyph by name, if it is selected."""
        return self._glyph_index.get(name)

    def preview(self, name: str, bridge: BridgeConfig, geometry: GeometryConfig) -> PreviewResult:
        """Stencilize one selected glyph synchronously for display."""
        glyph = self.glyph(name)
        if glyph is None:
            raise GlyphNotFoundError(name)
        result = process_glyph(
            glyph.to_dict(),
            bridge.model_dump(),
            self.units_per_em,
            geometry_dict=geometry.model_dump(),
        )
        if "error" in result:
            return PreviewResult(
                glyph_name=name,
                original=glyph,
                stenciled=None,
                bridges_added=0,
                error=str(result["error"]),
                duration_ms=float(result["duration_ms"]),
            )
        return PreviewResult(
            glyph_name=name,
            original=glyph,
            stenciled=Glyph.from_dict(result["glyph"]),
            bridges_added=int(result["bridges_added"]),
            error=None,
            duration_ms=float(result["duration_ms"]),
        )

    def _assert_source_unchanged(self, output_path: Path) -> None:
        """Raise a save error when the opened source file has changed."""
        reason = f"input font '{self.path}' changed on disk since it was opened; reopen it"
        try:
            digest = source_digest(self.path)
        except OSError as error:
            raise FontSaveError(str(output_path), reason) from error
        if digest != self.source_sha256:
            raise FontSaveError(str(output_path), reason)

    def save(
        self,
        output_path: Path,
        settings: StencilizerSettings,
        progress: ProgressCallback | None = None,
    ) -> ProcessingStats:
        """Stencilize the pinned source font in a private staging directory, then publish it."""
        try:
            self._assert_source_unchanged(output_path)
            if output_path.is_dir():
                raise FontSaveError(str(output_path), "output is a directory")
            if output_path.resolve() == self.path.resolve() or (
                output_path.exists() and output_path.samefile(self.path)
            ):
                raise FontSaveError(str(output_path), "output would overwrite the input font")
            if not output_path.parent.is_dir():
                raise FontSaveError(str(output_path), "output folder does not exist")
            self.processor.config = settings
            with tempfile.TemporaryDirectory(prefix="stencilizer-gui-") as staging:
                staged = Path(staging) / output_path.name
                stats = self.processor.process(
                    font_path=self.path,
                    output_path=staged,
                    max_workers=settings.processing.max_workers,
                    progress_callback=progress,
                    classification=self.classification,
                )
                self._assert_source_unchanged(output_path)
                _publish(staged, output_path)
            return stats
        except StencilizerError:
            raise
        except Exception as error:
            raise FontSaveError(str(output_path), _save_failure_reason(error)) from error
