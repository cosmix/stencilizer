"""Font session state for the desktop GUI."""

import hashlib
import os
import secrets
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.config.settings import BridgeDirection, GeometryConfig
from stencilizer.core import FontProcessor, process_glyph
from stencilizer.core.processor import GlyphClassification, ProgressCallback
from stencilizer.domain import Glyph
from stencilizer.exceptions import (
    FontLoadError,
    FontSaveError,
    GlyphNotFoundError,
    StencilizerError,
)
from stencilizer.gui.composites import (
    CompositeGlyph,
    compose,
    find_bridged_composites,
    load_component_outlines,
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
    composites: tuple[CompositeGlyph, ...]
    component_outlines: dict[str, Glyph]
    display_names: tuple[str, ...]
    _glyph_index: dict[str, Glyph] = field(init=False, repr=False)
    _composite_index: dict[str, CompositeGlyph] = field(init=False, repr=False)
    _composed: dict[str, Glyph] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index display glyphs and compose composite outlines once."""
        self._glyph_index = {glyph.name: glyph for glyph in self.island_glyphs}
        self._composite_index = {composite.name: composite for composite in self.composites}
        self._composed = {
            composite.name: compose(composite, self.component_outlines)
            for composite in self.composites
        }

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
                classification = processor.classify_glyphs(reader)
                island_names = {glyph.name for glyph in classification.glyphs_to_process}
                composites = find_bridged_composites(reader, island_names)
                component_outlines = load_component_outlines(reader, composites)
                displayed = island_names | {composite.name for composite in composites}
                display_names = tuple(
                    name for name in reader.font.getGlyphOrder() if name in displayed
                )
                return cls(
                    path=path,
                    font_format=reader.format,
                    units_per_em=reader.units_per_em,
                    glyph_count=reader.glyph_count,
                    ascender=int(reader.font["hhea"].ascent),
                    descender=int(reader.font["hhea"].descent),
                    classification=classification,
                    processor=processor,
                    source_sha256=source_sha256,
                    composites=composites,
                    component_outlines=component_outlines,
                    display_names=display_names,
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
        """Return an island or composed composite glyph by name."""
        return self._glyph_index.get(name) or self._composed.get(name)

    @property
    def display_glyphs(self) -> list[Glyph]:
        """Return each island and composite glyph in display order."""
        return [glyph for name in self.display_names if (glyph := self.glyph(name)) is not None]

    def direction_sources(self, name: str) -> tuple[str, ...]:
        """Return island glyphs whose direction affects ``name``."""
        if name in self._glyph_index:
            return (name,)
        composite = self._composite_index.get(name)
        return composite.sources if composite is not None else ()

    def _preview_island(
        self,
        glyph: Glyph,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection],
    ) -> PreviewResult:
        """Stencilize one island glyph with its requested direction."""
        configured_bridge = bridge.model_copy(
            update={"direction": directions.get(glyph.name, bridge.direction)}
        )
        result = process_glyph(
            glyph.to_dict(),
            configured_bridge.model_dump(),
            self.units_per_em,
            geometry_dict=geometry.model_dump(),
        )
        if "error" in result:
            return PreviewResult(
                glyph_name=glyph.name,
                original=glyph,
                stenciled=None,
                bridges_added=0,
                error=str(result["error"]),
                duration_ms=float(result["duration_ms"]),
            )
        return PreviewResult(
            glyph_name=glyph.name,
            original=glyph,
            stenciled=Glyph.from_dict(result["glyph"]),
            bridges_added=int(result["bridges_added"]),
            error=None,
            duration_ms=float(result["duration_ms"]),
        )

    def _preview_composite(
        self,
        composite: CompositeGlyph,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection],
    ) -> PreviewResult:
        """Stencilize each island source and compose the resulting outline."""
        stenciled_sources: dict[str, Glyph] = {}
        duration_ms = 0.0
        bridges_added = 0
        original = self._composed[composite.name]
        for source in composite.sources:
            result = self._preview_island(self._glyph_index[source], bridge, geometry, directions)
            duration_ms += result.duration_ms
            if result.error is not None:
                return PreviewResult(
                    composite.name, original, None, 0, f"{source}: {result.error}", duration_ms
                )
            if result.stenciled is None:
                return PreviewResult(
                    composite.name, original, None, 0, f"{source}: no preview outline", duration_ms
                )
            stenciled_sources[source] = result.stenciled
            bridges_added += result.bridges_added
        stenciled = compose(composite, {**self.component_outlines, **stenciled_sources})
        return PreviewResult(composite.name, original, stenciled, bridges_added, None, duration_ms)

    def preview(
        self,
        name: str,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> PreviewResult:
        """Stencilize one island glyph or its composed composite display glyph."""
        requested_directions = directions or {}
        glyph = self._glyph_index.get(name)
        if glyph is not None:
            return self._preview_island(glyph, bridge, geometry, requested_directions)
        composite = self._composite_index.get(name)
        if composite is not None:
            return self._preview_composite(composite, bridge, geometry, requested_directions)
        raise GlyphNotFoundError(name)

    def unbridged(
        self,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection] | None = None,
    ) -> frozenset[str]:
        """Return display glyph names for which no bridge could be placed."""
        requested_directions = directions or {}
        unbridged_names: set[str] = set()
        for glyph in self.island_glyphs:
            result = self._preview_island(glyph, bridge, geometry, requested_directions)
            if result.error is not None or result.bridges_added == 0:
                unbridged_names.add(glyph.name)
        for composite in self.composites:
            if all(source in unbridged_names for source in composite.sources):
                unbridged_names.add(composite.name)
        return frozenset(unbridged_names)

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
        directions: Mapping[str, BridgeDirection] | None = None,
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
                    directions=directions,
                )
                self._assert_source_unchanged(output_path)
                _publish(staged, output_path)
            return stats
        except StencilizerError:
            raise
        except Exception as error:
            raise FontSaveError(str(output_path), _save_failure_reason(error)) from error
