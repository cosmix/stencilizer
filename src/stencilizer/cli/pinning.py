"""Helpers for creating and publishing pinned variable-font instances."""

import copy
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.exceptions import FontLoadError, FontSaveError
from stencilizer.io.instance import instantiate_static, parse_instance_spec
from stencilizer.io.writer import update_font_names
from stencilizer.variable.reader import is_variable


def instance_workdir() -> tempfile.TemporaryDirectory[str]:
    """Temporary directory that holds pinned instances and stenciled intermediates."""
    return tempfile.TemporaryDirectory(prefix="stencilizer-instance-")


@contextmanager
def _open_lazy(path: Path) -> Iterator[TTFont]:
    """Open ``path`` without loading its tables; an open failure becomes ``FontLoadError``."""
    try:
        font = TTFont(path, lazy=True)
    except Exception as error:
        raise FontLoadError(str(path), str(error)) from error
    try:
        yield font
    finally:
        font.close()


@contextmanager
def pinned_input(input_font: Path, instance: str | None) -> Iterator[Path]:
    """Yield the input font, or a temporary static instance when requested."""
    if instance is None:
        yield input_font
        return
    with instance_workdir() as tmp:
        yield instantiate_static(input_font, instance, Path(tmp))


def validate_instance(input_font: Path, instance: str) -> None:
    """Validate an instance specification against the input font's axes."""
    with _open_lazy(input_font) as font:
        parse_instance_spec(instance, font)


def pin_stenciled(stenciled: Path, source: Path, instance: str, workdir: Path) -> Path:
    """Restore source names to a stenciled font, then write its static instance."""
    stenciled_font = TTFont(stenciled)
    try:
        source_font = TTFont(source)
        try:
            stenciled_font["name"] = copy.deepcopy(source_font["name"])
            renamed_path = workdir / f"{stenciled.stem}-renamed{stenciled.suffix}"
            stenciled_font.save(renamed_path)
        finally:
            source_font.close()
    finally:
        stenciled_font.close()
    return instantiate_static(renamed_path, instance, workdir)


def publish_pinned(pinned: Path, output_path: Path) -> Path:
    """Write a pinned font with its stencilized names at ``output_path``."""
    try:
        with tempfile.TemporaryDirectory(prefix=".stencilizer-pin-", dir=output_path.parent) as tmp:
            staged = Path(tmp) / output_path.name
            font = TTFont(pinned)
            try:
                update_font_names(font)
                font.save(staged)
            finally:
                font.close()
            staged.replace(output_path)
    except OSError as error:
        raise FontSaveError(str(output_path), str(error)) from error
    return output_path


def is_variable_font(path: Path) -> bool:
    """Return whether ``path`` contains a variable-font table."""
    with _open_lazy(path) as font:
        return is_variable(font)


def is_cff2_font(path: Path) -> bool:
    """Return whether ``path`` contains CFF2 outlines."""
    with _open_lazy(path) as font:
        return "CFF2" in font
