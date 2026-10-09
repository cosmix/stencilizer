"""Variable-font instancing helpers."""

from pathlib import Path
from typing import cast

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.varLib.instancer import (  # type: ignore[import-untyped]
    OverlapMode,
    instantiateVariableFont,
)

from stencilizer.exceptions import FontFormatError


def parse_instance_spec(spec: str, font: TTFont) -> dict[str, float]:
    """Validate an instance specification and fill in omitted axis defaults."""
    path = _font_path(font)
    if "fvar" not in font:
        raise FontFormatError(path, "--instance requires a variable font")

    axes = {str(axis.axisTag): axis for axis in font["fvar"].axes}
    limits = {tag: float(axis.defaultValue) for tag, axis in axes.items()}
    for item in spec.split(","):
        if "=" not in item:
            raise FontFormatError(path, f"malformed --instance item '{item}'")
        tag, value_text = item.split("=", 1)
        try:
            value = float(value_text)
        except ValueError as error:
            raise FontFormatError(path, f"malformed --instance item '{item}'") from error
        if tag not in axes:
            raise FontFormatError(path, f"unknown axis '{tag}'")
        axis = axes[tag]
        if not float(axis.minValue) <= value <= float(axis.maxValue):
            raise FontFormatError(
                path,
                f"axis '{tag}' value {value} outside {axis.minValue}..{axis.maxValue}",
            )
        limits[tag] = value
    return limits


def instantiate_static(font_path: Path, spec: str, workdir: Path) -> Path:
    """Write a static instance of ``font_path`` using the requested axis values."""
    opened: list[TTFont] = []
    try:
        font = TTFont(font_path)
        opened.append(font)
        limits = parse_instance_spec(spec, font)
        update_names = "STAT" in font
        try:
            instance = _instantiate(font, limits, update_names)
        except ValueError:
            if not update_names:
                raise
            font.close()
            font = TTFont(font_path)
            opened.append(font)
            instance = _instantiate(font, limits, False)
        opened.append(instance)
        output_path = workdir / f"{font_path.stem}-instance{font_path.suffix}"
        instance.save(output_path)
        return output_path
    finally:
        for handle in opened:
            handle.close()


def _instantiate(font: TTFont, limits: dict[str, float], update_names: bool) -> TTFont:
    return cast(
        "TTFont",
        instantiateVariableFont(
            font,
            limits,
            static=True,
            overlap=OverlapMode.REMOVE,
            downgradeCFF2="CFF2" in font,
            updateFontNames=update_names,
        ),
    )


def _font_path(font: TTFont) -> str:
    reader = getattr(font, "reader", None)
    file = getattr(reader, "file", None)
    return str(getattr(file, "name", "<font>"))
