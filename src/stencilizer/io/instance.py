"""Variable-font instancing helpers."""

from pathlib import Path
from typing import cast

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.varLib.instancer import (  # type: ignore[import-untyped]
    OverlapMode,
    instantiateVariableFont,
)

from stencilizer.exceptions import FontFormatError
from stencilizer.io.writer import NAME_ID_FAMILY, NAME_ID_FULL_NAME

NAME_ID_SUBFAMILY = 2


class InstanceSpecError(FontFormatError):
    """A bad ``--instance`` value, reported as a usage error that does not blame the font file."""

    def __str__(self) -> str:
        return self.details


def parse_instance_spec(spec: str, font: TTFont) -> dict[str, float]:
    """Validate an instance specification and fill in omitted axis defaults."""
    path = _font_path(font)
    if "fvar" not in font:
        raise InstanceSpecError(path, "--instance requires a variable font")

    axes = {str(axis.axisTag): axis for axis in font["fvar"].axes}
    limits = {tag: float(axis.defaultValue) for tag, axis in axes.items()}
    seen: set[str] = set()
    for item in spec.split(","):
        tag, value = _parse_item(item, path)
        if tag not in axes:
            raise _invalid_spec(path, f"unknown axis '{tag}'")
        if tag in seen:
            raise _invalid_spec(path, f"axis '{tag}' given more than once")
        seen.add(tag)
        axis = axes[tag]
        if not float(axis.minValue) <= value <= float(axis.maxValue):
            raise _invalid_spec(
                path, f"axis '{tag}' value {value} outside {axis.minValue}..{axis.maxValue}"
            )
        limits[tag] = value
    return limits


def _parse_item(item: str, path: str) -> tuple[str, float]:
    malformed = _invalid_spec(path, f"malformed item '{item}'")
    tag, separator, value_text = item.partition("=")
    if not separator:
        raise malformed
    try:
        return tag, float(value_text)
    except ValueError as error:
        raise malformed from error


def _invalid_spec(path: str, reason: str) -> InstanceSpecError:
    return InstanceSpecError(path, f"Invalid --instance: {reason}")


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
        _add_style_to_full_name(instance)
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


def _add_style_to_full_name(font: TTFont) -> None:
    """Append the style to a full name (ID 4) that only repeats the family name (ID 1).

    The instancer leaves ID 4 as just the family when it finds no STAT name for the location,
    and ``update_font_names`` takes the last word of ID 4 for the style, so it would otherwise
    insert " Stenciled" in front of the last word of the family.
    """
    names = font["name"]
    for record in list(names.names):
        if record.nameID != NAME_ID_FULL_NAME:
            continue
        key = (record.platformID, record.platEncID, record.langID)
        family = names.getName(NAME_ID_FAMILY, *key)
        style = names.getName(NAME_ID_SUBFAMILY, *key)
        if family is not None and style is not None and str(record) == str(family):
            names.setName(f"{family} {style}", NAME_ID_FULL_NAME, *key)


def _font_path(font: TTFont) -> str:
    reader = getattr(font, "reader", None)
    file = getattr(reader, "file", None)
    return str(getattr(file, "name", "<font>"))
