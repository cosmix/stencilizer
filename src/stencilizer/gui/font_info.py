"""Qt-free description of an open font for the sidebar information panel.

Every value is read defensively: a font may lack a table or a name record, and the panel is
informational, so a missing or unreadable item is skipped instead of failing the open.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fontTools.misc.timeTools import timestampToString  # type: ignore[import-untyped]

from stencilizer.core.processor import GlyphClassification
from stencilizer.gui.variable_session import read_axes

Row = tuple[str, str]

_SFNT_CONTAINERS = {
    "\x00\x01\x00\x00": "TrueType (.ttf)",
    "true": "TrueType (.ttf)",
    "OTTO": "OpenType (OTTO)",
}
_EMBEDDING_BITS = (
    (1, "Restricted license"),
    (2, "Preview & Print"),
    (3, "Editable"),
    (8, "No subsetting"),
    (9, "Bitmap only"),
)
_LEGAL_NAMES = (
    ("Copyright", 0),
    ("Trademark", 7),
    ("Manufacturer", 8),
    ("Designer", 9),
    ("Description", 10),
    ("Vendor URL", 11),
    ("Designer URL", 12),
    ("License", 13),
    ("License URL", 14),
)
_BYTES_PER_KB = 1024


@dataclass(frozen=True)
class InfoSection:
    """A titled group of ``(label, value)`` rows."""

    title: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class FontInfo:
    """Ordered information sections about one font, led by its display name."""

    title: str
    sections: tuple[InfoSection, ...]


def _table(font: Any, tag: str) -> Any | None:
    """Return the decompiled table, or None when absent or unreadable."""
    if tag not in font:
        return None
    try:
        return font[tag]
    except Exception:
        return None


def _name(font: Any, *name_ids: int) -> str | None:
    """Return the first non-empty name record among ``name_ids`` (Windows/English preferred)."""
    names = _table(font, "name")
    if names is None:
        return None
    for name_id in name_ids:
        text = names.getDebugName(name_id)
        if text and text.strip():
            return str(text).strip()
    return None


def _attr(table: Any | None, attribute: str) -> str | None:
    """Return a table attribute as text, or None when the table or attribute is missing."""
    value = getattr(table, attribute, None)
    if value is None:
        return None
    return f"{value:g}" if isinstance(value, float) else str(value)


def _section(title: str, rows: list[tuple[str, str | None]]) -> InfoSection:
    """Build a section from candidate rows, dropping those without a value."""
    return InfoSection(title, tuple((label, value) for label, value in rows if value))


def _file_size(path: Path) -> str | None:
    """Return the file size as ``168 KB`` or ``1.2 MB``."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < _BYTES_PER_KB:
        return f"{size} B"
    if size < _BYTES_PER_KB**2:
        return f"{size / _BYTES_PER_KB:.0f} KB"
    return f"{size / _BYTES_PER_KB**2:.1f} MB"


def _outlines(font: Any) -> str | None:
    """Describe the outline format."""
    if "CFF2" in font:
        return "CFF2 (cubic)"
    if "CFF " in font:
        return "CFF (cubic)"
    if "glyf" in font:
        return "TrueType (quadratic, glyf)"
    return None


def _container(font: Any) -> str | None:
    """Describe the sfnt container and any web-font wrapper."""
    base = _SFNT_CONTAINERS.get(font.sfntVersion)
    flavor = getattr(font, "flavor", None)
    if flavor:
        wrapper = str(flavor).upper()
        return f"{wrapper}, {base}" if base else wrapper
    return base


def _font_section(font: Any, path: Path) -> InfoSection:
    """Names, file name and size."""
    return _section(
        "Font",
        [
            ("Family", _name(font, 16, 1)),
            ("Style", _name(font, 17, 2)),
            ("Full name", _name(font, 4)),
            ("PostScript name", _name(font, 6)),
            ("Version", _name(font, 5)),
            ("Unique ID", _name(font, 3)),
            ("File name", path.name),
            ("File size", _file_size(path)),
        ],
    )


def _format_section(font: Any) -> InfoSection:
    """Outline format, container and variation axes."""
    rows: list[tuple[str, str | None]] = [
        ("Outlines", _outlines(font)),
        ("Container", _container(font)),
    ]
    fvar = _table(font, "fvar")
    if fvar is None:
        rows.append(("Variable", "No"))
        return _section("Format", rows)
    axes = read_axes(font)
    rows.append(
        ("Variable", f"Yes — {len(axes)} axes, {len(fvar.instances)} named instances"),
    )
    for axis in axes:
        span = f"{axis.minimum:g} \u2013 {axis.default:g} \u2013 {axis.maximum:g}"
        rows.append(
            (f"Axis {axis.tag}", span if axis.name == axis.tag else f"{span} ({axis.name})")
        )
    return _section("Format", rows)


def _metrics_section(font: Any) -> InfoSection:
    """Vertical metrics, classes and the font bounding box."""
    head, hhea, post = (_table(font, tag) for tag in ("head", "hhea", "post"))
    os2 = _table(font, "OS/2")
    has_height = getattr(os2, "version", 0) >= 2
    bounds = None
    if head is not None:
        bounds = " ".join(
            str(getattr(head, key, "")) for key in ("xMin", "yMin", "xMax", "yMax")
        ).strip()
    return _section(
        "Metrics",
        [
            ("Units per em", _attr(head, "unitsPerEm")),
            ("Ascender", _attr(hhea, "ascent")),
            ("Descender", _attr(hhea, "descent")),
            ("Line gap", _attr(hhea, "lineGap")),
            ("Typo ascender", _attr(os2, "sTypoAscender")),
            ("Typo descender", _attr(os2, "sTypoDescender")),
            ("Typo line gap", _attr(os2, "sTypoLineGap")),
            ("Win ascent", _attr(os2, "usWinAscent")),
            ("Win descent", _attr(os2, "usWinDescent")),
            ("Cap height", _attr(os2, "sCapHeight") if has_height else None),
            ("x-height", _attr(os2, "sxHeight") if has_height else None),
            ("Italic angle", _attr(post, "italicAngle")),
            ("Weight class", _attr(os2, "usWeightClass")),
            ("Width class", _attr(os2, "usWidthClass")),
            ("Underline position", _attr(post, "underlinePosition")),
            ("Underline thickness", _attr(post, "underlineThickness")),
            ("Bounding box", bounds),
        ],
    )


def _composite_total(font: Any) -> int | None:
    """Count composite glyphs in a ``glyf`` font; CFF has none."""
    glyf = _table(font, "glyf")
    if glyf is None:
        return None
    return sum(1 for name in font.getGlyphOrder() if glyf[name].isComposite())


def _glyphs_section(
    font: Any, classification: GlyphClassification, composites: int, displayed: int
) -> InfoSection:
    """Glyph, island and composite counts."""
    process = classification.glyphs_to_process
    unsupported = classification.unsupported_islands
    cmap = font.getBestCmap() or {}
    rows: list[tuple[str, str | None]] = [
        ("Glyphs", str(len(font.getGlyphOrder()))),
        ("Mapped code points", str(len(cmap))),
        ("Glyphs with islands", str(len(process))),
        ("Islands", str(sum(len(glyph.get_islands()) for glyph in process))),
    ]
    if unsupported:
        rows.append(
            (
                "Unsupported (variable data)",
                f"{len(unsupported)} glyphs, {sum(unsupported.values())} islands",
            )
        )
    total = _composite_total(font)
    rows += [
        ("Bridged composites", str(composites)),
        ("Composite glyphs", str(total) if total is not None else None),
        ("Shown in grid", str(displayed)),
        ("Skipped glyphs", str(classification.skipped_count)),
    ]
    return _section("Glyphs", rows)


def _feature_tags(table: Any | None) -> set[str]:
    """Return the feature tags of a GSUB or GPOS table."""
    layout = getattr(table, "table", None)
    feature_list = getattr(layout, "FeatureList", None)
    return {record.FeatureTag for record in getattr(feature_list, "FeatureRecord", [])}


def _script_tags(*tables: Any) -> set[str]:
    """Return the script tags of GSUB and GPOS tables."""
    tags: set[str] = set()
    for table in tables:
        script_list = getattr(getattr(table, "table", None), "ScriptList", None)
        tags.update(record.ScriptTag for record in getattr(script_list, "ScriptRecord", []))
    return tags


def _layout_section(font: Any) -> InfoSection:
    """GSUB and GPOS features, scripts and kerning source."""
    gsub, gpos = _table(font, "GSUB"), _table(font, "GPOS")
    gpos_features = _feature_tags(gpos)
    if "kern" in gpos_features:
        kerning = "GPOS"
    elif "kern" in font:
        kerning = "kern table"
    else:
        kerning = "None"
    return _section(
        "OpenType layout",
        [
            ("GSUB features", ", ".join(sorted(_feature_tags(gsub)))),
            ("GPOS features", ", ".join(sorted(gpos_features))),
            ("Scripts", ", ".join(sorted(_script_tags(gsub, gpos)))),
            ("Kerning", kerning),
        ],
    )


def _embedding(os2: Any | None) -> str | None:
    """Decode the OS/2 ``fsType`` embedding permissions."""
    fs_type = getattr(os2, "fsType", None)
    if fs_type is None:
        return None
    flags = [label for bit, label in _EMBEDDING_BITS if fs_type & (1 << bit)]
    return ", ".join(flags) if flags else "Installable"


def _legal_section(font: Any) -> InfoSection:
    """Copyright, license, credits, vendor ID and embedding permissions."""
    os2 = _table(font, "OS/2")
    rows: list[tuple[str, str | None]] = [(label, _name(font, i)) for label, i in _LEGAL_NAMES]
    vendor = getattr(os2, "achVendID", None)
    rows.append(("Vendor ID", str(vendor).strip() if vendor else None))
    rows.append(("Embedding", _embedding(os2)))
    return _section("Legal & credits", rows)


def _date(head: Any | None, attribute: str) -> str | None:
    """Format a ``head`` timestamp, or None when it is absent or out of range."""
    stamp = getattr(head, attribute, None)
    if stamp is None:
        return None
    try:
        return str(timestampToString(stamp))
    except (ValueError, OverflowError, OSError):
        return None


def build_font_info(
    font: Any,
    path: Path,
    classification: GlyphClassification,
    composites: int,
    displayed: int,
) -> FontInfo:
    """Describe ``font`` (an open fontTools ``TTFont``) and the session's glyph counts."""
    head = _table(font, "head")
    # TTFont defines keys() but no __iter__, so iterating it directly raises KeyError.
    tags = sorted(tag.strip() for tag in font.keys() if tag != "GlyphOrder")  # noqa: SIM118
    sections = (
        _font_section(font, path),
        _format_section(font),
        _metrics_section(font),
        _glyphs_section(font, classification, composites, displayed),
        _layout_section(font),
        _section("Tables", [("Count", str(len(tags))), ("Tags", ", ".join(tags))]),
        _legal_section(font),
        _section(
            "Dates", [("Created", _date(head, "created")), ("Modified", _date(head, "modified"))]
        ),
    )
    title = (
        _name(font, 4)
        or " ".join(filter(None, (_name(font, 16, 1), _name(font, 17, 2))))
        or path.name
    )
    return FontInfo(title, tuple(section for section in sections if section.rows))
