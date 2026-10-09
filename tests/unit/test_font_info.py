"""Tests for the Qt-free font information builder."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import LoggingConfig, StencilizerSettings
from stencilizer.core import FontProcessor
from stencilizer.core.processor import GlyphClassification
from stencilizer.gui.font_info import FontInfo, build_font_info
from stencilizer.gui.session import FontSession
from tests.font_helpers import COMMIT_MONO, INTER, LATO_BLACK, ROBOTO, UBUNTU


def _rows(info: FontInfo) -> dict[str, str]:
    """Flatten every section into one label to value mapping."""
    return {label: value for section in info.sections for label, value in section.rows}


def _open_info(path: Path, log_dir: Path) -> FontInfo:
    """Information gathered by opening ``path`` the way the GUI does."""
    settings = StencilizerSettings(logging=LoggingConfig(log_file=log_dir / "gui.log"))
    return FontSession.open(path, FontProcessor(settings)).info


@pytest.fixture(scope="module")
def roboto_info(tmp_path_factory: pytest.TempPathFactory) -> FontInfo:
    """Information for Roboto, opened once for the module."""
    return _open_info(ROBOTO, tmp_path_factory.mktemp("roboto"))


def test_roboto_format_metrics_and_counts(roboto_info: FontInfo) -> None:
    """Roboto reports TrueType outlines, its UPM and the session's glyph counts."""
    rows = _rows(roboto_info)

    assert rows["Outlines"] == "TrueType (quadratic, glyf)"
    assert rows["Container"] == "TrueType (.ttf)"
    assert rows["Variable"] == "No"
    assert rows["Units per em"] == "2048"
    assert rows["Glyphs with islands"] == "562"
    assert rows["Bridged composites"] == "465"
    assert rows["Shown in grid"] == "1027"
    assert rows["File name"] == "Roboto-Regular.ttf"
    assert rows["File size"].endswith("KB")
    assert rows["Copyright"]
    assert "kern" in rows["GPOS features"] or "liga" in rows["GSUB features"]
    assert roboto_info.title == rows["Full name"]


def test_commit_mono_is_cff(tmp_path: Path) -> None:
    """A CFF font reports cubic outlines in an OTTO container and no composite total."""
    rows = _rows(_open_info(COMMIT_MONO, tmp_path))

    assert rows["Outlines"] == "CFF (cubic)"
    assert rows["Container"] == "OpenType (OTTO)"
    assert "Composite glyphs" not in rows


def test_variable_font_lists_axes(tmp_path: Path) -> None:
    """A variable font lists its axes with minimum, default and maximum."""
    rows = _rows(_open_info(INTER, tmp_path))

    assert rows["Variable"].startswith("Yes")
    axis_rows = [value for label, value in rows.items() if label.startswith("Axis ")]
    assert axis_rows
    assert all(value.count("\u2013") >= 2 for value in axis_rows)


def test_missing_tables_and_names_do_not_crash(tmp_path: Path) -> None:
    """A font without OS/2, name, post or GSUB still yields an information object."""
    font = TTFont(ROBOTO)
    for tag in ("OS/2", "name", "post", "GSUB", "GPOS", "kern"):
        if tag in font:
            del font[tag]
    info = build_font_info(font, tmp_path / "missing.ttf", GlyphClassification(), 0, 0)
    rows = _rows(info)

    assert info.title == "missing.ttf"
    assert rows["Kerning"] == "None"
    assert "Copyright" not in rows
    assert "Cap height" not in rows


@pytest.mark.parametrize("path", [ROBOTO, LATO_BLACK, COMMIT_MONO, UBUNTU, INTER])
def test_real_fonts_open_with_info(path: Path, tmp_path: Path) -> None:
    """Opening a real font through the session builds its info, tables and dates included."""
    rows = _rows(_open_info(path, tmp_path))

    assert rows["File name"] == path.name
    assert "glyf" in rows["Tags"] or "CFF" in rows["Tags"]
    assert int(rows["Count"]) > 5
