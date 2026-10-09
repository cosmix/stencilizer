"""Tests for per-glyph bridge direction on the GUI font session."""

from pathlib import Path

import pytest
from fontTools.pens.recordingPen import DecomposingRecordingPen  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig
from stencilizer.config.settings import BridgeDirection, GeometryConfig
from stencilizer.core import FontProcessor
from stencilizer.gui.session import FontSession
from stencilizer.io import FontReader
from stencilizer.io.converter import _recording_to_contours
from tests.font_helpers import FIXTURES_DIR
from tests.gui.conftest import build_settings

LATO_UNBRIDGED_DEFAULT = frozenset(
    {
        "glyph00144",
        "glyph00558",
        "glyph01217",
        "glyph01224",
        "glyph01225",
        "glyph01235",
        "glyph01248",
        "glyph01249",
        "glyph01259",
        "glyph01263",
        "glyph01264",
        "glyph01272",
        "glyph01289",
        "glyph01290",
        "glyph02905",
        "glyph02907",
        "uni0234",
        "uni0235",
        "uni0236",
        "uni026C",
        "uni0286",
        "uni029D",
        "uni02B6",
        "uni02E0",
        "uni0363",
        "uni1D43",
        "uni1D44",
        "uni1D9D",
        "uni1DA8",
        "uni1DB6",
        "uni1DBD",
        "uni2090",
    }
)


def _geometry() -> GeometryConfig:
    """Return the default geometry configuration for previews."""
    return GeometryConfig()


def _spans(bbox: tuple[float, float, float, float], value: float) -> bool:
    """Return whether an (min_x, min_y, max_x, max_y) bbox spans a coordinate on the y axis."""
    return bbox[1] < value < bbox[3]


def test_display_glyphs_include_bridged_composites(
    processor: FontProcessor, roboto_path: Path, commit_mono_path: Path
) -> None:
    """Display glyphs list island glyphs and composites, in the font's glyph order."""
    session = FontSession.open(roboto_path, processor)

    assert len(session.display_glyphs) == 1027
    assert len(session.island_glyphs) == 562
    with FontReader(roboto_path) as reader:
        order = {name: index for index, name in enumerate(reader.font.getGlyphOrder())}
    assert list(session.display_names) == sorted(session.display_names, key=order.__getitem__)
    assert "Aacute" in session.display_names
    assert session.direction_sources("O") == ("O",)
    assert session.direction_sources("Aacute") == ("A",)
    assert session.direction_sources("Aring") == ("A", "ring")
    assert session.direction_sources("space") == ()

    commit_mono_session = FontSession.open(commit_mono_path, processor)
    island_names = {glyph.name for glyph in commit_mono_session.island_glyphs}
    assert set(commit_mono_session.display_names) == island_names


def test_preview_applies_glyph_direction(processor: FontProcessor, roboto_path: Path) -> None:
    """A direction supplied for a glyph overrides the bridge config for that glyph only."""
    session = FontSession.open(roboto_path, processor)

    via_directions = session.preview(
        "O", BridgeConfig(), _geometry(), {"O": BridgeDirection.HORIZONTAL}
    )
    via_config = session.preview(
        "O", BridgeConfig(direction=BridgeDirection.HORIZONTAL), _geometry()
    )
    auto = session.preview("O", BridgeConfig(), _geometry())
    other_glyph_direction = session.preview(
        "O", BridgeConfig(), _geometry(), {"A": BridgeDirection.HORIZONTAL}
    )

    assert via_directions.stenciled is not None
    assert via_config.stenciled is not None
    assert auto.stenciled is not None
    assert other_glyph_direction.stenciled is not None
    assert [c.to_dict() for c in via_directions.stenciled.contours] == [
        c.to_dict() for c in via_config.stenciled.contours
    ]
    assert via_directions.stenciled.to_dict() != auto.stenciled.to_dict()
    assert other_glyph_direction.stenciled.to_dict() == auto.stenciled.to_dict()


def test_composite_preview_follows_base_direction(
    processor: FontProcessor, roboto_path: Path
) -> None:
    """A composite's preview is built from its base glyph's directed preview."""
    session = FontSession.open(roboto_path, processor)
    directions = {"A": BridgeDirection.HORIZONTAL}

    a_preview = session.preview("A", BridgeConfig(), _geometry(), directions)
    aacute_preview = session.preview("Aacute", BridgeConfig(), _geometry(), directions)
    aacute_auto = session.preview("Aacute", BridgeConfig(), _geometry())

    assert a_preview.stenciled is not None
    assert aacute_preview.stenciled is not None
    assert aacute_auto.stenciled is not None
    a_contours = [c.to_dict() for c in a_preview.stenciled.contours]
    aacute_contours = [c.to_dict() for c in aacute_preview.stenciled.contours]
    assert all(contour in aacute_contours for contour in a_contours)
    assert aacute_preview.bridges_added == a_preview.bridges_added
    assert aacute_preview.stenciled.to_dict() != aacute_auto.stenciled.to_dict()
    assert session.preview("Aacute", BridgeConfig(), _geometry()).original is session.glyph(
        "Aacute"
    )


def test_unbridged_lists_glyphs_without_bridges(
    processor: FontProcessor, lato_black_path: Path
) -> None:
    """Glyphs where no bridge could be placed are reported, direction overrides included."""
    session = FontSession.open(lato_black_path, processor)

    default_unbridged = session.unbridged(BridgeConfig(), _geometry())
    with_direction = session.unbridged(
        BridgeConfig(), _geometry(), {"A": BridgeDirection.HORIZONTAL}
    )

    assert default_unbridged == LATO_UNBRIDGED_DEFAULT
    assert "A" not in with_direction
    assert "Aacute" not in with_direction


@pytest.mark.usefixtures("staging_root")
@pytest.mark.parametrize(
    "source_name", ["Roboto-Regular.ttf", "Lato-Black.ttf"], ids=["roboto", "lato"]
)
def test_saved_font_changes_only_displayed_glyphs(
    processor: FontProcessor,
    tmp_path: Path,
    source_name: str,
) -> None:
    """Saving changes only glyphs that display; every unchanged display glyph is unbridged."""
    source = FIXTURES_DIR / source_name
    session = FontSession.open(source, processor)
    output_path = tmp_path / "out.ttf"

    stats = session.save(output_path, build_settings(processor))

    assert stats.error_count == 0
    with FontReader(source) as input_reader, FontReader(output_path) as output_reader:
        input_glyph_set = input_reader.font.getGlyphSet()
        output_glyph_set = output_reader.font.getGlyphSet()
        changed: set[str] = set()
        for name in input_reader.font.getGlyphOrder():
            input_pen = DecomposingRecordingPen(input_glyph_set)
            input_glyph_set[name].draw(input_pen)
            output_pen = DecomposingRecordingPen(output_glyph_set)
            output_glyph_set[name].draw(output_pen)
            if input_pen.value != output_pen.value:
                changed.add(name)

    display_names = set(session.display_names)
    unbridged_names = session.unbridged(BridgeConfig(), _geometry())
    assert changed <= display_names
    for name in display_names - changed:
        assert name in unbridged_names


@pytest.mark.usefixtures("staging_root")
def test_save_applies_directions(
    processor: FontProcessor, roboto_path: Path, tmp_path: Path
) -> None:
    """A save applies its explicit per-glyph directions to the written outlines."""
    session = FontSession.open(roboto_path, processor)
    output_path = tmp_path / "out.ttf"

    stats = session.save(
        output_path, build_settings(processor), directions={"O": BridgeDirection.HORIZONTAL}
    )

    assert stats.error_count == 0
    with FontReader(output_path) as reader:
        saved_o = reader.get_glyph("O")
        assert saved_o is not None
        assert not any(_spans(c.bounding_box(), 728) for c in saved_o.contours)

        glyph_set = reader.font.getGlyphSet()
        pen = DecomposingRecordingPen(glyph_set)
        glyph_set["Oacute"].draw(pen)
        oacute_contours = _recording_to_contours(pen.value)
        assert not any(_spans(c.bounding_box(), 728) for c in oacute_contours)
