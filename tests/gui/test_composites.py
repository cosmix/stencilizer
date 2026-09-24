"""Tests for resolving and drawing composite glyphs."""

from pathlib import Path
from typing import Any, cast

import pytest
from fontTools.fontBuilder import FontBuilder  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import DecomposingRecordingPen  # type: ignore[import-untyped]
from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore[import-untyped]

from stencilizer.core import FontProcessor
from stencilizer.domain import Contour, Glyph, GlyphMetadata, WindingDirection
from stencilizer.gui.composites import (
    ComponentPart,
    CompositeGlyph,
    component_parts,
    compose,
    find_bridged_composites,
    load_component_outlines,
)
from stencilizer.io import FontReader
from stencilizer.io.converter import _recording_to_contours


def test_finds_bridged_composites_in_glyph_order(
    processor: FontProcessor, roboto_path: Path
) -> None:
    """Roboto composites reference island glyphs in font order with source details."""
    with FontReader(roboto_path) as reader:
        classification = processor.classify_glyphs(reader)
        composites = find_bridged_composites(
            reader, {glyph.name for glyph in classification.glyphs_to_process}
        )
        names = [composite.name for composite in composites]
        aacute = next(composite for composite in composites if composite.name == "Aacute")
        aring = next(composite for composite in composites if composite.name == "Aring")

        assert names == [name for name in reader.font.getGlyphOrder() if name in set(names)]
    assert len(composites) == 465
    assert aacute.sources == ("A",)
    assert [(part.base, part.transform) for part in aacute.parts] == [
        ("A", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        ("acute", (1.0, 0.0, 0.0, 1.0, 447.0, 310.0)),
    ]
    assert aring.sources == ("A", "ring")
    assert all(composite.sources for composite in composites)
    assert "O" not in names
    assert "space" not in names


def test_composed_outlines_match_fonttools_decomposition(
    processor: FontProcessor, roboto_path: Path
) -> None:
    """Every selected Roboto composite matches fontTools decomposition exactly."""
    with FontReader(roboto_path) as reader:
        classification = processor.classify_glyphs(reader)
        composites = find_bridged_composites(
            reader, {glyph.name for glyph in classification.glyphs_to_process}
        )
        outlines = load_component_outlines(reader, composites)
        glyph_set = reader.font.getGlyphSet()

        for composite in composites:
            pen = DecomposingRecordingPen(glyph_set)
            glyph_set[composite.name].draw(pen)
            expected = _recording_to_contours(pen.value)
            actual = compose(composite, outlines)
            _assert_matching_outlines(actual, expected)


def test_nested_and_mirrored_components(tmp_path: Path) -> None:
    """Nested components compose transforms and mirrored outlines retain winding."""
    path = _build_component_font(tmp_path / "components.ttf")
    with FontReader(path) as reader:
        glyph_set = reader.font.getGlyphSet()
        nested_parts = component_parts(glyph_set, "Onested")
        composites = find_bridged_composites(reader, {"O"})
        mirror = next(composite for composite in composites if composite.name == "Omirror")
        outlines = load_component_outlines(reader, composites)
        source = outlines["O"]
        # A freshly read outline carries no computed winding label; assign the labels an
        # analyzed source would have (matching its actual, already-asserted signed areas)
        # so composing through a mirror has a real label to preserve, not just None.
        source.contours[0].direction = WindingDirection.CLOCKWISE
        source.contours[1].direction = WindingDirection.COUNTER_CLOCKWISE
        mirrored = compose(mirror, outlines)

    assert [(part.base, part.transform) for part in nested_parts] == [
        ("O", (1.0, 0.0, 0.0, 1.0, 100.0, 0.0)),
        ("acute", (1.0, 0.0, 0.0, 1.0, 100.0, 500.0)),
    ]
    assert mirrored.contours[0].signed_area() < 0
    assert mirrored.contours[1].signed_area() > 0
    for actual, expected in zip(mirrored.contours, source.contours, strict=True):
        assert actual.direction == expected.direction
        assert (actual.direction is WindingDirection.CLOCKWISE) == (actual.signed_area() < 0)


def test_compose_leaves_inputs_untouched(processor: FontProcessor, roboto_path: Path) -> None:
    """Repeated composition is equal and does not alter the loaded source outlines."""
    with FontReader(roboto_path) as reader:
        classification = processor.classify_glyphs(reader)
        composites = find_bridged_composites(
            reader, {glyph.name for glyph in classification.glyphs_to_process}
        )
        composite = next(item for item in composites if item.name == "Aacute")
        outlines = load_component_outlines(reader, [composite])
        original = {name: glyph.to_dict() for name, glyph in outlines.items()}
        first = compose(composite, outlines)
        second = compose(composite, outlines)

    assert first == second
    assert {name: glyph.to_dict() for name, glyph in outlines.items()} == original


class _StubComponentGlyph:
    """A glyph-set entry that draws either an outline or fixed component references."""

    def __init__(self, components: tuple[str, ...] = (), *, outline: bool = False) -> None:
        """Store the component bases (or outline flag) this stub draws."""
        self._components = components
        self._outline = outline

    def draw(self, pen: Any) -> None:
        """Draw an outline leaf, or an ``addComponent`` call per stored base."""
        if self._outline:
            pen.moveTo((0, 0))
            pen.lineTo((1, 0))
            pen.closePath()
        for base in self._components:
            pen.addComponent(base, (1, 0, 0, 1, 0, 0))


def test_cyclic_components_raise() -> None:
    """A chain that revisits a glyph already on its path raises instead of recursing forever."""
    self_referencing = {"A": _StubComponentGlyph(("A",))}
    with pytest.raises(ValueError, match="references itself"):
        component_parts(self_referencing, "A")

    two_glyph_loop = {
        "A": _StubComponentGlyph(("B",)),
        "B": _StubComponentGlyph(("A",)),
    }
    with pytest.raises(ValueError, match="references itself"):
        component_parts(two_glyph_loop, "A")

    diamond = {
        "leaf": _StubComponentGlyph(outline=True),
        "C": _StubComponentGlyph(("leaf",)),
        "D": _StubComponentGlyph(("leaf",)),
        "Parent": _StubComponentGlyph(("C", "D")),
    }
    parts = component_parts(diamond, "Parent")
    assert [part.base for part in parts] == ["leaf", "leaf"]


def test_component_expansion_is_bounded() -> None:
    """A wide or deep untrusted component graph is rejected instead of exploding or hanging."""
    doubling = {f"level{i}": _StubComponentGlyph((f"level{i + 1}",) * 2) for i in range(40)}
    doubling["level40"] = _StubComponentGlyph(outline=True)
    with pytest.raises(ValueError, match="composite glyph"):
        component_parts(doubling, "level0")

    chain = {f"chain{i}": _StubComponentGlyph((f"chain{i + 1}",)) for i in range(40)}
    chain["chain40"] = _StubComponentGlyph(outline=True)
    with pytest.raises(ValueError, match="composite glyph"):
        component_parts(chain, "chain0")

    legitimate = {
        "leaf": _StubComponentGlyph(outline=True),
        "mid": _StubComponentGlyph(("leaf",)),
        "top": _StubComponentGlyph(("mid",)),
    }
    assert [part.base for part in component_parts(legitimate, "top")] == ["leaf"]


class _MissingBaseReader:
    """A reader stub whose glyph lookup always reports a missing base."""

    def get_glyph(self, _name: str) -> Glyph | None:
        """Report every requested glyph as missing."""
        return None


def test_missing_component_base_raises() -> None:
    """A composite whose base glyph cannot be loaded is reported, not silently dropped."""
    composite = CompositeGlyph(
        metadata=GlyphMetadata(name="composed", unicode=None, advance_width=0, left_side_bearing=0),
        parts=(ComponentPart("missing", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),),
        sources=(),
    )
    with pytest.raises(ValueError, match="missing"):
        load_component_outlines(cast("FontReader", _MissingBaseReader()), [composite])


def _assert_matching_outlines(actual: Glyph, expected: list[Contour]) -> None:
    """Check decomposed contours point-for-point with a float tolerance."""
    assert len(actual.contours) == len(expected)
    for actual_contour, expected_contour in zip(actual.contours, expected, strict=True):
        assert len(actual_contour.points) == len(expected_contour.points)
        for actual_point, expected_point in zip(
            actual_contour.points, expected_contour.points, strict=True
        ):
            assert actual_point.point_type is expected_point.point_type
            assert abs(actual_point.x - expected_point.x) < 1e-6
            assert abs(actual_point.y - expected_point.y) < 1e-6


def _build_component_font(path: Path) -> Path:
    """Build a tiny TrueType font containing nested and mirrored components."""
    glyphs = {
        ".notdef": _outline(()),
        "O": _outline(
            ((0, 0), (0, 400), (400, 400), (400, 0)),
            ((100, 100), (300, 100), (300, 300), (100, 300)),
        ),
        "acute": _outline(((0, 0), (100, 0), (50, 100))),
        "Oacute": _components(("O", (1, 0, 0, 1, 0, 0)), ("acute", (1, 0, 0, 1, 0, 500))),
        "Onested": _components(("Oacute", (1, 0, 0, 1, 100, 0))),
        "Omirror": _components(("O", (-1, 0, 0, 1, 600, 0))),
    }
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(list(glyphs))
    builder.setupCharacterMap({})
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(dict.fromkeys(glyphs, (600, 0)))
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Components", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)
    return path


def _outline(*contours: tuple[tuple[int, int], ...]) -> object:
    """Create a simple TrueType outline glyph from polygon contours."""
    pen = TTGlyphPen(None)
    for contour in contours:
        if contour:
            pen.moveTo(contour[0])
            for point in contour[1:]:
                pen.lineTo(point)
            pen.closePath()
    return pen.glyph()


def _components(*parts: tuple[str, tuple[int, int, int, int, int, int]]) -> object:
    """Create a TrueType glyph from component references."""
    pen = TTGlyphPen(dict.fromkeys(name for name, _ in parts))
    for name, transform in parts:
        pen.addComponent(name, transform)
    return pen.glyph()
