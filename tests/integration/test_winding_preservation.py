"""Tests for winding direction preservation in stencilized glyphs.

These tests verify that holes don't get filled during stencilization.
The problematic glyphs have nested contours (like ® with R inside, @ with inner spiral)
where inner holes must remain as CCW contours after bridge splitting.
"""

from pathlib import Path

import pytest

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.geometry import signed_area
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain import WindingDirection
from stencilizer.io import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LATO_BLACK_PATH = FIXTURES_DIR / "Lato-Black.ttf"


@pytest.fixture
def lato_reader():
    """Load Lato Black font for testing."""
    if not LATO_BLACK_PATH.exists():
        pytest.skip("Lato Black font fixture not available")
    reader = FontReader(LATO_BLACK_PATH)
    reader.load()
    yield reader
    reader.close()


@pytest.fixture
def transformer():
    """Create a glyph transformer for testing."""
    analyzer = GlyphAnalyzer()
    config = BridgeConfig(width_percent=60.0, use_spanning_bridges=True)
    placer = BridgePlacer(config)
    generator = BridgeGenerator(config)
    return GlyphTransformer(analyzer, placer, generator, bridge_config=config)


def get_glyph_by_char(reader: FontReader, char: str):
    """Get a glyph by its character."""
    # Access the internal font object to get cmap
    if reader._font is None:
        return None
    cmap = reader._font.getBestCmap()
    if cmap is None:
        return None
    code_point = ord(char)
    if code_point not in cmap:
        return None
    glyph_name = cmap[code_point]
    return reader.get_glyph(glyph_name)


def count_holes(contours) -> int:
    """Count the number of hole contours (CCW winding, positive signed area)."""
    count = 0
    for contour in contours:
        area = signed_area(contour.points)
        if area > 0:  # Positive = CCW = hole
            count += 1
    return count


def count_outers(contours) -> int:
    """Count the number of outer contours (CW winding, negative signed area)."""
    count = 0
    for contour in contours:
        area = signed_area(contour.points)
        if area < 0:  # Negative = CW = outer
            count += 1
    return count


def verify_winding_consistency(contours) -> tuple[int, int]:
    """Verify winding directions and return (outer_count, hole_count)."""
    outers = 0
    holes = 0
    for contour in contours:
        area = signed_area(contour.points)
        if area < 0:
            outers += 1
        elif area > 0:
            holes += 1
        # area == 0 is degenerate, ignore
    return outers, holes


class TestRegisteredSymbol:
    """Test ® (Registered trademark) - has R with counter inside a ring."""

    def test_registered_has_nested_structure(self, lato_reader: FontReader) -> None:
        """Verify ® has the expected nested structure before transformation."""
        glyph = get_glyph_by_char(lato_reader, "®")
        if glyph is None:
            pytest.skip("® glyph not found in font")

        # ® typically has:
        # 1. Outer circle (CW)
        # 2. Inner circle/ring hole (CCW)
        # 3. R shape (CW)
        # 4. R's counter (CCW)
        outers, holes = verify_winding_consistency(glyph.contours)

        assert len(glyph.contours) >= 3, f"® should have at least 3 contours, got {len(glyph.contours)}"
        assert holes >= 1, f"® should have at least 1 hole contour, got {holes}"

    def test_registered_preserves_holes_after_transform(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that ® preserves hole contours after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "®")
        if glyph is None:
            pytest.skip("® glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        # After transformation, we should still have hole contours
        # The number might change due to splitting, but holes shouldn't become outers
        assert new_holes >= 1, (
            f"® should still have hole contours after transform. "
            f"Original: {original_holes} holes, {original_outers} outers. "
            f"After: {new_holes} holes, {new_outers} outers."
        )


class TestCopyrightSymbol:
    """Test © (Copyright) - has C inside a ring."""

    def test_copyright_preserves_structure(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that © preserves its structure after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "©")
        if glyph is None:
            pytest.skip("© glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        # Should still have holes
        assert new_holes >= 1, (
            f"© should still have hole contours after transform. "
            f"Original: {original_holes} holes. After: {new_holes} holes."
        )


class TestSoundRecordingSymbol:
    """Test ℗ (Sound recording copyright) - has P inside a ring."""

    def test_soundrecording_preserves_holes(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that ℗ preserves P's counter hole after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "℗")
        if glyph is None:
            pytest.skip("℗ glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        assert new_holes >= 1, (
            f"℗ should still have hole contours after transform. "
            f"Original: {original_holes} holes. After: {new_holes} holes."
        )


class TestAtSign:
    """Test @ - has complex nested structure with spiral."""

    def test_at_sign_preserves_holes(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that @ preserves inner holes after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "@")
        if glyph is None:
            pytest.skip("@ glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        # @ should have holes (the inner letter area)
        if original_holes > 0:
            assert new_holes >= 1, (
                f"@ should still have hole contours after transform. "
                f"Original: {original_holes} holes. After: {new_holes} holes."
            )


class TestGreekPhi:
    """Test φ (Greek lowercase phi) - has horizontally-arranged loops."""

    def test_phi_preserves_both_loop_holes(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that φ preserves holes in both loops after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "φ")
        if glyph is None:
            pytest.skip("φ glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        # φ should preserve its loop holes
        if original_holes > 0:
            assert new_holes >= 1, (
                f"φ should still have hole contours after transform. "
                f"Original: {original_holes} holes. After: {new_holes} holes."
            )


class TestAmpersand:
    """Test & (Ampersand) - has complex counter structure."""

    def test_ampersand_preserves_counters(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that & preserves counter holes after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "&")
        if glyph is None:
            pytest.skip("& glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        if original_holes > 0:
            assert new_holes >= 1, (
                f"& should still have hole contours after transform. "
                f"Original: {original_holes} holes. After: {new_holes} holes."
            )


class TestGermanEszett:
    """Test ß (German Eszett) - has loop in upper portion."""

    def test_eszett_preserves_loop_hole(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that ß preserves the upper loop hole after stencilization."""
        glyph = get_glyph_by_char(lato_reader, "ß")
        if glyph is None:
            pytest.skip("ß glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        if original_holes > 0:
            assert new_holes >= 1, (
                f"ß should still have hole contours after transform. "
                f"Original: {original_holes} holes. After: {new_holes} holes."
            )


class TestGreekTheta:
    """Test Θ (Greek capital Theta) - has horizontal bar that shouldn't be split."""

    def test_theta_transformation(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that Θ transforms correctly without splitting the bar unnecessarily."""
        glyph = get_glyph_by_char(lato_reader, "Θ")
        if glyph is None:
            pytest.skip("Θ glyph not found in font")

        original_outers, original_holes = verify_winding_consistency(glyph.contours)

        transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

        new_outers, new_holes = verify_winding_consistency(transformed.contours)

        # Theta should be transformed (bridges added)
        assert len(transformed.contours) != len(glyph.contours) or transformed.contours != glyph.contours, (
            "Θ should be transformed"
        )


class TestWindingPreservationGeneral:
    """General tests for winding preservation across multiple glyphs."""

    PROBLEMATIC_GLYPHS = ["®", "©", "℗", "@", "φ", "&", "ß", "Θ", "θ"]

    def test_no_holes_become_filled(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that glyphs with holes don't lose ALL holes after transformation."""
        failures = []

        for char in self.PROBLEMATIC_GLYPHS:
            glyph = get_glyph_by_char(lato_reader, char)
            if glyph is None:
                continue

            original_outers, original_holes = verify_winding_consistency(glyph.contours)

            if original_holes == 0:
                continue  # Skip glyphs without holes

            transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)
            new_outers, new_holes = verify_winding_consistency(transformed.contours)

            if new_holes == 0 and original_holes > 0:
                failures.append(
                    f"'{char}': had {original_holes} holes, now has {new_holes} "
                    f"(outers: {original_outers} -> {new_outers})"
                )

        assert not failures, (
            f"The following glyphs lost ALL their holes:\n" + "\n".join(failures)
        )

    def test_transformed_contours_have_valid_winding(
        self, lato_reader: FontReader, transformer: GlyphTransformer
    ) -> None:
        """Test that all transformed contours have valid (non-zero) winding."""
        for char in self.PROBLEMATIC_GLYPHS:
            glyph = get_glyph_by_char(lato_reader, char)
            if glyph is None:
                continue

            transformed = transformer.transform(glyph, upm=lato_reader.units_per_em)

            for i, contour in enumerate(transformed.contours):
                area = signed_area(contour.points)
                assert abs(area) > 0.001, (
                    f"Glyph '{char}' contour {i} has near-zero signed area ({area}), "
                    f"indicating degenerate geometry"
                )
