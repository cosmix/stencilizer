"""Diagnostic tests to understand what's happening with problematic glyphs."""

from pathlib import Path

import pytest

from stencilizer.config import BridgeConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.bridge import BridgeGenerator, BridgePlacer
from stencilizer.core.geometry import signed_area
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.io import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LATO_BLACK_PATH = FIXTURES_DIR / "Lato-Black.ttf"


def get_glyph_by_char(reader: FontReader, char: str):
    """Get a glyph by its character."""
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


def get_glyph_by_name(reader: FontReader, name: str):
    """Get a glyph by its name."""
    return reader.get_glyph(name)


def analyze_contours(contours, label=""):
    """Print detailed analysis of contours."""
    print(f"\n=== {label} ===")
    print(f"Total contours: {len(contours)}")

    total_outer_area = 0
    total_hole_area = 0

    for i, contour in enumerate(contours):
        area = signed_area(contour.points)
        bbox = contour.bounding_box()
        winding = "CW (outer)" if area < 0 else "CCW (hole)" if area > 0 else "ZERO"
        print(f"  Contour {i}: {len(contour.points)} pts, area={area:.1f}, {winding}")
        print(f"    bbox: x=[{bbox[0]:.0f}, {bbox[2]:.0f}], y=[{bbox[1]:.0f}, {bbox[3]:.0f}]")

        if area < 0:
            total_outer_area += abs(area)
        else:
            total_hole_area += area

    print(f"Total outer area: {total_outer_area:.1f}")
    print(f"Total hole area: {total_hole_area:.1f}")
    print(f"Net filled area: {total_outer_area - total_hole_area:.1f}")

    return total_outer_area, total_hole_area


class TestDiagnostic:
    """Diagnostic tests to understand glyph transformation issues."""

    @pytest.fixture
    def reader(self):
        if not LATO_BLACK_PATH.exists():
            pytest.skip("Lato Black font not available")
        reader = FontReader(LATO_BLACK_PATH)
        reader.load()
        yield reader
        reader.close()

    @pytest.fixture
    def transformer(self):
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0, use_spanning_bridges=True)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        return GlyphTransformer(analyzer, placer, generator, bridge_config=config)

    def test_diagnose_registered(self, reader, transformer):
        """Diagnose ® transformation."""
        glyph = get_glyph_by_char(reader, "®")
        if glyph is None:
            pytest.skip("® not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL ® ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED ®")

        # Hole area should not decrease dramatically
        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

        assert hole_ratio >= 0.5, f"Lost too much hole area: {hole_ratio:.2%}"

    def test_diagnose_at_sign(self, reader, transformer):
        """Diagnose @ transformation."""
        glyph = get_glyph_by_char(reader, "@")
        if glyph is None:
            pytest.skip("@ not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL @ ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED @")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

        # @ has an oval hole - when bridge cuts through middle, loses more than 50%
        assert hole_ratio >= 0.4, f"Lost too much hole area: {hole_ratio:.2%}"

    def test_diagnose_phi(self, reader, transformer):
        """Diagnose φ transformation."""
        glyph = get_glyph_by_char(reader, "φ")
        if glyph is None:
            pytest.skip("φ not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL φ ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED φ")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

        assert hole_ratio >= 0.5, f"Lost too much hole area: {hole_ratio:.2%}"

    def test_diagnose_ampersand(self, reader, transformer):
        """Diagnose & transformation."""
        glyph = get_glyph_by_char(reader, "&")
        if glyph is None:
            pytest.skip("& not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL & ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED &")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

        assert hole_ratio >= 0.5, f"Lost too much hole area: {hole_ratio:.2%}"

    def test_diagnose_f_ligatures(self, reader, transformer):
        """Diagnose f ligature transformations."""
        # Unicode f-ligatures: ff, fi, fl, ffi, ffl
        ligature_names = ["uniFB00", "uniFB01", "uniFB02", "uniFB03", "uniFB04"]

        for name in ligature_names:
            glyph = get_glyph_by_name(reader, name)
            if glyph is None:
                continue

            print(f"\n{'='*60}")
            _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL {name}")

            if orig_hole == 0:
                print(f"No holes in {name}, skipping")
                continue

            transformed = transformer.transform(glyph, upm=reader.units_per_em)

            _, new_hole = analyze_contours(transformed.contours, f"TRANSFORMED {name}")

            hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
            print(f"\nHole area ratio for {name}: {hole_ratio:.2%}")

            if hole_ratio < 0.5:
                print(f"WARNING: Lost too much hole area in {name}!")

    def test_diagnose_lowercase_f(self, reader, transformer):
        """Diagnose lowercase f - has a hook that creates a counter."""
        glyph = get_glyph_by_char(reader, "f")
        if glyph is None:
            pytest.skip("f not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL f ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED f")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

    def test_diagnose_lowercase_b(self, reader, transformer):
        """Diagnose lowercase b - has a bowl counter."""
        glyph = get_glyph_by_char(reader, "b")
        if glyph is None:
            pytest.skip("b not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL b ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED b")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"\nHole area ratio: {hole_ratio:.2%}")

    def test_diagnose_simple_o(self, reader, transformer):
        """Diagnose simple O transformation - this should work correctly."""
        glyph = get_glyph_by_char(reader, "O")
        if glyph is None:
            pytest.skip("O not found")

        _, orig_hole = analyze_contours(glyph.contours, f"ORIGINAL O ({glyph.name})")

        transformed = transformer.transform(glyph, upm=reader.units_per_em)

        _, new_hole = analyze_contours(transformed.contours, "TRANSFORMED O")

        # For simple O, we should have 4 contours: 2 outer + 2 hole (left/right pieces)
        print(f"\nExpected: 4 contours (2 outer + 2 hole)")
        print(f"Got: {len(transformed.contours)} contours")

        hole_ratio = new_hole / orig_hole if orig_hole > 0 else 1.0
        print(f"Hole area ratio: {hole_ratio:.2%}")

        # O should preserve most of its hole area (some lost to bridge gap)
        assert hole_ratio >= 0.7, f"Lost too much hole area: {hole_ratio:.2%}"
