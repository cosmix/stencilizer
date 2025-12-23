"""End-to-end test that writes stencilized font and verifies output."""

import tempfile
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from stencilizer.config import StencilizerSettings
from stencilizer.core.processor import FontProcessor

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LATO_BLACK_PATH = FIXTURES_DIR / "Lato-Black.ttf"


def analyze_glyph_from_font(font: TTFont, glyph_name: str):
    """Analyze a glyph directly from fontTools TTFont."""
    glyf_table = font["glyf"]
    glyph = glyf_table[glyph_name]

    if glyph.numberOfContours <= 0:
        return None, None, None

    # Get coordinates
    coords = glyph.coordinates
    end_pts = glyph.endPtsOfContours

    contours = []
    start = 0
    for end in end_pts:
        contour_coords = coords[start : end + 1]
        contours.append(contour_coords)
        start = end + 1

    # Calculate areas
    outer_area = 0
    hole_area = 0

    for contour in contours:
        # Calculate signed area
        n = len(contour)
        if n < 3:
            continue
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += contour[i][0] * contour[j][1]
            area -= contour[j][0] * contour[i][1]
        area /= 2.0

        if area < 0:
            outer_area += abs(area)
        else:
            hole_area += area

    return len(contours), outer_area, hole_area


class TestEndToEndOutput:
    """Test that stencilized fonts have correct winding in output file."""

    def test_stencilize_and_verify_b(self):
        """Stencilize font and verify 'b' glyph in output."""
        if not LATO_BLACK_PATH.exists():
            pytest.skip("Lato Black not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Lato-Stencil.ttf"

            # Process the font
            stats = processor.process(
                font_path=LATO_BLACK_PATH,
                output_path=output_path,
                max_workers=1,
            )

            print(f"\nProcessed {stats.processed_count} glyphs")
            print(f"Added {stats.bridges_added} bridges")

            # Load original and output
            orig_font = TTFont(str(LATO_BLACK_PATH))
            out_font = TTFont(str(output_path))

            # Analyze 'b' in both
            orig_contours, orig_outer, orig_hole = analyze_glyph_from_font(orig_font, "b")
            out_contours, out_outer, out_hole = analyze_glyph_from_font(out_font, "b")

            print(f"\nOriginal 'b': {orig_contours} contours, outer={orig_outer:.0f}, hole={orig_hole:.0f}")
            print(f"Output 'b': {out_contours} contours, outer={out_outer:.0f}, hole={out_hole:.0f}")

            orig_font.close()
            out_font.close()

            # Verify holes exist in output
            assert out_hole is not None and out_hole > 0, f"'b' lost all holes! outer={out_outer}, hole={out_hole}"

            # Verify hole area didn't decrease too much
            if orig_hole is not None and orig_hole > 0:
                ratio = out_hole / orig_hole
                print(f"Hole ratio: {ratio:.2%}")
                assert ratio >= 0.5, f"'b' lost too much hole area: {ratio:.2%}"

    def test_stencilize_and_verify_registered(self):
        """Stencilize font and verify ® glyph in output."""
        if not LATO_BLACK_PATH.exists():
            pytest.skip("Lato Black not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Lato-Stencil.ttf"

            processor.process(
                font_path=LATO_BLACK_PATH,
                output_path=output_path,
                max_workers=1,
            )

            orig_font = TTFont(str(LATO_BLACK_PATH))
            out_font = TTFont(str(output_path))

            # Get glyph name for ®
            cmap = orig_font.getBestCmap()
            glyph_name = cmap.get(ord("®")) if cmap is not None else None

            if glyph_name:
                orig_contours, orig_outer, orig_hole = analyze_glyph_from_font(orig_font, glyph_name)
                out_contours, out_outer, out_hole = analyze_glyph_from_font(out_font, glyph_name)

                print(f"\nOriginal '®': {orig_contours} contours, outer={orig_outer:.0f}, hole={orig_hole:.0f}")
                print(f"Output '®': {out_contours} contours, outer={out_outer:.0f}, hole={out_hole:.0f}")

                if orig_hole is not None and orig_hole > 0 and out_hole is not None:
                    ratio = out_hole / orig_hole
                    print(f"Hole ratio: {ratio:.2%}")
                    assert ratio >= 0.3, f"'®' lost too much hole area: {ratio:.2%}"

            orig_font.close()
            out_font.close()

    def test_compare_all_glyphs_with_holes(self):
        """Compare all glyphs that had holes - verify they still have holes."""
        if not LATO_BLACK_PATH.exists():
            pytest.skip("Lato Black not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Lato-Stencil.ttf"

            processor.process(
                font_path=LATO_BLACK_PATH,
                output_path=output_path,
                max_workers=1,
            )

            orig_font = TTFont(str(LATO_BLACK_PATH))
            out_font = TTFont(str(output_path))

            glyph_names = orig_font.getGlyphOrder()

            lost_holes = []
            preserved = 0
            total_with_holes = 0

            for name in glyph_names:
                try:
                    _, _, orig_hole = analyze_glyph_from_font(orig_font, name)
                    if orig_hole is None or orig_hole == 0:
                        continue

                    total_with_holes += 1
                    _, _, out_hole = analyze_glyph_from_font(out_font, name)

                    if out_hole == 0:
                        lost_holes.append(f"{name}: had {orig_hole:.0f} hole area, now 0")
                    else:
                        preserved += 1

                except Exception:
                    pass

            orig_font.close()
            out_font.close()

            print(f"\nTotal glyphs with holes: {total_with_holes}")
            print(f"Preserved holes: {preserved}")
            print(f"Lost ALL holes: {len(lost_holes)}")

            if lost_holes:
                print("\nGlyphs that lost ALL holes:")
                for item in lost_holes[:20]:
                    print(f"  {item}")

            # At least 90% should preserve holes
            preservation_rate = preserved / total_with_holes if total_with_holes > 0 else 1.0
            print(f"\nPreservation rate: {preservation_rate:.1%}")

            assert preservation_rate >= 0.9, (
                f"Too many glyphs lost holes: {preservation_rate:.1%} preserved"
            )
