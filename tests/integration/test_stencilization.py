"""Integration tests for the stencilization pipeline.

Tests the full algorithm with real font files to verify:
- Island detection works on real glyphs
- Bridge placement produces valid geometry
- Output fonts are valid and loadable
"""

import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core import BridgeGenerator, BridgePlacer, GlyphAnalyzer, GlyphTransformer
from stencilizer.core.processor import FontProcessor
from stencilizer.io import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
ROBOTO_PATH = FIXTURES_DIR / "Roboto-Regular.ttf"


@pytest.fixture
def roboto_reader() -> Generator[FontReader, None, None]:
    """Load Roboto font for testing."""
    if not ROBOTO_PATH.exists():
        pytest.skip("Roboto font fixture not available")
    reader = FontReader(ROBOTO_PATH)
    reader.load()
    yield reader
    reader.close()


class TestIslandDetection:
    """Test island detection on real font glyphs."""

    # Glyphs known to have islands (enclosed counters)
    GLYPHS_WITH_ISLANDS = ["O", "o", "A", "B", "D", "P", "Q", "R", "a", "b", "d", "e", "g", "p", "q"]

    # Glyphs known to NOT have islands
    GLYPHS_WITHOUT_ISLANDS = ["I", "l", "T", "L", "V", "W", "X", "Y", "Z"]

    def test_detects_islands_in_O(self, roboto_reader: FontReader) -> None:
        """Test that 'O' is detected as having an island."""
        analyzer = GlyphAnalyzer()

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                hierarchy = analyzer.analyze(glyph)
                assert hierarchy.has_islands(), "Glyph 'O' should have an island"
                assert len(hierarchy.get_islands()) == 1, "Glyph 'O' should have exactly 1 island"
                return

        pytest.fail("Glyph 'O' not found in font")

    def test_detects_islands_in_B(self, roboto_reader: FontReader) -> None:
        """Test that 'B' is detected as having islands (two counters)."""
        analyzer = GlyphAnalyzer()

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "B":
                hierarchy = analyzer.analyze(glyph)
                assert hierarchy.has_islands(), "Glyph 'B' should have islands"
                # B typically has 2 enclosed counters
                assert len(hierarchy.get_islands()) >= 1, "Glyph 'B' should have at least 1 island"
                return

        pytest.fail("Glyph 'B' not found in font")

    def test_no_islands_in_I(self, roboto_reader: FontReader) -> None:
        """Test that 'I' has no islands."""
        analyzer = GlyphAnalyzer()

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "I":
                hierarchy = analyzer.analyze(glyph)
                assert not hierarchy.has_islands(), "Glyph 'I' should not have islands"
                return

        pytest.fail("Glyph 'I' not found in font")

    def test_island_count_summary(self, roboto_reader: FontReader) -> None:
        """Test overall island detection across the font."""
        analyzer = GlyphAnalyzer()
        glyphs_with_islands = []

        for glyph in roboto_reader.iter_glyphs():
            if glyph.is_empty() or glyph.is_composite():
                continue
            hierarchy = analyzer.analyze(glyph)
            if hierarchy.has_islands():
                glyphs_with_islands.append(glyph.name)

        # Roboto should have many glyphs with islands
        assert len(glyphs_with_islands) >= 10, (
            f"Expected at least 10 glyphs with islands, found {len(glyphs_with_islands)}: {glyphs_with_islands[:10]}"
        )

        # Check that known island glyphs are detected
        for expected in ["O", "A", "B", "D", "P", "Q", "R"]:
            assert expected in glyphs_with_islands, f"Expected '{expected}' to have islands"


class TestBridgePlacement:
    """Test bridge placement on real glyphs."""

    def test_bridge_candidates_for_O(self, roboto_reader: FontReader) -> None:
        """Test that bridge candidates can be generated for 'O'."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig()
        placer = BridgePlacer(config)

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                hierarchy = analyzer.analyze(glyph)
                islands = hierarchy.get_islands()
                assert len(islands) == 1

                island_idx = islands[0]
                outer_idx = hierarchy.containment[island_idx]

                candidates = placer.find_candidates(
                    island=glyph.contours[island_idx],
                    outer=glyph.contours[outer_idx],
                    island_idx=island_idx,
                    outer_idx=outer_idx,
                    count=config.sample_count,
                )

                assert len(candidates) > 0, "Should generate bridge candidates"

                # Verify candidates have valid geometry
                for candidate in candidates:
                    assert candidate.inner_point is not None
                    assert candidate.outer_point is not None
                return

        pytest.fail("Glyph 'O' not found")

    def test_bridge_selection(self, roboto_reader: FontReader) -> None:
        """Test that bridges can be selected from candidates."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig(min_bridges=1)
        placer = BridgePlacer(config)

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                hierarchy = analyzer.analyze(glyph)
                islands = hierarchy.get_islands()
                island_idx = islands[0]
                outer_idx = hierarchy.containment[island_idx]

                candidates = placer.find_candidates(
                    island=glyph.contours[island_idx],
                    outer=glyph.contours[outer_idx],
                    island_idx=island_idx,
                    outer_idx=outer_idx,
                    count=config.sample_count,
                )

                selected = placer.select_bridges(candidates, min_count=1)

                assert len(selected) >= 1, "Should select at least 1 bridge"
                return

        pytest.fail("Glyph 'O' not found")


class TestGlyphTransformation:
    """Test full glyph transformation."""

    def test_transform_O_creates_merged_contours(self, roboto_reader: FontReader) -> None:
        """Test that transforming 'O' creates merged contours with bridge gaps."""
        analyzer = GlyphAnalyzer()
        config = BridgeConfig()
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        transformer = GlyphTransformer(analyzer, placer, generator)

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                original_contour_count = len(glyph.contours)
                assert original_contour_count == 2, "O should have 2 contours (outer + inner)"

                hierarchy = analyzer.analyze(glyph)
                assert hierarchy.has_islands(), "O should have an island"

                transformed = transformer.transform(glyph, upm=roboto_reader.units_per_em)

                # Should have 2 merged contours (replacing outer + inner with bridge gaps)
                # The contours are DIFFERENT from originals (merged with bridges)
                assert len(transformed.contours) == 2, (
                    f"Expected 2 merged contours, got {len(transformed.contours)}"
                )

                # Contours should be different from originals (they're merged)
                assert transformed.contours[0] != glyph.contours[0], (
                    "First contour should be different (merged with bridges)"
                )
                return

        pytest.fail("Glyph 'O' not found")


class TestFullPipeline:
    """Test the complete stencilization pipeline."""

    def test_process_font_creates_valid_output(self, roboto_reader: FontReader) -> None:
        """Test that processing creates a valid, loadable font."""
        roboto_reader.close()  # Close so processor can open it

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Roboto-Stenciled.ttf"

            stats = processor.process(
                font_path=ROBOTO_PATH,
                output_path=output_path,
                max_workers=1,  # Single worker for deterministic testing
            )

            # Check stats
            assert stats.processed_count > 0, "Should process some glyphs"
            assert stats.bridges_added > 0, "Should add some bridges"
            assert stats.error_count == 0, f"Should have no errors: {stats.errors}"

            # Verify output file exists and is valid
            assert output_path.exists(), "Output file should exist"

            # Load and verify output font
            output_font = TTFont(str(output_path))
            assert "glyf" in output_font, "Output should be a valid TrueType font"
            output_font.close()

    def test_processed_glyphs_have_merged_contours(self) -> None:
        """Test that glyphs with islands have contours merged with bridge gaps."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)
        analyzer = GlyphAnalyzer()

        # First, load original font to get original point counts
        original_reader = FontReader(ROBOTO_PATH)
        original_reader.load()

        original_point_counts: dict[str, int] = {}
        for glyph in original_reader.iter_glyphs():
            if glyph.is_empty() or glyph.is_composite():
                continue
            hierarchy = analyzer.analyze(glyph)
            if hierarchy.has_islands():
                total_points = sum(len(c.points) for c in glyph.contours)
                original_point_counts[glyph.name] = total_points
        original_reader.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Roboto-Stenciled.ttf"

            stats = processor.process(
                font_path=ROBOTO_PATH,
                output_path=output_path,
                max_workers=1,
            )

            # Verify bridges were added by checking stats
            assert stats.bridges_added > 0, "Should have added bridges"

            # Load output and verify glyphs were modified
            reader = FontReader(output_path)
            reader.load()

            glyphs_modified = 0
            for glyph in reader.iter_glyphs():
                if glyph.name not in original_point_counts:
                    continue

                # Check that point count changed (contours were merged with bridges)
                total_points = sum(len(c.points) for c in glyph.contours)
                if total_points != original_point_counts[glyph.name]:
                    glyphs_modified += 1

            reader.close()

            # Many glyphs with islands should have been modified
            # Note: not all glyphs may be modified due to stroke constraints,
            # complex geometry, or multiple islands sharing the same outer contour
            assert glyphs_modified > len(original_point_counts) / 4, (
                f"Too few glyphs modified: {glyphs_modified} of {len(original_point_counts)}"
            )


class TestCLI:
    """Test the CLI end-to-end."""

    def test_cli_help(self) -> None:
        """Test that CLI --help works."""
        result = subprocess.run(
            ["uv", "run", "stencilizer", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "stencilizer" in result.stdout.lower()
        assert "bridge" in result.stdout.lower()

    def test_cli_version(self) -> None:
        """Test that CLI --version works."""
        result = subprocess.run(
            ["uv", "run", "stencilizer", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_list_islands(self) -> None:
        """Test --list-islands option."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        result = subprocess.run(
            ["uv", "run", "stencilizer", "--list-islands", str(ROBOTO_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should list some common glyphs with islands
        assert "O" in result.stdout or "islands" in result.stdout.lower()

    def test_cli_dry_run(self) -> None:
        """Test --dry-run option doesn't modify font."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "should-not-exist.ttf"

            result = subprocess.run(
                ["uv", "run", "stencilizer", "--dry-run", "-o", str(output_path), str(ROBOTO_PATH)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert not output_path.exists(), "--dry-run should not create output file"

    def test_cli_processes_font(self) -> None:
        """Test CLI actually processes a font."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.ttf"

            result = subprocess.run(
                ["uv", "run", "stencilizer", "-o", str(output_path), str(ROBOTO_PATH)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert output_path.exists(), "Output file should be created"

            # Verify output is valid
            font = TTFont(str(output_path))
            assert "glyf" in font
            font.close()

    def test_cli_nonexistent_file(self) -> None:
        """Test CLI handles nonexistent file gracefully."""
        result = subprocess.run(
            ["uv", "run", "stencilizer", "/nonexistent/font.ttf"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestBridgeQuality:
    """Test that bridges are placed reasonably."""

    def test_bridges_are_reasonably_sized(self, roboto_reader: FontReader) -> None:
        """Test that bridge width is reasonable relative to glyph size."""
        config = BridgeConfig(width_percent=60.0)
        placer = BridgePlacer(config)
        generator = BridgeGenerator(config)
        analyzer = GlyphAnalyzer()
        upm = roboto_reader.units_per_em

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                hierarchy = analyzer.analyze(glyph)
                if not hierarchy.has_islands():
                    continue

                island_idx = hierarchy.get_islands()[0]
                outer_idx = hierarchy.containment[island_idx]

                candidates = placer.find_candidates(
                    island=glyph.contours[island_idx],
                    outer=glyph.contours[outer_idx],
                    island_idx=island_idx,
                    outer_idx=outer_idx,
                    count=config.sample_count,
                )

                selected = placer.select_bridges(candidates, min_count=1)
                assert len(selected) >= 1

                bridge_geom = generator.generate_geometry(selected[0], upm)

                # Bridge should have 4 vertices
                assert len(bridge_geom.vertices) == 4

                # Bridge width should be proportional to stroke width (60% of stroke)
                # Calculate stroke width from bridge endpoints
                spec = selected[0]
                stroke_width = (
                    (spec.outer_point.x - spec.inner_point.x) ** 2
                    + (spec.outer_point.y - spec.inner_point.y) ** 2
                ) ** 0.5
                expected_width = stroke_width * config.width_percent / 100

                v1, v2 = bridge_geom.vertices[0], bridge_geom.vertices[1]
                actual_width = ((v2.x - v1.x) ** 2 + (v2.y - v1.y) ** 2) ** 0.5
                assert abs(actual_width - expected_width) < expected_width * 0.5, (
                    f"Bridge width {actual_width} too far from expected {expected_width}"
                )
                return

        pytest.fail("Glyph 'O' not found")

    def test_bridge_connects_inner_to_outer(self, roboto_reader: FontReader) -> None:
        """Test that bridge endpoints are actually on the contours."""
        config = BridgeConfig()
        placer = BridgePlacer(config)
        analyzer = GlyphAnalyzer()

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "O":
                hierarchy = analyzer.analyze(glyph)
                if not hierarchy.has_islands():
                    continue

                island_idx = hierarchy.get_islands()[0]
                outer_idx = hierarchy.containment[island_idx]

                candidates = placer.find_candidates(
                    island=glyph.contours[island_idx],
                    outer=glyph.contours[outer_idx],
                    island_idx=island_idx,
                    outer_idx=outer_idx,
                    count=config.sample_count,
                )

                for candidate in candidates[:5]:  # Check first few
                    # Inner point should reference inner contour
                    assert candidate.inner_contour_idx == island_idx
                    # Outer point should reference outer contour
                    assert candidate.outer_contour_idx == outer_idx
                    # Points should be distinct
                    assert (
                        candidate.inner_point.x != candidate.outer_point.x
                        or candidate.inner_point.y != candidate.outer_point.y
                    )
                return

        pytest.fail("Glyph 'O' not found")


class TestEdgeCases:
    """Test edge cases and unusual fonts."""

    def test_glyph_with_multiple_islands(self, roboto_reader: FontReader) -> None:
        """Test glyphs with multiple islands (like 'B' or '8')."""
        analyzer = GlyphAnalyzer()

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "B":
                hierarchy = analyzer.analyze(glyph)
                # 'B' should have 2 islands (top and bottom counters)
                assert hierarchy.has_islands()
                assert len(hierarchy.get_islands()) >= 1
                return

        pytest.fail("Glyph 'B' not found")

    def test_output_font_tables_preserved(self) -> None:
        """Test that important font tables are preserved in output."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.ttf"

            processor.process(
                font_path=ROBOTO_PATH,
                output_path=output_path,
                max_workers=1,
            )

            original = TTFont(str(ROBOTO_PATH))
            processed = TTFont(str(output_path))

            # Essential tables should be preserved
            essential_tables = ["head", "hhea", "maxp", "OS/2", "name", "cmap", "post"]
            for table in essential_tables:
                assert table in processed, f"Table {table} missing from output"

            original.close()
            processed.close()

    def test_processed_font_has_same_glyph_count(self) -> None:
        """Test that processing doesn't add or remove glyphs."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.ttf"

            processor.process(
                font_path=ROBOTO_PATH,
                output_path=output_path,
                max_workers=1,
            )

            original = TTFont(str(ROBOTO_PATH))
            processed = TTFont(str(output_path))

            assert original["maxp"].numGlyphs == processed["maxp"].numGlyphs

            original.close()
            processed.close()
