"""Integration tests for spanning bridges and font formats."""

import tempfile
from pathlib import Path

import pytest
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.config import BridgeConfig, StencilizerSettings
from stencilizer.core import GlyphAnalyzer
from stencilizer.core.processor import FontProcessor
from stencilizer.io import FontReader
from tests.integration.conftest import COMMIT_MONO_OTF_PATH, ROBOTO_PATH


class TestSpanningBridges:
    """Test spanning bridges feature on shaped glyphs."""

    def test_glyph_eight_shape_uses_spanning_bridges(self, roboto_reader: FontReader) -> None:
        """Test that real glyphs with vertically-stacked islands use spanning bridges."""
        from stencilizer.core.analyzer import GlyphAnalyzer
        from stencilizer.core.surgery import GlyphTransformer

        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0, use_spanning_bridges=True)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

        # Use real glyph 'B' which has vertically-stacked islands
        for glyph in roboto_reader.iter_glyphs():
            if glyph.name == "B":
                hierarchy = analyzer.analyze(glyph)
                if not hierarchy.has_islands():
                    pytest.skip("Glyph 'B' doesn't have islands in this font")

                islands = hierarchy.get_islands()
                if len(islands) < 2:
                    pytest.skip("Glyph 'B' doesn't have multiple islands")

                original_contour_count = len(glyph.contours)
                transformed = transformer.transform(glyph, upm=roboto_reader.units_per_em)

                # With spanning bridges enabled, the glyph should be transformed
                # and should have different (typically fewer) contours
                assert len(transformed.contours) != original_contour_count, (
                    f"Expected transformation with spanning bridges, "
                    f"got same count: {len(transformed.contours)}"
                )

                # Verify the contours actually changed (not just same contours)
                assert transformed.contours != glyph.contours, (
                    "Contours should be different after transformation"
                )
                return

        pytest.fail("Glyph 'B' not found in font")

    def test_spanning_bridges_config_flag_exists(self) -> None:
        """Test that the use_spanning_bridges config flag exists and can be set."""

        # Test that the config flag can be set to True
        config_enabled = BridgeConfig(width_percent=60.0, use_spanning_bridges=True)
        assert config_enabled.use_spanning_bridges is True, (
            "use_spanning_bridges should be True when explicitly set"
        )

        # Test that the config flag can be set to False
        config_disabled = BridgeConfig(width_percent=60.0, use_spanning_bridges=False)
        assert config_disabled.use_spanning_bridges is False, (
            "use_spanning_bridges should be False when explicitly set"
        )

        # Test default value
        config_default = BridgeConfig(width_percent=60.0)
        assert hasattr(config_default, "use_spanning_bridges"), (
            "BridgeConfig should have use_spanning_bridges attribute"
        )

    def test_spanning_bridges_with_multiple_islands(self, roboto_reader: FontReader) -> None:
        """Test that spanning bridges work with glyphs that have 2+ vertically-stacked islands."""
        from stencilizer.core.analyzer import GlyphAnalyzer
        from stencilizer.core.surgery import GlyphTransformer

        analyzer = GlyphAnalyzer()
        config = BridgeConfig(width_percent=60.0, use_spanning_bridges=True)
        transformer = GlyphTransformer(analyzer=analyzer, bridge_config=config)

        # Try to find any glyph with 2+ islands
        candidates = ["B", "8", "g", "%"]

        for glyph in roboto_reader.iter_glyphs():
            if glyph.name in candidates:
                hierarchy = analyzer.analyze(glyph)
                if not hierarchy.has_islands():
                    continue

                islands = hierarchy.get_islands()
                if len(islands) < 2:
                    continue

                # Found a glyph with multiple islands
                original_contour_count = len(glyph.contours)
                transformed = transformer.transform(glyph, upm=roboto_reader.units_per_em)

                # Verify transformation occurred
                assert (
                    len(transformed.contours) != original_contour_count
                    or transformed.contours != glyph.contours
                ), f"Glyph '{glyph.name}' should be transformed with spanning bridges"

                # The result should have valid contours
                assert len(transformed.contours) >= 1, (
                    f"Transformed glyph should have at least 1 contour, got {len(transformed.contours)}"
                )
                return

        pytest.skip("No suitable multi-island glyph found in font")


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

    def test_output_font_tables_preserved(self, settings: StencilizerSettings) -> None:
        """Test that important font tables are preserved in output."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

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

    def test_processed_font_has_same_glyph_count(self, settings: StencilizerSettings) -> None:
        """Test that processing doesn't add or remove glyphs."""
        if not ROBOTO_PATH.exists():
            pytest.skip("Roboto font fixture not available")

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


class TestOpenTypeFonts:
    """Test OpenType (CFF) font processing."""

    def test_otf_font_detected_as_opentype(self, commit_mono_otf_reader: FontReader) -> None:
        """Test that OTF font is detected as OpenType format."""
        font_format = commit_mono_otf_reader.format
        assert font_format == "OpenType", f"Expected 'OpenType', got '{font_format}'"

    def test_process_otf_font_creates_valid_output(self, settings: StencilizerSettings) -> None:
        """Test that processing OTF creates a valid, loadable font."""
        if not COMMIT_MONO_OTF_PATH.exists():
            pytest.skip("CommitMono OTF font fixture not available")

        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "CommitMono-Stenciled.otf"

            stats = processor.process(
                font_path=COMMIT_MONO_OTF_PATH,
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
            assert "CFF " in output_font, "Output should be a valid OpenType/CFF font"
            output_font.close()

    def test_otf_glyphs_have_valid_charstrings(self, settings: StencilizerSettings) -> None:
        """Test that processed OTF glyphs have valid charstrings that can be drawn."""
        if not COMMIT_MONO_OTF_PATH.exists():
            pytest.skip("CommitMono OTF font fixture not available")

        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "CommitMono-Stenciled.otf"

            processor.process(
                font_path=COMMIT_MONO_OTF_PATH,
                output_path=output_path,
                max_workers=1,
            )

            # Load output font and verify glyphs can be drawn
            output_font = TTFont(str(output_path))
            glyph_set = output_font.getGlyphSet()

            # Test that we can draw glyphs with islands
            test_glyphs = ["O", "A", "B", "D", "P", "Q", "R", "a", "b", "d", "e", "p", "q"]
            glyphs_drawn = 0

            for glyph_name in test_glyphs:
                if glyph_name in glyph_set:
                    try:
                        # Try to draw the glyph - this validates the charstrings
                        pen = RecordingPen()
                        glyph_set[glyph_name].draw(pen)
                        glyphs_drawn += 1
                    except Exception as e:
                        pytest.fail(f"Failed to draw glyph '{glyph_name}': {e}")

            assert glyphs_drawn > 0, "Should have drawn at least some test glyphs"
            output_font.close()

    def test_otf_island_detection(self, commit_mono_otf_reader: FontReader) -> None:
        """Test that island detection works on CFF font glyphs.

        Note: CommitMono Cosmix is a stencil variant, so typical letters don't have islands.
        We test with special glyphs that do have islands like Theta, copyright, registered.
        """
        analyzer = GlyphAnalyzer()
        glyphs_with_islands = []

        for glyph in commit_mono_otf_reader.iter_glyphs():
            if glyph.is_empty() or glyph.is_composite():
                continue
            hierarchy = analyzer.analyze(glyph)
            if hierarchy.has_islands():
                glyphs_with_islands.append(glyph.name)

        # CommitMono Cosmix should have some glyphs with islands (special symbols)
        assert len(glyphs_with_islands) >= 5, (
            f"Expected at least 5 glyphs with islands, found {len(glyphs_with_islands)}: {glyphs_with_islands[:10]}"
        )

        # Check that special symbol glyphs with islands are detected
        # (using glyphs that actually have islands in this stencil font variant)
        symbol_island_glyphs = ["Theta", "copyright", "registered"]
        found_symbols = [g for g in symbol_island_glyphs if g in glyphs_with_islands]
        assert len(found_symbols) >= 2, (
            f"Expected at least 2 symbol island glyphs, found {len(found_symbols)}: {found_symbols}"
        )
