# CFF2 Font Support Implementation Plan

## Executive Summary

This document details the implementation plan for adding static CFF2 (OpenType-CFF2) font support to stencilizer. CFF2 shares the same Type 2 charstring operators as CFF, making this a moderate-complexity feature.

**Scope**: Static CFF2 fonts only. Variable CFF2 fonts (with `fvar` table or blend operators) will receive a clear error message.

---

## 1. Technical Background

### 1.1 What is CFF2?

CFF2 (Compact Font Format version 2) is an updated PostScript outline format introduced in OpenType 1.8 primarily to support variable fonts. Key characteristics:

- Uses same Type 2 charstring operators as CFF for drawing (moveTo, lineTo, curveTo)
- Cubic Bezier curves (same as CFF)
- Same winding convention as CFF (opposite to TrueType)
- Can be static (single master) or variable (multiple masters with interpolation)

### 1.2 CFF vs CFF2 Structural Differences

| Aspect                  | CFF                            | CFF2                                |
| ----------------------- | ------------------------------ | ----------------------------------- |
| Table name in font      | `"CFF "` (with trailing space) | `"CFF2"`                            |
| TopDict access          | `cff.topDictIndex[0]`          | `cff.topDictIndex[0]` (same)        |
| GlobalSubrs             | `cff.GlobalSubrs`              | `cff.GlobalSubrs` (same)            |
| Private Dict            | `topDict.Private` (single)     | Via `FDArray[fdIndex].Private`      |
| FDSelect                | Only for CID-keyed fonts       | Always present (even for single FD) |
| Width in charstring     | First operand encodes width    | Width NOT in charstring             |
| `endchar` operator      | Required at end                | Not used                            |
| `T2CharStringPen` param | `CFF2=False` (default)         | `CFF2=True, width=None`             |
| Max stack depth         | 48                             | 513                                 |
| Blend operators         | Not supported                  | `blend`, `vsindex` for variation    |

### 1.3 fonttools API for CFF2

From `.venv/lib/python3.13/site-packages/fontTools/pens/t2CharStringPen.py`:

```python
class T2CharStringPen(BasePen):
    def __init__(
        self,
        width: float | None,
        glyphSet: Dict[str, Any] | None,
        roundTolerance: float = 0.5,
        CFF2: bool = False,  # <-- Key parameter for CFF2
    ):
        # When CFF2=True:
        # - Sets maxstack to 513 (vs 48)
        # - Asserts width is None (not encoded in CFF2)
        # - Omits endchar operator from output
```

### 1.4 Current CFF Implementation

**Reading** (`converter.py:18-67`):

```python
def fonttools_glyph_to_domain(name, fonttools_glyph, font):
    pen = RecordingPen()
    fonttools_glyph.draw(pen)  # Works for CFF, CFF2, and TrueType

    contours = _recording_to_contours(pen.value)

    # CFF fonts use opposite winding convention from TrueType
    is_cff = "CFF " in font
    if is_cff:
        for contour in contours:
            contour.points = list(reversed(contour.points))
```

**Writing** (`converter.py:243-301`):

```python
def _update_cff_glyph(glyph, _, font):
    cff_table = font["CFF "]
    top_dict = cff_table.cff.topDictIndex[0]
    charstrings = top_dict.CharStrings
    private = top_dict.Private
    global_subrs = cff_table.cff.GlobalSubrs

    pen = T2CharStringPen(
        width=glyph.metadata.advance_width,
        glyphSet=font.getGlyphSet()
    )

    # Draw contours (reversed to restore CFF winding)
    # ...

    charstring = pen.getCharString(private=private, globalSubrs=global_subrs)
    charstrings[glyph_name] = charstring
```

---

## 2. Implementation Details

### 2.1 File: `src/stencilizer/io/converter.py`

#### Change 1: Update imports (line 10)

No new imports needed. `T2CharStringPen` already imported.

#### Change 2: Update winding normalization (lines 49-54)

**Current code:**

```python
# CFF fonts use opposite winding convention from TrueType.
# Reverse contour points to normalize to TrueType convention.
is_cff = "CFF " in font
if is_cff:
    for contour in contours:
        contour.points = list(reversed(contour.points))
```

**New code:**

```python
# CFF/CFF2 fonts use opposite winding convention from TrueType.
# Reverse contour points to normalize to TrueType convention.
is_cff_variant = "CFF " in font or "CFF2" in font
if is_cff_variant:
    for contour in contours:
        contour.points = list(reversed(contour.points))
```

**Rationale:** CFF2 uses identical winding convention to CFF. The `RecordingPen` already handles CFF2 drawing commands correctly, so we just need to ensure winding normalization applies.

#### Change 3: Update write dispatcher (lines 88-95)

**Current code:**

```python
def domain_glyph_to_fonttools(
    glyph: Glyph,
    original_glyph: Any,
    font: TTFont
) -> None:
    is_truetype = "glyf" in font

    if is_truetype:
        _update_truetype_glyph(glyph, original_glyph, font)
    elif "CFF " in font:
        _update_cff_glyph(glyph, original_glyph, font)
    else:
        raise NotImplementedError("Unsupported font format")
```

**New code:**

```python
def domain_glyph_to_fonttools(
    glyph: Glyph,
    original_glyph: Any,
    font: TTFont
) -> None:
    """Update fonttools glyph from domain model.

    Converts domain Glyph back to fonttools representation and updates
    the glyph in place. Handles TrueType, OpenType/CFF, and OpenType/CFF2 formats.

    Args:
        glyph: Domain glyph model with modifications
        original_glyph: Original fonttools glyph object to update
        font: The TTFont object

    Raises:
        NotImplementedError: If glyph format is not supported or is variable CFF2
    """
    is_truetype = "glyf" in font

    if is_truetype:
        _update_truetype_glyph(glyph, original_glyph, font)
    elif "CFF " in font:
        _update_cff_glyph(glyph, original_glyph, font)
    elif "CFF2" in font:
        if _is_variable_cff2(font):
            raise NotImplementedError(
                "Variable CFF2 fonts are not supported. "
                "Please use a static (non-variable) font or convert to static first."
            )
        _update_cff2_glyph(glyph, original_glyph, font)
    else:
        raise NotImplementedError("Unsupported font format")
```

#### Change 4: Add variable CFF2 detection function (after line 95)

```python
def _is_variable_cff2(font: TTFont) -> bool:
    """Check if font is a variable CFF2 font.

    Variable CFF2 fonts contain interpolation data that we cannot preserve
    when modifying glyphs. This function detects variable fonts by checking
    for the presence of variation-related tables and data.

    Args:
        font: The TTFont object

    Returns:
        True if font is variable CFF2, False if static CFF2
    """
    if "CFF2" not in font:
        return False

    # Check for Font Variations table (definitive indicator)
    if "fvar" in font:
        return True

    # Check for VariationStore in TopDict (may exist without fvar in edge cases)
    try:
        cff2 = font["CFF2"]
        top_dict = cff2.cff.topDictIndex[0]
        if hasattr(top_dict, 'VarStore') and top_dict.VarStore is not None:
            return True
    except (KeyError, IndexError, AttributeError):
        pass

    return False
```

#### Change 5: Add CFF2 glyph update function (after `_update_cff_glyph`, around line 302)

```python
def _update_cff2_glyph(glyph: Glyph, _: Any, font: TTFont) -> None:
    """Update CFF2/OpenType glyph from domain model.

    CFF2 differs from CFF in several ways:
    - Uses FDArray for Private dicts (via FDSelect to map glyph index to FD index)
    - Does not encode glyph width in charstrings (width is in hmtx only)
    - Does not use endchar operator
    - Has larger max stack depth (513 vs 48)

    Note: Domain contours use TrueType winding convention (normalized on read).
    We must reverse points when writing back to restore CFF winding convention.

    Args:
        glyph: Domain glyph model
        _: Original fonttools glyph (unused)
        font: The TTFont object
    """
    cff2_table = font["CFF2"]
    top_dict = cff2_table.cff.topDictIndex[0]
    charstrings = top_dict.CharStrings
    glyph_name = glyph.name
    global_subrs = cff2_table.cff.GlobalSubrs

    # Get the correct Private dict for this glyph
    # CFF2 uses FDArray + FDSelect to map glyphs to Font Dicts
    private = _get_cff2_private_for_glyph(top_dict, glyph_name, charstrings)

    # CFF2 mode: width=None (not encoded in charstring), CFF2=True
    pen = T2CharStringPen(
        width=None,
        glyphSet=font.getGlyphSet(),  # type: ignore[arg-type]
        CFF2=True
    )

    for contour in glyph.contours:
        # Reverse points to restore CFF winding convention
        points = list(reversed(contour.points))
        if not points:
            continue

        first_point = points[0]
        pen.moveTo((first_point.x, first_point.y))

        i = 1
        while i < len(points):
            point = points[i]

            if point.point_type == PointType.ON_CURVE:
                pen.lineTo((point.x, point.y))
                i += 1

            elif point.point_type == PointType.OFF_CURVE_CUBIC:
                # Cubic Bezier curve: two control points + end point
                if i + 2 < len(points):
                    p1 = point
                    p2 = points[i + 1]
                    p3 = points[i + 2]

                    pen.curveTo(
                        (p1.x, p1.y),
                        (p2.x, p2.y),
                        (p3.x, p3.y)
                    )
                    i += 3
                else:
                    # Incomplete curve data, skip
                    i += 1

            else:
                # Skip unexpected point types (e.g., quadratic in CFF2)
                i += 1

        pen.closePath()

    charstring = pen.getCharString(private=private, globalSubrs=global_subrs)
    charstrings[glyph_name] = charstring


def _get_cff2_private_for_glyph(
    top_dict: Any,
    glyph_name: str,
    charstrings: Any
) -> Any:
    """Get the correct Private dict for a glyph in CFF2.

    CFF2 uses FDArray (array of Font Dicts, each with its own Private dict)
    and FDSelect (mapping from glyph index to FD index).

    For simple fonts with a single FD, this returns FDArray[0].Private.
    For multi-FD fonts (CID-keyed or merged fonts), uses FDSelect to find
    the correct FD index for the glyph.

    Args:
        top_dict: CFF2 TopDict containing FDArray and FDSelect
        glyph_name: Name of the glyph to look up
        charstrings: CharStrings dict for resolving glyph indices

    Returns:
        Private dict for the glyph

    Note:
        Falls back to FDArray[0].Private if FDSelect is missing or
        glyph cannot be found. This handles simple single-FD fonts
        and provides graceful degradation for edge cases.
    """
    fd_array = top_dict.FDArray

    # Check if FDSelect exists and is populated
    if hasattr(top_dict, 'FDSelect') and top_dict.FDSelect is not None:
        fd_select = top_dict.FDSelect

        # Get glyph index from charstrings order
        # CharStrings is an IndexedDict in fonttools, keys() gives glyph order
        if hasattr(charstrings, 'keys'):
            charset = list(charstrings.keys())
            if glyph_name in charset:
                glyph_index = charset.index(glyph_name)

                # FDSelect can be accessed by index
                # Format 0: array of FD indices
                # Format 3: range-based (fonttools abstracts this)
                try:
                    fd_index = fd_select[glyph_index]
                    if 0 <= fd_index < len(fd_array):
                        return fd_array[fd_index].Private
                except (IndexError, TypeError, KeyError):
                    pass

    # Fallback: use first FD (works for all single-FD fonts)
    return fd_array[0].Private
```

### 2.2 File: `src/stencilizer/io/reader.py`

**No changes required.**

The `format` property at lines 63-64 already handles CFF2 detection:

```python
if "CFF " in self._font or "CFF2" in self._font:
    return "OpenType"
return "TrueType"
```

Reading works because:

1. `fonttools.TTFont` loads CFF2 tables automatically
2. `font.getGlyphSet()` returns a GlyphSet that works for all formats
3. `RecordingPen` captures drawing commands abstractly

### 2.3 File: `src/stencilizer/README.md`

#### Update Font Format Support table (lines 285-291)

**Current:**

```markdown
| Format                          | Extension     | Outline Type              | Status             |
| ------------------------------- | ------------- | ------------------------- | ------------------ |
| TrueType                        | `.ttf`        | TrueType (`glyf` table)   | ✅ Fully supported |
| OpenType with TrueType outlines | `.otf`        | TrueType (`glyf` table)   | ✅ Fully supported |
| OpenType with CFF outlines      | `.otf`        | PostScript (`CFF` table)  | ✅ Fully supported |
| OpenType with CFF2 outlines     | `.otf`        | PostScript (`CFF2` table) | ❌ Not supported   |
| Variable fonts                  | `.ttf`/`.otf` | Variable (`fvar` table)   | ❌ Not supported   |
```

**New:**

```markdown
| Format                                 | Extension     | Outline Type               | Status             |
| -------------------------------------- | ------------- | -------------------------- | ------------------ |
| TrueType                               | `.ttf`        | TrueType (`glyf` table)    | ✅ Fully supported |
| OpenType with TrueType outlines        | `.otf`        | TrueType (`glyf` table)    | ✅ Fully supported |
| OpenType with CFF outlines             | `.otf`        | PostScript (`CFF` table)   | ✅ Fully supported |
| OpenType with CFF2 outlines (static)   | `.otf`        | PostScript (`CFF2` table)  | ✅ Fully supported |
| OpenType with CFF2 outlines (variable) | `.otf`        | Variable (`CFF2` + `fvar`) | ❌ Not supported   |
| Variable fonts                         | `.ttf`/`.otf` | Variable (`fvar` table)    | ❌ Not supported   |
```

#### Update "How to identify your font's format" section (lines 293-301)

Add after line 299:

```markdown
- **OTF with CFF2 outlines (static)**: Modern OpenType fonts may use CFF2 even without variation support. These are fully supported.
- **OTF with CFF2 outlines (variable)**: Variable fonts use CFF2 with blend operators for interpolation. These are not supported—use a static instance instead.
```

#### Update "Future Work" section (lines 303-307)

**Current:**

```markdown
## Future Work

- OpenType CFF2 font support
- Variable font support (fonts with `fvar` table)
```

**New:**

```markdown
## Future Work

- Variable font support (fonts with `fvar` table)
  - This includes variable CFF2 fonts with blend operators
```

---

## 3. Test Implementation

### 3.1 Create CFF2 Test Fixture

Create a script to generate the CFF2 fixture from the existing CFF fixture:

**File: `tests/fixtures/create_cff2_fixture.py`**

```python
#!/usr/bin/env python3
"""Generate CFF2 test fixture from CFF font.

This script converts the existing CFF fixture (CommitMono-Cosmix-700-Regular.otf)
to CFF2 format for testing CFF2 support.

Usage:
    python tests/fixtures/create_cff2_fixture.py
"""
from pathlib import Path

from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2
from fontTools.ttLib import TTFont

FIXTURES_DIR = Path(__file__).parent
INPUT_FONT = FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf"
OUTPUT_FONT = FIXTURES_DIR / "CommitMono-CFF2.otf"


def main() -> None:
    """Convert CFF font to CFF2."""
    if not INPUT_FONT.exists():
        print(f"Error: Input font not found: {INPUT_FONT}")
        return

    print(f"Loading {INPUT_FONT}...")
    font = TTFont(str(INPUT_FONT))

    # Verify it's CFF
    if "CFF " not in font:
        print("Error: Input font is not CFF format")
        return

    print("Converting CFF to CFF2...")
    convertCFFToCFF2(font)

    # Verify conversion
    if "CFF2" not in font:
        print("Error: Conversion failed - no CFF2 table")
        return

    print(f"Saving {OUTPUT_FONT}...")
    font.save(str(OUTPUT_FONT))
    font.close()

    print(f"Successfully created CFF2 fixture: {OUTPUT_FONT}")

    # Verify the output
    verify_font = TTFont(str(OUTPUT_FONT))
    assert "CFF2" in verify_font, "Output font missing CFF2 table"
    assert "CFF " not in verify_font, "Output font still has CFF table"
    verify_font.close()
    print("Verification passed!")


if __name__ == "__main__":
    main()
```

Run this once to create the fixture:

```bash
python tests/fixtures/create_cff2_fixture.py
```

### 3.2 Integration Tests

**File: `tests/integration/test_stencilization.py`**

Add after the existing CFF tests:

```python
# Add to imports
from fontTools.pens.recordingPen import RecordingPen

# Add fixture path constant
COMMIT_MONO_CFF2_PATH = FIXTURES_DIR / "CommitMono-CFF2.otf"


@pytest.fixture
def commit_mono_cff2_reader() -> Generator[FontReader, None, None]:
    """Load CommitMono CFF2 font for testing."""
    if not COMMIT_MONO_CFF2_PATH.exists():
        pytest.skip("CommitMono CFF2 font fixture not available")
    reader = FontReader(COMMIT_MONO_CFF2_PATH)
    reader.load()
    yield reader
    reader.close()


class TestCFF2FontProcessing:
    """Test CFF2 font processing."""

    def test_cff2_font_detected_as_opentype(
        self, commit_mono_cff2_reader: FontReader
    ) -> None:
        """Test that CFF2 font is detected as OpenType format."""
        font_format = commit_mono_cff2_reader.format
        assert font_format == "OpenType", f"Expected 'OpenType', got '{font_format}'"

    def test_cff2_font_has_cff2_table(
        self, commit_mono_cff2_reader: FontReader
    ) -> None:
        """Test that CFF2 font has CFF2 table, not CFF."""
        # Access internal font object for table check
        assert commit_mono_cff2_reader._font is not None
        assert "CFF2" in commit_mono_cff2_reader._font
        assert "CFF " not in commit_mono_cff2_reader._font

    def test_cff2_glyphs_can_be_read(
        self, commit_mono_cff2_reader: FontReader
    ) -> None:
        """Test that CFF2 glyphs can be converted to domain model."""
        # Test common glyphs with islands
        for glyph_name in ["O", "A", "B", "D", "P", "R", "Q"]:
            glyph = commit_mono_cff2_reader.get_glyph(glyph_name)
            if glyph is not None:
                assert glyph.name == glyph_name
                assert len(glyph.contours) > 0, f"Glyph {glyph_name} has no contours"

    def test_process_cff2_font_creates_valid_output(self) -> None:
        """Test that processing CFF2 creates a valid, loadable font."""
        if not COMMIT_MONO_CFF2_PATH.exists():
            pytest.skip("CommitMono CFF2 font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "CommitMono-Stenciled-CFF2.otf"

            stats = processor.process(
                font_path=COMMIT_MONO_CFF2_PATH,
                output_path=output_path,
                max_workers=1,  # Single worker for deterministic testing
            )

            # Verify processing succeeded
            assert stats.processed_count > 0, "No glyphs were processed"
            assert stats.bridges_added > 0, "No bridges were added"
            assert stats.error_count == 0, f"Errors occurred: {stats.error_count}"
            assert output_path.exists(), "Output file was not created"

            # Verify output is valid CFF2 font
            output_font = TTFont(str(output_path))
            assert "CFF2" in output_font, "Output font missing CFF2 table"
            assert "CFF " not in output_font, "Output font has unexpected CFF table"
            output_font.close()

    def test_cff2_processed_glyphs_can_be_drawn(self) -> None:
        """Test that processed CFF2 glyphs have valid, drawable charstrings."""
        if not COMMIT_MONO_CFF2_PATH.exists():
            pytest.skip("CommitMono CFF2 font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.otf"

            processor.process(
                font_path=COMMIT_MONO_CFF2_PATH,
                output_path=output_path,
                max_workers=1,
            )

            output_font = TTFont(str(output_path))
            glyph_set = output_font.getGlyphSet()

            # Draw several glyphs that typically have islands
            test_glyphs = ["O", "A", "B", "D", "P", "Q", "R", "zero", "four", "six", "eight", "nine"]
            drawn_count = 0

            for glyph_name in test_glyphs:
                if glyph_name in glyph_set:
                    pen = RecordingPen()
                    try:
                        glyph_set[glyph_name].draw(pen)
                        drawn_count += 1
                    except Exception as e:
                        pytest.fail(f"Failed to draw glyph {glyph_name}: {e}")

            assert drawn_count > 0, "No test glyphs could be drawn"
            output_font.close()

    def test_cff2_output_preserves_font_metadata(self) -> None:
        """Test that CFF2 processing preserves essential font metadata."""
        if not COMMIT_MONO_CFF2_PATH.exists():
            pytest.skip("CommitMono CFF2 font fixture not available")

        settings = StencilizerSettings()
        processor = FontProcessor(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.otf"

            processor.process(
                font_path=COMMIT_MONO_CFF2_PATH,
                output_path=output_path,
                max_workers=1,
            )

            input_font = TTFont(str(COMMIT_MONO_CFF2_PATH))
            output_font = TTFont(str(output_path))

            # Check UPM is preserved
            assert output_font["head"].unitsPerEm == input_font["head"].unitsPerEm

            # Check glyph count is preserved
            assert output_font["maxp"].numGlyphs == input_font["maxp"].numGlyphs

            input_font.close()
            output_font.close()


class TestVariableCFF2Detection:
    """Test detection of variable CFF2 fonts."""

    def test_static_cff2_not_detected_as_variable(
        self, commit_mono_cff2_reader: FontReader
    ) -> None:
        """Test that static CFF2 font is not detected as variable."""
        from stencilizer.io.converter import _is_variable_cff2

        assert commit_mono_cff2_reader._font is not None
        assert not _is_variable_cff2(commit_mono_cff2_reader._font)

    def test_variable_cff2_raises_not_implemented(self) -> None:
        """Test that variable CFF2 fonts raise NotImplementedError.

        Note: This test requires a variable CFF2 font fixture.
        We create a mock to simulate the detection.
        """
        from unittest.mock import MagicMock, patch

        from stencilizer.io.converter import domain_glyph_to_fonttools
        from stencilizer.domain.contour import Contour, Point
        from stencilizer.domain.glyph import Glyph, GlyphMetadata

        # Create a mock font that appears to be variable CFF2
        mock_font = MagicMock()
        mock_font.__contains__ = lambda self, key: key in ["CFF2", "fvar"]
        mock_font.__getitem__ = MagicMock()

        glyph = Glyph(
            metadata=GlyphMetadata("A", None, 500, 0),
            contours=[Contour(points=[Point(0, 0), Point(100, 0), Point(100, 100)])]
        )

        with pytest.raises(NotImplementedError) as exc_info:
            domain_glyph_to_fonttools(glyph, None, mock_font)

        assert "Variable CFF2" in str(exc_info.value)
```

### 3.3 Unit Tests

**File: `tests/unit/test_io.py`**

Add after the existing `TestCffGlyphUpdate` class:

```python
class TestCff2GlyphUpdate:
    """Tests for CFF2 glyph update functionality."""

    def test_update_cff2_glyph_uses_cff2_mode(self) -> None:
        """Test that _update_cff2_glyph creates charstring with CFF2=True."""
        from unittest.mock import MagicMock, Mock, patch

        from stencilizer.io.converter import _update_cff2_glyph
        from stencilizer.domain.contour import Contour, Point, PointType
        from stencilizer.domain.glyph import Glyph, GlyphMetadata

        # Create mock CFF2 font structure
        mock_font = MagicMock()
        mock_cff2_table = MagicMock()
        mock_top_dict = MagicMock()
        mock_charstrings = {}
        mock_fd_array = [MagicMock()]
        mock_fd_array[0].Private = MagicMock()
        mock_global_subrs = MagicMock()

        mock_cff2_table.cff.topDictIndex = [mock_top_dict]
        mock_top_dict.CharStrings = mock_charstrings
        mock_top_dict.FDArray = mock_fd_array
        mock_top_dict.FDSelect = None  # Simple font without FDSelect
        mock_cff2_table.cff.GlobalSubrs = mock_global_subrs

        mock_font.__getitem__ = Mock(return_value=mock_cff2_table)
        mock_font.getGlyphSet.return_value = {}

        glyph = Glyph(
            metadata=GlyphMetadata("A", None, 500, 0),
            contours=[
                Contour(points=[
                    Point(0, 0, PointType.ON_CURVE),
                    Point(100, 0, PointType.ON_CURVE),
                    Point(100, 100, PointType.ON_CURVE),
                ])
            ]
        )

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = MagicMock()
            mock_charstring = MagicMock()
            mock_pen.getCharString.return_value = mock_charstring
            mock_pen_class.return_value = mock_pen

            _update_cff2_glyph(glyph, None, mock_font)

            # Verify T2CharStringPen was called with CFF2=True and width=None
            mock_pen_class.assert_called_once()
            call_kwargs = mock_pen_class.call_args[1]
            assert call_kwargs.get("CFF2") is True, "CFF2 parameter should be True"
            assert call_kwargs.get("width") is None, "width should be None for CFF2"

    def test_update_cff2_glyph_stores_charstring(self) -> None:
        """Test that the updated charstring is stored in CharStrings dict."""
        from unittest.mock import MagicMock, Mock, patch

        from stencilizer.io.converter import _update_cff2_glyph
        from stencilizer.domain.contour import Contour, Point, PointType
        from stencilizer.domain.glyph import Glyph, GlyphMetadata

        mock_font = MagicMock()
        mock_cff2_table = MagicMock()
        mock_top_dict = MagicMock()
        mock_charstrings = {}  # Dict we'll verify gets updated
        mock_fd_array = [MagicMock()]
        mock_fd_array[0].Private = MagicMock()
        mock_global_subrs = MagicMock()

        mock_cff2_table.cff.topDictIndex = [mock_top_dict]
        mock_top_dict.CharStrings = mock_charstrings
        mock_top_dict.FDArray = mock_fd_array
        mock_top_dict.FDSelect = None
        mock_cff2_table.cff.GlobalSubrs = mock_global_subrs

        mock_font.__getitem__ = Mock(return_value=mock_cff2_table)
        mock_font.getGlyphSet.return_value = {}

        glyph = Glyph(
            metadata=GlyphMetadata("TestGlyph", None, 500, 0),
            contours=[]
        )

        with patch("stencilizer.io.converter.T2CharStringPen") as mock_pen_class:
            mock_pen = MagicMock()
            mock_charstring = MagicMock()
            mock_pen.getCharString.return_value = mock_charstring
            mock_pen_class.return_value = mock_pen

            _update_cff2_glyph(glyph, None, mock_font)

            # Verify charstring was stored
            assert "TestGlyph" in mock_charstrings
            assert mock_charstrings["TestGlyph"] is mock_charstring


class TestCff2PrivateDictLookup:
    """Tests for CFF2 Private dict lookup via FDSelect."""

    def test_get_private_with_fdselect(self) -> None:
        """Test Private dict lookup when FDSelect is present."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _get_cff2_private_for_glyph

        # Create mock with FDSelect
        mock_top_dict = MagicMock()
        mock_private_0 = MagicMock(name="Private0")
        mock_private_1 = MagicMock(name="Private1")

        mock_fd_array = [MagicMock(), MagicMock()]
        mock_fd_array[0].Private = mock_private_0
        mock_fd_array[1].Private = mock_private_1
        mock_top_dict.FDArray = mock_fd_array

        # FDSelect maps glyph indices to FD indices
        mock_fd_select = MagicMock()
        mock_fd_select.__getitem__ = Mock(side_effect=lambda idx: 1 if idx == 1 else 0)
        mock_top_dict.FDSelect = mock_fd_select

        # CharStrings with glyph order
        mock_charstrings = MagicMock()
        mock_charstrings.keys.return_value = ["A", "B", "C"]

        # Glyph "B" is at index 1, should get Private from FDArray[1]
        result = _get_cff2_private_for_glyph(mock_top_dict, "B", mock_charstrings)
        assert result is mock_private_1

    def test_get_private_without_fdselect(self) -> None:
        """Test Private dict lookup falls back to FDArray[0] without FDSelect."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _get_cff2_private_for_glyph

        mock_top_dict = MagicMock()
        mock_private_0 = MagicMock(name="Private0")

        mock_fd_array = [MagicMock()]
        mock_fd_array[0].Private = mock_private_0
        mock_top_dict.FDArray = mock_fd_array
        mock_top_dict.FDSelect = None

        mock_charstrings = MagicMock()

        result = _get_cff2_private_for_glyph(mock_top_dict, "A", mock_charstrings)
        assert result is mock_private_0


class TestVariableCff2Detection:
    """Tests for variable CFF2 detection."""

    def test_detects_fvar_table(self) -> None:
        """Test that fonts with fvar are detected as variable."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _is_variable_cff2

        mock_font = MagicMock()
        mock_font.__contains__ = lambda self, key: key in ["CFF2", "fvar"]

        assert _is_variable_cff2(mock_font) is True

    def test_detects_varstore(self) -> None:
        """Test that fonts with VarStore in TopDict are detected as variable."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _is_variable_cff2

        mock_font = MagicMock()
        mock_font.__contains__ = lambda self, key: key == "CFF2"

        mock_cff2 = MagicMock()
        mock_top_dict = MagicMock()
        mock_top_dict.VarStore = MagicMock()  # Has VarStore
        mock_cff2.cff.topDictIndex = [mock_top_dict]
        mock_font.__getitem__ = lambda self, key: mock_cff2 if key == "CFF2" else None

        assert _is_variable_cff2(mock_font) is True

    def test_static_cff2_not_variable(self) -> None:
        """Test that static CFF2 fonts are not detected as variable."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _is_variable_cff2

        mock_font = MagicMock()
        mock_font.__contains__ = lambda self, key: key == "CFF2"

        mock_cff2 = MagicMock()
        mock_top_dict = MagicMock()
        mock_top_dict.VarStore = None  # No VarStore
        mock_cff2.cff.topDictIndex = [mock_top_dict]
        mock_font.__getitem__ = lambda self, key: mock_cff2 if key == "CFF2" else None

        assert _is_variable_cff2(mock_font) is False

    def test_non_cff2_font_not_variable(self) -> None:
        """Test that non-CFF2 fonts return False."""
        from unittest.mock import MagicMock

        from stencilizer.io.converter import _is_variable_cff2

        mock_font = MagicMock()
        mock_font.__contains__ = lambda self, key: key == "glyf"

        assert _is_variable_cff2(mock_font) is False
```

---

## 4. Implementation Order

1. **Create CFF2 test fixture** (5 min)

   - Run `create_cff2_fixture.py` to generate `CommitMono-CFF2.otf`
   - Verify fixture loads correctly

2. **Implement `_is_variable_cff2()`** (10 min)

   - Add function to `converter.py`
   - Add unit tests

3. **Update winding normalization** (5 min)

   - Change `is_cff` to `is_cff_variant`
   - Include `"CFF2" in font` check

4. **Implement `_get_cff2_private_for_glyph()`** (15 min)

   - Add function to `converter.py`
   - Handle FDSelect and fallback cases
   - Add unit tests

5. **Implement `_update_cff2_glyph()`** (20 min)

   - Add function to `converter.py`
   - Use `T2CharStringPen(width=None, CFF2=True)`
   - Add unit tests

6. **Update write dispatcher** (5 min)

   - Add CFF2 branch with variable check
   - Update docstring

7. **Add integration tests** (20 min)

   - Add CFF2 test class
   - Test full processing pipeline

8. **Update documentation** (10 min)

   - Update README.md format table
   - Update "How to identify" section
   - Update "Future Work" section

9. **Run full test suite** (10 min)
   - Verify no regressions
   - Check coverage

---

## 5. Risk Assessment

| Risk                                   | Likelihood | Impact                       | Mitigation                               |
| -------------------------------------- | ---------- | ---------------------------- | ---------------------------------------- |
| Variable CFF2 font passed to processor | Medium     | High (corrupt output)        | Early detection with clear error message |
| Multi-FD font Private dict mismatch    | Low        | Medium (invalid charstrings) | Proper FDSelect handling with fallback   |
| Winding convention wrong for CFF2      | Very Low   | High (inverted glyphs)       | Same convention as CFF, already tested   |
| fonttools CFF2 API differences         | Low        | Medium (runtime errors)      | Test with real CFF2 fixture              |
| CFF2 fixture creation fails            | Low        | Medium (no tests)            | fonttools CFFToCFF2 is well-tested       |

---

## 6. Verification Checklist

After implementation, verify:

- [ ] Static CFF2 font loads without error
- [ ] CFF2 font detected as "OpenType" format
- [ ] CFF2 glyphs with islands are identified
- [ ] Bridges are correctly added to CFF2 glyphs
- [ ] Output font is valid CFF2 (loads in fonttools)
- [ ] Output glyphs can be drawn without error
- [ ] Variable CFF2 fonts raise clear error message
- [ ] All existing tests still pass (TrueType, CFF)
- [ ] Documentation updated
- [ ] No type errors from mypy
- [ ] No lint errors from ruff

---

## 7. Files Summary

| File                                       | Action | Changes                            |
| ------------------------------------------ | ------ | ---------------------------------- |
| `src/stencilizer/io/converter.py`          | Modify | Add 4 functions, update 2 existing |
| `src/stencilizer/io/reader.py`             | None   | Already handles CFF2               |
| `tests/fixtures/create_cff2_fixture.py`    | Create | Script to generate fixture         |
| `tests/fixtures/CommitMono-CFF2.otf`       | Create | Generated test fixture             |
| `tests/integration/test_stencilization.py` | Modify | Add CFF2 test classes              |
| `tests/unit/test_io.py`                    | Modify | Add CFF2 unit test classes         |
| `README.md`                                | Modify | Update format table and docs       |