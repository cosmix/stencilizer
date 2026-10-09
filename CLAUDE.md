# Stencilizer

Python CLI tool that converts fonts to stencil-ready versions by adding bridges to enclosed contours (islands).

## Architecture Overview

```text
CLI (typer)
    └── FontProcessor (parallel orchestration)
            ├── FontReader (fonttools TTFont)
            │       └── converter.fonttools_glyph_to_domain()
            ├── GlyphAnalyzer (contour hierarchy, island detection)
            ├── GlyphTransformer (contour surgery)
            └── FontWriter
                    └── converter.domain_glyph_to_fonttools()
```

## Key Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| CLI | `src/stencilizer/cli/app.py` | Entry point, Typer commands |
| Config | `src/stencilizer/config/settings.py` | Pydantic settings models |
| Domain | `src/stencilizer/domain/` | `Glyph`, `Contour`, `Point` models |
| I/O | `src/stencilizer/io/` | `FontReader`, `FontWriter`, format converters |
| Core | `src/stencilizer/core/` | Analyzer, geometry, bridge placement, surgery |
| Exceptions | `src/stencilizer/exceptions.py` | Exception hierarchy |

## Font Format Handling

### Supported Formats

- **TrueType** (`.ttf`): `glyf` table, quadratic curves
- **OpenType/TrueType** (`.otf`): `glyf` table in OT container
- **OpenType/CFF** (`.otf`): `CFF ` table, cubic curves
- **OpenType/CFF2** (`.otf`): `CFF2` table, static fonts only

### Not Supported

- Variable fonts (`fvar` table)
- Variable CFF2 (blend operators)

### Format Detection

```python
# reader.py
if "CFF " in font or "CFF2" in font:
    return "OpenType"
return "TrueType"
```

### Winding Convention Normalization

CFF/CFF2 use opposite winding from TrueType:
- TrueType: CCW=outer, CW=inner
- CFF/CFF2: CW=outer, CCW=inner

Normalization happens in `converter.py`:
- **Read**: Reverse CFF/CFF2 contour points to match TrueType convention
- **Write**: Reverse points back to restore CFF convention

## Key Files for Font I/O

### `src/stencilizer/io/converter.py`

Core conversion between fonttools and domain models:

- `fonttools_glyph_to_domain()`: Read any format via `RecordingPen`
- `domain_glyph_to_fonttools()`: Dispatch to format-specific writers
- `_update_truetype_glyph()`: Write with `TTGlyphPen`
- `_update_cff_glyph()`: Write with `T2CharStringPen`
- `_update_cff2_glyph()`: Write with `T2CharStringPen(CFF2=True)`

### `src/stencilizer/io/reader.py`

Font loading via fonttools `TTFont`. Provides `iter_glyphs()` iterator.

### `src/stencilizer/io/writer.py`

Font saving with name table updates (adds "Stenciled" suffix).

## Domain Models

### Point Types

```python
class PointType(Enum):
    ON_CURVE = auto()         # On-curve point
    OFF_CURVE_QUAD = auto()   # TrueType quadratic control point
    OFF_CURVE_CUBIC = auto()  # CFF cubic control point
```

### Contour

Points list with winding direction. Normalized to TrueType convention.

### Glyph

Contours + metadata (name, unicode, advance_width, lsb).

## Testing

### Fixtures

- `tests/fixtures/Roboto-Regular.ttf` - TrueType
- `tests/fixtures/Lato-Black.ttf` - TrueType
- `tests/fixtures/CommitMono-Cosmix-700-Regular.otf` - CFF

### Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - Full pipeline tests with real fonts

## Development

### Commands

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/stencilizer tests

# Linting
ruff check src tests
ruff format src tests
```

### Dependencies

- `fonttools>=4.65.0` - Font parsing
- `pydantic>=2.13.5` - Data validation
- `rich>=15.0.0` - CLI output
- `structlog>=26.1.0` - Logging
- `typer>=0.27.2` - CLI framework

## Conventions

- All contours normalized to TrueType winding convention internally
- Parallel processing via `ProcessPoolExecutor` with picklable functions
- Domain models support dict serialization for IPC
- Bridge width calculated as a percentage of a reference stroke of 10% of the font's UPM
