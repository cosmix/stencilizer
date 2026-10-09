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
            ├── variable/ (fonts with an fvar table: surgery replayed on every master)
            └── FontWriter
                    ├── converter.domain_glyph_to_fonttools()
                    └── variable.write_gvar / variable.write_cff2
```

## Key Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| CLI | `src/stencilizer/cli/app.py` | Entry point, Typer commands |
| Config | `src/stencilizer/config/settings.py` | Pydantic settings models |
| Domain | `src/stencilizer/domain/` | `Glyph`, `Contour`, `Point` models |
| I/O | `src/stencilizer/io/` | `FontReader`, `FontWriter`, format converters, `--instance` pinning |
| Core | `src/stencilizer/core/` | Analyzer, geometry, bridge placement, surgery |
| Variable | `src/stencilizer/variable/` | Variable-font engine: reader, solver, replay, `gvar` and CFF2 writers |
| Exceptions | `src/stencilizer/exceptions.py` | Exception hierarchy |

## Font Format Handling

### Supported Formats

- **TrueType** (`.ttf`): `glyf` table, quadratic curves
- **OpenType/TrueType** (`.otf`): `glyf` table in OT container
- **OpenType/CFF** (`.otf`): `CFF ` table, cubic curves
- **OpenType/CFF2** (`.otf`): `CFF2` table, static fonts
- **Variable TrueType** (`fvar` + `glyf`): the default outline is stenciled and the surgery is
  replayed on a master at every `gvar` support peak; `glyf` and `gvar` tuples are rewritten
- **Variable CFF2** (`fvar` + `CFF2`): same replay; charstrings are rewritten with `blend` operands

### Not Supported

- `fvar` with `CFF ` outlines: the font loads, but saving fails with "unsupported variable
  outline format" once a glyph needs writing
- Glyphs whose variation data the engine cannot use (for example several `vsindex` values in one
  CFF2 charstring): left unchanged, counted as skipped, with their islands counted as unbridged

### Format Detection

```python
# reader.py (FontReader.format)
if "CFF " in font or "CFF2" in font:
    return "OpenType"
return "TrueType"

# variable/reader.py
is_variable(font)  # "fvar" in font; FontProcessor then runs variable/processing.py
```

`FontWriter.update_glyph` rejects fonts with `fvar`; they go through `update_variable_glyph`,
which dispatches on `glyf` (`write_gvar`) or `CFF2` (`write_cff2`).

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

Font loading via fonttools `TTFont`. Provides `iter_glyphs()` iterator and `unicode_by_name`,
the glyph-name to code-point map built once per loaded font.

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

### VariableGlyph

`variable/model.py`: the default `Glyph`, the variation `Support` regions, and one full master
`Glyph` per support.

## Testing

### Fixtures

- `tests/fixtures/Roboto-Regular.ttf` - TrueType
- `tests/fixtures/Lato-Black.ttf` - TrueType
- `tests/fixtures/CommitMono-Cosmix-700-Regular.otf` - CFF
- `tests/fixtures/variable/` - variable subsets: Ubuntu and Inter (`gvar`), Cantarell (CFF2)

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
- Parallel processing via `ProcessPoolExecutor` with a spawn context and picklable functions;
  entry points call `multiprocessing.freeze_support()` and scripts guard `if __name__ == "__main__"`
- Domain models support dict serialization for IPC
- Bridge width calculated as a percentage of a reference stroke of 10% of the font's UPM
