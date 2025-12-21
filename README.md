# Stencilizer

Convert TrueType and OpenType fonts to stencil-ready versions by automatically adding bridges to enclosed contours.

## Overview

Stencilizer is a Python CLI tool that transforms regular fonts into stencil fonts by detecting "islands" (enclosed contours) in glyphs and adding bridges to connect them. This is essential for creating fonts suitable for stencil cutting, where disconnected parts would fall out.

Characters like **O**, **A**, **B**, **D**, **P**, **R**, **Q**, **4**, **6**, **8**, **9**, and **@** all have enclosed contours that need bridges to remain connected during cutting.

## Features

- **Automatic Island Detection**: Intelligently identifies enclosed contours using contour hierarchy analysis
- **Smart Bridge Placement**: Places bridges optimally based on contour geometry and user preferences
- **Parallel Processing**: Leverages multicore CPUs for fast processing of large fonts
- **Flexible Configuration**: Control bridge width, position, and count
- **Multiple Output Modes**:
  - Full processing (default)
  - Dry-run analysis
  - Island listing
- **Rich CLI Output**: Beautiful console output with progress tracking
- **Detailed Logging**: Optional file logging for debugging and analysis
- **Format Support**: Works with both TTF and OTF font formats

## Installation

### Using uv (recommended)

```bash
uv pip install stencilizer
```

### Using pip

```bash
pip install stencilizer
```

### From source

```bash
git clone https://github.com/yourusername/stencilizer.git
cd stencilizer
uv pip install -e .
```

## Quick Start

Convert a font with default settings:

```bash
stencilizer Roboto-Regular.ttf
```

This creates `Roboto-Regular-stenciled.ttf` in the same directory.

## Usage

### Basic Usage

```bash
# Convert with default settings
stencilizer input.ttf

# Specify output path
stencilizer input.ttf -o output.ttf

# Adjust bridge width (30-90% of stroke width)
stencilizer input.ttf --bridge-width 70

# Set minimum bridges per island
stencilizer input.ttf --min-bridges 2
```

### Bridge Position Control

Control where bridges are placed:

```bash
# Automatic placement (default)
stencilizer input.ttf --position auto

# Prefer top bridges
stencilizer input.ttf --position top

# Prefer both top and bottom
stencilizer input.ttf --position top_bottom

# Other options: bottom, left, right
```

### Analysis Modes

```bash
# List all glyphs with islands
stencilizer input.ttf --list-islands

# Dry run (analyze without modifying)
stencilizer input.ttf --dry-run

# Verbose output
stencilizer input.ttf --verbose

# Quiet mode
stencilizer input.ttf --quiet
```

### Performance Options

```bash
# Control parallel workers
stencilizer input.ttf --workers 4

# Use all available cores (default)
stencilizer input.ttf
```

### Logging

```bash
# Enable file logging
stencilizer input.ttf --log-file stencilizer.log

# Set log level
stencilizer input.ttf --log-level DEBUG
```

## Configuration

Stencilizer can be configured via environment variables with the `STENCILIZER_` prefix:

```bash
# Bridge configuration
export STENCILIZER_BRIDGE__WIDTH_PERCENT=70.0
export STENCILIZER_BRIDGE__MIN_BRIDGES=2
export STENCILIZER_BRIDGE__POSITION_PREFERENCE=top

# Processing configuration
export STENCILIZER_PROCESSING__MAX_WORKERS=4
export STENCILIZER_PROCESSING__SKIP_COMPOSITE=true

# Logging configuration
export STENCILIZER_LOGGING__LOG_LEVEL=INFO
```

## How It Works

### 1. Glyph Analysis

Stencilizer analyzes each glyph to identify its contour hierarchy:

- **Outer contours**: Main glyph shapes (counter-clockwise winding)
- **Inner contours**: Enclosed areas (clockwise winding)
- **Islands**: Inner contours that are fully enclosed and need bridges

### 2. Bridge Placement

For each island, the algorithm:

1. Samples candidate points around the island perimeter
2. Calculates distances to the containing outer contour
3. Evaluates positions based on:
   - User-specified position preference
   - Geometric stability
   - Aesthetic considerations
4. Places the minimum required number of bridges

### 3. Glyph Transformation

Bridges are added by cutting notches into both the island and outer contour, creating connection points while preserving the overall glyph structure.

### 4. Parallel Processing

Glyphs are processed in parallel using Python's ProcessPoolExecutor, enabling efficient utilization of multi-core systems.

## Architecture

```text
stencilizer/
├── cli/              # CLI interface and output formatting
├── config/           # Configuration and settings
├── core/             # Core processing algorithms
│   ├── analyzer.py   # Island detection and hierarchy analysis
│   ├── bridge.py     # Bridge placement and generation
│   ├── geometry.py   # Geometric utilities
│   ├── processor.py  # Parallel processing orchestration
│   └── surgery.py    # Glyph transformation
├── domain/           # Domain models (Glyph, Contour, Bridge)
├── io/               # Font reading and writing
└── utils/            # Logging and utilities
```

## Examples

### Convert with custom bridge settings

```bash
stencilizer Roboto-Regular.ttf \
  --bridge-width 80 \
  --min-bridges 2 \
  --position top_bottom \
  -o Roboto-Stencil.ttf
```

### Analyze before processing

```bash
# Check which glyphs have islands
stencilizer Roboto-Regular.ttf --list-islands

# See what would be done
stencilizer Roboto-Regular.ttf --dry-run --verbose
```

### Process with detailed logging

```bash
stencilizer Roboto-Regular.ttf \
  --log-file processing.log \
  --log-level DEBUG \
  --verbose
```

## Requirements

- Python 3.11 or higher
- fonttools >= 4.47.0
- pydantic >= 2.5.0
- rich >= 13.7.0
- structlog >= 24.1.0
- typer >= 0.9.0

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/stencilizer.git
cd stencilizer

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stencilizer --cov-report=html

# Run specific test modules
pytest tests/unit/test_analyzer.py
```

### Code Quality

```bash
# Type checking
mypy src/stencilizer tests

# Linting and formatting
ruff check src tests
ruff format src tests
```

## Troubleshooting

### Font not loading

Ensure your font file is a valid TTF or OTF file and not corrupted.

### No islands found

Some fonts may not have enclosed contours. Use `--list-islands` to check which glyphs have islands.

### Bridge width too wide/narrow

Adjust the `--bridge-width` parameter (range: 30-90% of stroke width). Default is 60%.

### Processing errors

Enable detailed logging to diagnose issues:

```bash
stencilizer input.ttf --log-file debug.log --log-level DEBUG
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Dimosthenis Kaponis <dkaponis@gmail.com>

## Version

0.1.0
