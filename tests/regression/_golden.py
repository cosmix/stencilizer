"""Generate and compare golden outputs from the stable processing API."""

import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

from stencilizer.config.settings import BridgeConfig, StencilizerSettings
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.processor import FontProcessor, process_glyph
from stencilizer.domain import Glyph
from stencilizer.io.reader import FontReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FIXTURES: dict[str, Path] = {
    "roboto": FIXTURES_DIR / "Roboto-Regular.ttf",
    "lato": FIXTURES_DIR / "Lato-Black.ttf",
    "commitmono": FIXTURES_DIR / "CommitMono-Cosmix-700-Regular.otf",
}
GOLDEN_DIR = Path(__file__).parent / "golden"

SerializedGlyph = list[list[list[float | str]]]
_POINT_TYPE_NAMES = {1: "ON_CURVE", 2: "OFF_CURVE_QUAD", 3: "OFF_CURVE_CUBIC"}


def load_font_glyphs(font_key: str) -> tuple[int, dict[str, Glyph]]:
    """Read every glyph and the native units per em from a fixture font."""
    reader = FontReader(FIXTURES[font_key])
    reader.load()
    try:
        return reader.units_per_em, {
            glyph.to_dict()["metadata"]["name"]: glyph for glyph in reader.iter_glyphs()
        }
    finally:
        reader.close()


def _scale_bbox(value: Any, factor: float) -> Any:
    """Scale all coordinates in a bounding box while preserving its shape."""
    if isinstance(value, dict):
        return {key: _scale_bbox(item, factor) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_scale_bbox(item, factor) for item in value)
    if isinstance(value, list):
        return [_scale_bbox(item, factor) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value * factor
    return value


def scale_glyph(glyph: Glyph, factor: float) -> Glyph:
    """Return a separate glyph with all geometry and length metadata scaled."""
    data = glyph.to_dict()
    for contour in data["contours"]:
        for point in contour["points"]:
            point["x"] *= factor
            point["y"] *= factor

    metadata = data["metadata"]
    for key in ("advance_width", "lsb", "left_side_bearing"):
        if key in metadata:
            metadata[key] *= factor
    for key in ("bbox", "bounding_box", "bounds"):
        if key in metadata:
            metadata[key] = _scale_bbox(metadata[key], factor)
    for key in ("x_min", "y_min", "x_max", "y_max", "xmin", "ymin", "xmax", "ymax"):
        if key in metadata:
            metadata[key] *= factor
    return Glyph.from_dict(data)


def normalize_to_1000(glyph: Glyph, upm: int) -> Glyph:
    """Scale a glyph from its native UPM without rounding coordinates."""
    return scale_glyph(glyph, 1000 / upm)


def run_process_glyph(glyph: Glyph, upm: int, reference_stroke_width: float | None = None) -> Glyph:
    """Call the stable per-glyph API with default bridge settings."""
    result = process_glyph(
        glyph.to_dict(), BridgeConfig().model_dump(), upm, reference_stroke_width
    )
    if "error" in result:
        name = glyph.to_dict()["metadata"]["name"]
        raise AssertionError(f"{name}: {result['error']}\n{result.get('traceback', '')}")
    return Glyph.from_dict(result["glyph"])


def serialize_glyph(glyph: Glyph) -> SerializedGlyph:
    """Keep only ordered contour geometry and point type names."""
    data = glyph.to_dict()
    return [
        [[point["x"], point["y"], _POINT_TYPE_NAMES[point["type"]]] for point in contour["points"]]
        for contour in data["contours"]
    ]


def assert_glyphs_close(
    actual: Glyph,
    expected_serialized: SerializedGlyph,
    *,
    scale: float = 1.0,
    tol: float = 1e-6,
    context: str = "",
) -> None:
    """Compare contour structure, point types, and scaled coordinates."""
    actual_serialized = serialize_glyph(actual)
    if len(actual_serialized) != len(expected_serialized):
        raise AssertionError(
            f"{context} contour=-1 point=-1 contour count "
            f"actual={len(actual_serialized)} expected={len(expected_serialized)}"
        )

    limit = tol * max(1, scale)
    for contour_index, (contour, expected_contour) in enumerate(
        zip(actual_serialized, expected_serialized, strict=True)
    ):
        if len(contour) != len(expected_contour):
            raise AssertionError(
                f"{context} contour={contour_index} point=-1 point count "
                f"actual={len(contour)} expected={len(expected_contour)}"
            )
        for point_index, (point, expected_point) in enumerate(
            zip(contour, expected_contour, strict=True)
        ):
            expected_x = float(expected_point[0]) * scale
            expected_y = float(expected_point[1]) * scale
            expected_value = [expected_x, expected_y, expected_point[2]]
            if point[2] != expected_point[2]:
                raise AssertionError(
                    f"{context} contour={contour_index} point={point_index} type "
                    f"actual={point} expected={expected_value}"
                )
            if any(
                not abs(float(actual_coordinate) - expected_coordinate) <= limit
                for actual_coordinate, expected_coordinate in zip(
                    point[:2],
                    (expected_x, expected_y),
                    strict=True,
                )
            ):
                raise AssertionError(
                    f"{context} contour={contour_index} point={point_index} coordinates "
                    f"actual={point} expected={expected_value} tol={limit}"
                )


def _record_font(path: Path) -> dict[str, Any]:
    """Record every output glyph in the font's own glyph order."""
    font = TTFont(str(path))
    try:
        glyph_set = font.getGlyphSet()
        recordings: dict[str, Any] = {}
        for name in font.getGlyphOrder():
            pen = RecordingPen()
            glyph_set[name].draw(pen)
            recordings[name] = pen.value
        return recordings
    finally:
        font.close()


def _write_golden(path: Path, data: object) -> None:
    """Write stable JSON and gzip headers for reproducible golden files."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with (
        path.open("wb") as output,
        gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed,
    ):
        compressed.write(encoded)


def generate() -> None:
    """Capture normalized glyph behavior and the complete CommitMono pipeline."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = GlyphAnalyzer()
    for font_key in FIXTURES:
        upm, glyphs = load_font_glyphs(font_key)
        outputs: dict[str, SerializedGlyph] = {}
        for name, glyph in sorted(glyphs.items()):
            glyph_data = glyph.to_dict()
            if not glyph_data["contours"] or glyph_data["is_composite"]:
                continue
            if not analyzer.analyze(glyph).has_islands():
                continue
            normalized = normalize_to_1000(glyph, upm)
            outputs[name] = serialize_glyph(run_process_glyph(normalized, 1000, None))
        _write_golden(GOLDEN_DIR / f"{font_key}.json.gz", {"source_upm": upm, "glyphs": outputs})

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "out.otf"
        FontProcessor(StencilizerSettings()).process(FIXTURES["commitmono"], output_path)
        _write_golden(GOLDEN_DIR / "commitmono_pipeline.json.gz", _record_font(output_path))


if __name__ == "__main__":
    generate()
