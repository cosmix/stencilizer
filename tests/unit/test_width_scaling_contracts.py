"""Contracts for variable-font bridge width scaling (stage width-scaling).

The engine contracts stencil small synthetic glyphs, a square ring around a square
counter at UPM 1000, and measure each bridge's gap at a normalized location. The CLI
contracts run the Typer app on the variable and static fixture fonts.
"""

from pathlib import Path
from typing import Any

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
from fontTools.ttLib.removeOverlaps import removeOverlaps  # type: ignore[import-untyped]
from typer.testing import CliRunner, Result

from stencilizer.cli.app import app
from stencilizer.config import BridgeConfig, BridgeWidthScaling, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.domain.glyph import Glyph
from stencilizer.io import FontReader
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.variable.holes import enclosed_counters
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.transform import VariableOutcome, transform_variable_glyph
from tests.font_helpers import CANTARELL, INTER, ROBOTO, glyph_at, island_count, units_per_em
from tests.unit._variable_cases import Outline, glyph_from_outlines

UPM = 1000
BOLD = Support((("wght", 0.0, 1.0, 1.0),))
THIN = Support((("wght", -1.0, -1.0, 0.0),))
BLACK_AT: dict[str, float] = {"wght": 1.0}
THIN_AT: dict[str, float] = {"wght": -1.0}
TOLERANCE = 1.0
PROPORTIONAL = BridgeWidthScaling.PROPORTIONAL
FIXED = BridgeWidthScaling.FIXED
STATIC_WARNING = "Width scaling applies only to variable fonts; using fixed width."
CFF2_WARNING = (
    "Proportional width scaling with --instance is not supported for CFF2 fonts; "
    "pinning first with fixed width."
)


def _outer(width: float, height: float) -> Outline:
    """A clockwise rectangle from the origin, wound as ``_variable_cases.STEM``."""
    return [(0.0, 0.0), (0.0, height), (width, height), (width, 0.0)]


def _counter(x0: float, y0: float, x1: float, y1: float) -> Outline:
    """A counter-clockwise rectangle."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _variable(
    default: list[Outline], masters: list[tuple[Support, list[Outline]]]
) -> VariableGlyph:
    return VariableGlyph(
        glyph_from_outlines(*default),
        tuple(support for support, _ in masters),
        tuple(glyph_from_outlines(*outlines) for _, outlines in masters),
        ("wght",),
    )


def _ring() -> VariableGlyph:
    return _variable(
        [_outer(1000, 1000), _counter(200, 200, 800, 800)],
        [
            (BOLD, [_outer(1000, 1000), _counter(300, 300, 700, 700)]),
            (THIN, [_outer(1000, 1000), _counter(50, 50, 950, 950)]),
        ],
    )


def _bold_counter(x0: float, x1: float) -> VariableGlyph:
    """The ring's default with one bold master whose counter spans x0..x1, y 300..700."""
    return _variable(
        [_outer(1000, 1000), _counter(200, 200, 800, 800)],
        [(BOLD, [_outer(1000, 1000), _counter(x0, 300, x1, 700)])],
    )


def _transform(
    vg: VariableGlyph,
    direction: BridgeDirection = BridgeDirection.VERTICAL,
    **scaling: Any,
) -> VariableOutcome:
    bridge = BridgeConfig(direction=direction, width_percent=60, **scaling)
    return transform_variable_glyph(vg, bridge, GeometryConfig(), UPM)


def _gap(glyph: Glyph, lo: float, hi: float, axis: str = "x") -> float:
    """Spread of the distinct point coordinates strictly inside lo..hi (by 0.5 unit)."""
    values = {
        getattr(point, axis)
        for contour in glyph.contours
        for point in contour.points
        if lo + 0.5 < getattr(point, axis) < hi - 0.5
    }
    assert values, (lo, hi, axis)
    return float(max(values) - min(values))


def _assert_gap(
    outcome: VariableOutcome, location: dict[str, float], span: tuple[float, float], expected: float
) -> None:
    gap = _gap(outcome.glyph.instance(location), *span)
    assert abs(gap - expected) <= TOLERANCE, (location, gap, expected)


def test_proportional_gap_follows_bold_stroke() -> None:
    outcome = _transform(_ring(), width_scaling=PROPORTIONAL, scaling_strength=100)
    assert outcome.bridge_count >= 1
    _assert_gap(outcome, {}, (200, 800), 60.0)
    _assert_gap(outcome, BLACK_AT, (300, 700), 90.0)


def test_fixed_mode_keeps_default_gap() -> None:
    outcome = _transform(_ring(), width_scaling=FIXED)
    assert outcome.bridge_count >= 1
    _assert_gap(outcome, BLACK_AT, (300, 700), 60.0)
    _assert_gap(outcome, THIN_AT, (50, 950), 60.0)


def test_minimum_width_clamps_thin_master() -> None:
    outcome = _transform(
        _ring(), width_scaling=PROPORTIONAL, scaling_strength=100, min_width_percent=40
    )
    assert outcome.bridge_count >= 1
    _assert_gap(outcome, THIN_AT, (50, 950), 40.0)


def test_scaling_strength_softens_gap() -> None:
    outcome = _transform(_ring(), width_scaling=PROPORTIONAL, scaling_strength=50)
    assert outcome.bridge_count >= 1
    _assert_gap(outcome, BLACK_AT, (300, 700), 60.0 * 1.5**0.5)


def test_proportional_falls_back_to_fixed() -> None:
    outcome = _transform(_bold_counter(465, 535), width_scaling=PROPORTIONAL)
    assert outcome.bridge_count >= 1
    _assert_gap(outcome, BLACK_AT, (465, 535), 60.0)


def test_fixed_falls_back_to_mean_targets() -> None:
    outcome = _transform(_bold_counter(475, 525), width_scaling=FIXED)
    assert outcome.bridge_count >= 1
    assert _gap(outcome.glyph.instance(BLACK_AT), 475, 525) < 50.0


def test_scaled_steps_fall_back_to_mean_targets() -> None:
    outcome = _transform(_bold_counter(475, 525), width_scaling=PROPORTIONAL)
    assert outcome.bridge_count >= 1
    assert _gap(outcome.glyph.instance(BLACK_AT), 475, 525) < 50.0


def _two_counters_vertical() -> VariableGlyph:
    return _variable(
        [_outer(1600, 1000), _counter(200, 200, 600, 800), _counter(1000, 200, 1400, 800)],
        [
            (
                BOLD,
                [_outer(1600, 1000), _counter(200, 300, 600, 700), _counter(1000, 200, 1400, 800)],
            )
        ],
    )


def _two_counters_horizontal() -> VariableGlyph:
    return _variable(
        [_outer(1000, 1600), _counter(200, 200, 800, 600), _counter(200, 1000, 800, 1400)],
        [
            (
                BOLD,
                [_outer(1000, 1600), _counter(300, 200, 700, 600), _counter(200, 1000, 800, 1400)],
            )
        ],
    )


def test_per_bridge_ratios_are_independent() -> None:
    cases = (
        (_two_counters_vertical(), BridgeDirection.VERTICAL, "x"),
        (_two_counters_horizontal(), BridgeDirection.HORIZONTAL, "y"),
    )
    for vg, direction, axis in cases:
        outcome = _transform(vg, direction, width_scaling=PROPORTIONAL, scaling_strength=100)
        assert outcome.bridge_count == 2, direction
        expected: list[tuple[dict[str, float], tuple[float, float], float]] = [
            ({}, (200, 600), 60.0),
            ({}, (1000, 1400), 60.0),
            (BLACK_AT, (200, 600), 90.0),
            (BLACK_AT, (1000, 1400), 60.0),
        ]
        for location, span, want in expected:
            gap = _gap(outcome.glyph.instance(location), *span, axis=axis)
            assert abs(gap - want) <= TOLERANCE, (direction, location, span, gap, want)


def test_proportional_only_success_is_kept() -> None:
    wide = _variable(
        [_outer(1000, 1000), _counter(200, 200, 800, 800)],
        [(BOLD, [_outer(6000, 1000), _counter(2980, 50, 3020, 950)])],
    )
    proportional = _transform(wide, width_scaling=PROPORTIONAL)
    assert proportional.bridge_count == 1
    _assert_gap(proportional, BLACK_AT, (2980, 3020), 30.0)
    fixed = _transform(wide, width_scaling=FIXED)
    assert fixed.glyph is wide
    assert (fixed.bridge_count, fixed.unbridged_count) == (0, 1)


# CLI contracts


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")


def _flat(text: str) -> str:
    """Drop whitespace so assertions survive the console folding long lines."""
    return "".join(text.split())


def _run(tmp_path: Path, font: Path, name: str, *args: str) -> tuple[Result, Path]:
    """Run the CLI on ``font`` writing ``name`` under tmp_path; assert a clean exit."""
    out = tmp_path / f"{name}{font.suffix}"
    log = tmp_path / f"{name}.log"
    result = CliRunner().invoke(app, [str(font), "-o", str(out), "--log-file", str(log), *args])
    assert result.exit_code == 0, (name, result.output)
    return result, out


def _outlines(path: Path, location: dict[str, float]) -> dict[str, dict[str, Any]]:
    font = TTFont(path)
    return {name: glyph_at(font, name, location).to_dict() for name in font.getGlyphOrder()}


def test_cli_proportional_changes_masters_only(tmp_path: Path) -> None:
    _, fixed = _run(tmp_path, INTER, "fixed", "-q")
    _, proportional = _run(tmp_path, INTER, "proportional", "-q", "--width-scaling", "proportional")
    source = _outlines(INTER, {})
    fixed_default, proportional_default = _outlines(fixed, {}), _outlines(proportional, {})
    for name, outline in fixed_default.items():
        if source[name] in (outline, proportional_default[name]):
            continue
        assert proportional_default[name] == outline, name
    assert _outlines(fixed, BLACK_AT) != _outlines(proportional, BLACK_AT)


def test_cli_scaling_options_reach_config(tmp_path: Path) -> None:
    _, fixed = _run(tmp_path, INTER, "fixed", "-q")
    _, strength = _run(
        tmp_path,
        INTER,
        "strength",
        "-q",
        "--width-scaling",
        "proportional",
        "--scaling-strength",
        "0",
    )
    _, minimum = _run(
        tmp_path,
        INTER,
        "minimum",
        "-q",
        "--width-scaling",
        "proportional",
        "--min-bridge-width",
        "110",
    )
    for location in ({}, BLACK_AT, THIN_AT):
        assert _outlines(strength, location) == _outlines(fixed, location), location
    assert _outlines(minimum, THIN_AT) == _outlines(fixed, THIN_AT)
    assert _outlines(minimum, BLACK_AT) != _outlines(fixed, BLACK_AT)


def _merged_glyph(path: Path, name: str) -> tuple[Glyph, int]:
    """Glyph ``name`` of a static font after fontTools overlap removal, and the font's UPM."""
    font = TTFont(path)
    removeOverlaps(font, [name])
    glyph = fonttools_glyph_to_domain(name, font.getGlyphSet()[name], font)
    return glyph, units_per_em(font)


def test_cli_instance_proportional_stencils_first(tmp_path: Path) -> None:
    pin = ("-q", "--instance", "wght=900")
    _, fixed = _run(tmp_path, INTER, "fixed", *pin)
    _, proportional = _run(tmp_path, INTER, "proportional", *pin, "--width-scaling", "proportional")
    fixed_font, proportional_font = TTFont(fixed), TTFont(proportional)
    assert "fvar" not in fixed_font
    assert "fvar" not in proportional_font
    for name in fixed_font.getGlyphOrder():
        fixed_advance = fixed_font["hmtx"].metrics[name][0]
        assert proportional_font["hmtx"].metrics[name][0] == fixed_advance, name
    assert glyph_at(proportional_font, "o", {}).to_dict() != glyph_at(fixed_font, "o", {}).to_dict()
    ampersand, upm = _merged_glyph(proportional, "ampersand")
    assert island_count(ampersand, upm) == 0
    assert proportional_font["name"].getDebugName(1) == fixed_font["name"].getDebugName(1)


def test_cli_instance_proportional_open_off_master(tmp_path: Path) -> None:
    _, out = _run(
        tmp_path,
        INTER,
        "off-master",
        "-q",
        "--instance",
        "wght=550",
        "--width-scaling",
        "proportional",
    )
    with FontReader(out) as reader:
        upm = reader.units_per_em
        glyphs = list(reader.iter_glyphs())
    assert glyphs
    for glyph in glyphs:
        if glyph.name == "ampersand":
            continue
        assert enclosed_counters(glyph, upm) == 0, glyph.name
    ampersand, merged_upm = _merged_glyph(out, "ampersand")
    assert island_count(ampersand, merged_upm) == 0


def test_cli_instance_cff2_pins_first(tmp_path: Path) -> None:
    pin = ("--instance", "wght=250")
    _, fixed = _run(tmp_path, CANTARELL, "fixed", "-q", *pin)
    result, proportional = _run(
        tmp_path, CANTARELL, "proportional", *pin, "--width-scaling", "proportional"
    )
    assert _flat(CFF2_WARNING) in _flat(result.output)
    assert _outlines(proportional, {}) == _outlines(fixed, {})


def test_cli_static_font_warns(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ROBOTO, "roboto", "--width-scaling", "proportional")
    assert _flat(STATIC_WARNING) in _flat(result.output)
    dry = CliRunner().invoke(app, [str(ROBOTO), "--dry-run", "--width-scaling", "proportional"])
    assert dry.exit_code == 0, dry.output
    assert _flat(STATIC_WARNING) in _flat(dry.output)
    assert "Width scaling         fixed" in dry.output
    result, _ = _run(tmp_path, INTER, "inter", "--width-scaling", "proportional")
    assert _flat(STATIC_WARNING) not in _flat(result.output)
    pinned = CliRunner().invoke(
        app, [str(INTER), "--instance", "wght=900", "--dry-run", "--width-scaling", "proportional"]
    )
    assert pinned.exit_code == 0, pinned.output
    assert _flat(STATIC_WARNING) not in _flat(pinned.output)
