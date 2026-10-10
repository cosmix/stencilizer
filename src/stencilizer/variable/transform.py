"""Stencil a variable glyph: surgery on the default master, replayed on every master."""

import time
import traceback
from dataclasses import dataclass, replace
from typing import Any

from stencilizer.config import BridgeConfig, BridgeWidthScaling, GeometryConfig
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.surgery import GlyphTransformer
from stencilizer.domain.glyph import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable.align import align_to_lines, snap_distance
from stencilizer.variable.bridge_width import fallback_steps, width_rule
from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.replay import map_surgery, replay
from stencilizer.variable.rounding import round_variable_glyph
from stencilizer.variable.validate import validate


@dataclass(frozen=True)
class VariableOutcome:
    """The glyph to write and the default-master surgery counts behind it."""

    glyph: VariableGlyph
    bridge_count: int
    unbridged_count: int


def _island_count(glyph: Glyph, upm: int) -> int:
    return len(GlyphAnalyzer().analyze(glyph, upm).get_islands())


def _bridged(
    vg: VariableGlyph,
    merged: VariableGlyph,
    bridge: BridgeConfig,
    geometry: GeometryConfig,
    upm: int,
    *,
    step: BridgeWidthScaling | None = None,
) -> VariableOutcome | None:
    """Default surgery replayed on every master, rounded and validated; None on failure.

    ``step`` is the width mode that sizes each bridge's gap in every master; None keeps
    every bridge line at the mean of its cut points, unpaired.
    """
    transformer = GlyphTransformer(
        analyzer=GlyphAnalyzer(), bridge_config=bridge, geometry_config=geometry
    )
    outcome = transformer.transform_with_outcome(merged.default, upm=upm)
    if outcome.bridge_count == 0:
        return VariableOutcome(vg, 0, outcome.unbridged_count)
    snap = snap_distance(geometry, upm)
    smap = map_surgery(merged.default, outcome.glyph, snap)
    if smap is None:
        return None
    default = align_to_lines(smap, merged.default, outcome.glyph, snap=snap)
    if default is None:
        return None
    if step is not None:
        scaling = bridge.model_copy(update={"width_scaling": step})
        smap = replace(smap, widths=width_rule(smap, merged.default, default, scaling, upm))
    replayed: list[Glyph] = []
    for master in merged.masters:
        glyph = replay(smap, merged.default, default, master)
        if glyph is None:
            return None
        replayed.append(glyph)
    result = round_variable_glyph(
        VariableGlyph(default, merged.supports, tuple(replayed), vg.axis_tags, vg.cff2)
    )
    if not validate(result, upm, allowed_islands=outcome.unbridged_count, lines=smap.lines):
        return None
    return VariableOutcome(result, outcome.bridge_count, outcome.unbridged_count)


def _first_bridged(
    vg: VariableGlyph,
    merged: VariableGlyph,
    bridge: BridgeConfig,
    geometry: GeometryConfig,
    upm: int,
) -> VariableOutcome | None:
    """The outcome of the first fallback step that succeeds, or None when every one fails."""
    for step in fallback_steps(bridge):
        outcome = _bridged(vg, merged, bridge, geometry, upm, step=step)
        if outcome is not None:
            return outcome
    return None


def transform_variable_glyph(
    vg: VariableGlyph, bridge: BridgeConfig, geometry: GeometryConfig, upm: int
) -> VariableOutcome:
    """Bridge ``vg`` in every master, or return it unchanged with its islands counted.

    The replay tries ``bridge``'s width mode, then fixed width, then per-line mean
    targets; the first that bridges every master wins. The unchanged outcome counts the
    islands of the deepest stage reached: the overlap-merged default, else the flattened
    default, else the input default. A ``VariationDataError`` from any step gives the
    unchanged outcome; it never escapes.
    """
    if vg.default.is_empty():
        return VariableOutcome(vg, 0, 0)
    counted = vg.default
    try:
        flat = flatten_compatible(vg, upm)
        counted = flat.default
        merged = remove_overlaps_compatible(flat)
        if merged is None:
            return VariableOutcome(vg, 0, _island_count(counted, upm))
        counted = merged.default
        islands = _island_count(merged.default, upm)
        if islands == 0:
            return VariableOutcome(vg, 0, 0)
        outcome = _first_bridged(vg, merged, bridge, geometry, upm)
        return outcome if outcome is not None else VariableOutcome(vg, 0, islands)
    except VariationDataError:
        return VariableOutcome(vg, 0, _island_count(counted, upm))


def process_variable_glyph(
    vg_dict: dict[str, Any],
    config_dict: dict[str, Any],
    upm: int,
    geometry_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a serialized variable glyph in a worker; keys match ``process_glyph``'s."""
    start_time = time.time()
    try:
        geometry = (
            GeometryConfig(**geometry_dict) if geometry_dict is not None else GeometryConfig()
        )
        outcome = transform_variable_glyph(
            VariableGlyph.from_dict(vg_dict), BridgeConfig(**config_dict), geometry, upm
        )
        return {
            "glyph": outcome.glyph.to_dict(),
            "bridges_added": outcome.bridge_count,
            "unbridged_count": outcome.unbridged_count,
            "duration_ms": (time.time() - start_time) * 1000,
        }
    except Exception as error:
        return {
            "error": str(error),
            "glyph_name": vg_dict.get("default", {}).get("metadata", {}).get("name", "unknown"),
            "traceback": traceback.format_exc(),
            "duration_ms": (time.time() - start_time) * 1000,
        }
