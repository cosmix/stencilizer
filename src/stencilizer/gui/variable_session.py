"""Qt-free variable-font state for GUI sessions: axes, location mapping, outcome cache."""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fontTools.varLib.models import (  # type: ignore[import-untyped]
    normalizeLocation,
    piecewiseLinearMap,
)

from stencilizer.config import BridgeConfig, GeometryConfig
from stencilizer.config.settings import BridgeDirection
from stencilizer.domain import Glyph
from stencilizer.exceptions import VariationDataError
from stencilizer.variable.model import VariableGlyph
from stencilizer.variable.processing import UNSUPPORTED_REASON
from stencilizer.variable.transform import VariableOutcome, transform_variable_glyph

logger = logging.getLogger(__name__)

OUTCOME_CACHE_ENTRIES = 64

# (glyph name, bridge JSON, geometry JSON, effective direction)
OutcomeKey = tuple[str, str, str, BridgeDirection]
AvarSegments = dict[str, dict[float, float]]


@dataclass(frozen=True)
class AxisInfo:
    """One fvar axis in user-space units."""

    tag: str
    name: str
    minimum: float
    default: float
    maximum: float


def read_axes(font: Any) -> tuple[AxisInfo, ...]:
    """Return the fvar axes of ``font``; the name falls back to the tag."""
    if "fvar" not in font:
        return ()
    names = font.get("name")
    axes = []
    for axis in font["fvar"].axes:
        label = names.getDebugName(axis.axisNameID) if names is not None else None
        axes.append(
            AxisInfo(
                tag=axis.axisTag,
                name=label or axis.axisTag,
                minimum=float(axis.minValue),
                default=float(axis.defaultValue),
                maximum=float(axis.maxValue),
            )
        )
    return tuple(axes)


def read_avar(font: Any) -> AvarSegments:
    """Return the avar segment maps of ``font``, empty when it has none."""
    if "avar" not in font:
        return {}
    return {tag: dict(mapping) for tag, mapping in font["avar"].segments.items()}


def normalize(
    location_user: Mapping[str, float], axes: tuple[AxisInfo, ...], avar_segments: AvarSegments
) -> dict[str, float]:
    """Map a user-space location to the normalized space the masters live in (fvar, then avar)."""
    triples = {a.tag: (a.minimum, a.default, a.maximum) for a in axes}
    normalized: dict[str, float] = normalizeLocation(dict(location_user), triples)
    return {
        tag: float(piecewiseLinearMap(value, avar_segments[tag])) if tag in avar_segments else value
        for tag, value in normalized.items()
    }


class VariableOutcomeCache:
    """LRU of full outcomes for previews plus a bridge-count-only map for the survey.

    Holds one parameter set (bridge and geometry JSON); a request with another set
    clears both maps. The lock guards lookups and inserts only, never a computation.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._outcomes: OrderedDict[OutcomeKey, VariableOutcome] = OrderedDict()
        self._counts: dict[OutcomeKey, int] = {}
        self._params: tuple[str, str] | None = None
        self._lock = threading.Lock()

    @property
    def outcome_keys(self) -> list[OutcomeKey]:
        """Cached outcome keys, least recently used first."""
        with self._lock:
            return list(self._outcomes)

    @property
    def counts(self) -> dict[OutcomeKey, int]:
        """A copy of the survey's per-key bridge counts."""
        with self._lock:
            return dict(self._counts)

    def _sync(self, key: OutcomeKey) -> None:
        """Clear both maps when ``key`` belongs to another parameter set. Lock must be held."""
        params = (key[1], key[2])
        if params != self._params:
            self._outcomes.clear()
            self._counts.clear()
            self._params = params

    def outcome(self, key: OutcomeKey, compute: Callable[[], VariableOutcome]) -> VariableOutcome:
        """Return the cached outcome for ``key``, computing it outside the lock on a miss."""
        with self._lock:
            self._sync(key)
            hit = self._outcomes.get(key)
            if hit is not None:
                self._outcomes.move_to_end(key)
                return hit
        value = compute()
        with self._lock:
            self._sync(key)
            self._outcomes[key] = value
            self._outcomes.move_to_end(key)
            while len(self._outcomes) > self._max_entries:
                self._outcomes.popitem(last=False)
        return value

    def bridge_count(self, key: OutcomeKey, compute: Callable[[], VariableOutcome]) -> int:
        """Return the bridge count for ``key``, storing only the integer."""
        with self._lock:
            self._sync(key)
            if key in self._counts:
                return self._counts[key]
            hit = self._outcomes.get(key)
            if hit is not None:
                return hit.bridge_count
        count = compute().bridge_count
        with self._lock:
            self._sync(key)
            self._counts[key] = count
        return count


@dataclass(frozen=True)
class VariablePreview:
    """Original and stenciled outlines of one variable glyph at one location."""

    original: Glyph
    stenciled: Glyph | None
    bridges_added: int
    error: str | None


class VariableSurface:
    """Variable data of an open session: glyphs, axes and the outcome cache."""

    def __init__(
        self,
        axes: tuple[AxisInfo, ...],
        avar_segments: AvarSegments,
        glyphs: dict[str, VariableGlyph],
        unsupported: dict[str, Glyph],
        upm: int,
        cache: VariableOutcomeCache,
    ) -> None:
        self.axes = axes
        self.unsupported = unsupported
        self._avar = avar_segments
        self._glyphs = glyphs
        self._upm = upm
        self._cache = cache

    def normalize(self, location: Mapping[str, float] | None) -> dict[str, float]:
        """Normalize a user-space location; ``None`` or missing axes mean fvar defaults."""
        return normalize(location or {}, self.axes, self._avar)

    def _key(
        self, name: str, bridge: BridgeConfig, geometry: GeometryConfig, direction: BridgeDirection
    ) -> OutcomeKey:
        return (name, bridge.model_dump_json(), geometry.model_dump_json(), direction)

    def _compute(
        self,
        name: str,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        direction: BridgeDirection,
    ) -> Callable[[], VariableOutcome]:
        configured = bridge.model_copy(update={"direction": direction})
        return lambda: transform_variable_glyph(self._glyphs[name], configured, geometry, self._upm)

    def preview(
        self,
        name: str,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection],
        location: Mapping[str, float] | None,
    ) -> VariablePreview:
        """Preview ``name`` at a user-space location; a slider move costs only ``instance``."""
        if name in self.unsupported:
            return VariablePreview(self.unsupported[name], None, 0, UNSUPPORTED_REASON)
        direction = directions.get(name, bridge.direction)
        normalized = self.normalize(location)
        try:
            original = self._glyphs[name].instance(normalized)
        except VariationDataError:
            # Singular variation data: show the default outline, as the survey and CLI skip it.
            return VariablePreview(self._glyphs[name].default, None, 0, UNSUPPORTED_REASON)
        try:
            outcome = self._cache.outcome(
                self._key(name, bridge, geometry, direction),
                self._compute(name, bridge, geometry, direction),
            )
            if outcome.bridge_count == 0:
                # Like the static path: the unchanged outline, no bridge placed.
                return VariablePreview(original, original, 0, None)
            return VariablePreview(
                original, outcome.glyph.instance(normalized), outcome.bridge_count, None
            )
        except Exception as error:  # mirrors process_glyph's catch-all on the static path
            return VariablePreview(original, None, 0, str(error))

    def bridge_count(
        self,
        name: str,
        bridge: BridgeConfig,
        geometry: GeometryConfig,
        directions: Mapping[str, BridgeDirection],
    ) -> int:
        """Bridge count of ``name`` for the survey, without keeping its outline.

        A transform failure counts as zero bridges, as an errored static glyph does.
        """
        direction = directions.get(name, bridge.direction)
        try:
            return self._cache.bridge_count(
                self._key(name, bridge, geometry, direction),
                self._compute(name, bridge, geometry, direction),
            )
        except Exception:
            logger.debug("Bridge count failed for glyph %s", name, exc_info=True)
            return 0
