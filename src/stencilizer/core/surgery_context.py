"""Shared state for one glyph's contour surgery."""

from dataclasses import dataclass, field

from stencilizer.config.settings import BridgeDirection, GeometryConfig
from stencilizer.core.analyzer import ContourHierarchy
from stencilizer.core.merger import ContourMerger
from stencilizer.domain import Contour, Glyph


@dataclass
class SurgeryContext:
    glyph: Glyph
    hierarchy: ContourHierarchy
    bridge_width: float
    merger: ContourMerger
    geometry: GeometryConfig
    upm: int
    use_spanning: bool
    direction: BridgeDirection = BridgeDirection.AUTO
    processed: set[int] = field(default_factory=set)
    contours: list[Contour] = field(default_factory=list)
    contour_to_idx: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.contour_to_idx = {id(c): i for i, c in enumerate(self.glyph.contours)}

    def track_nested(self, nested: list[Contour]) -> None:
        for contour in nested:
            idx = self.contour_to_idx.get(id(contour))
            if idx is not None:
                self.processed.add(idx)

    def merge(
        self,
        inner: Contour,
        outer: Contour,
        nested: list[Contour],
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
    ) -> list[Contour]:
        if not force_horizontal and not force_vertical:
            force_horizontal = self.direction is BridgeDirection.HORIZONTAL
            force_vertical = self.direction is BridgeDirection.VERTICAL
        return self.merger.merge_contours_with_bridges(
            inner,
            outer,
            self.bridge_width,
            force_horizontal=force_horizontal,
            force_vertical=force_vertical,
            all_contours=self.glyph.contours,
            processed_nested=nested,
            upm=self.upm,
            geometry=self.geometry,
        )
