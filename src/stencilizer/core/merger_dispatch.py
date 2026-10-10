"""Choose and construct bridge contours after geometry checks."""

from dataclasses import dataclass

from stencilizer.config.settings import GeometryConfig
from stencilizer.core.horizontal_bridge import create_horizontal_bridge_contours
from stencilizer.core.merger_checks import MergeMeasure
from stencilizer.core.vertical_bridge import create_vertical_bridge_contours
from stencilizer.domain import Contour


@dataclass
class MergeDispatch:
    inner: Contour
    outer: Contour
    measure: MergeMeasure
    all_contours: list[Contour] | None
    processed_nested: list[Contour] | None
    geometry: GeometryConfig
    upm: int

    def horizontal(
        self, center_y: float | None = None, extent: tuple[float, float] | None = None
    ) -> list[Contour]:
        m = self.measure
        return create_horizontal_bridge_contours(
            self.inner,
            self.outer,
            m.center_y if center_y is None else center_y,
            m.half_width,
            *(extent or (m.inner_min_x, m.inner_max_x)),
            all_contours=self.all_contours,
            processed_nested=self.processed_nested,
            epsilon=self.geometry.get_line_epsilon(self.upm),
            bridge_tolerance=self.geometry.get_contour_gap(self.upm),
            point_tolerance=self.geometry.get_point_dedup_tolerance(self.upm),
        )

    def vertical(
        self, center_x: float | None = None, extent: tuple[float, float] | None = None
    ) -> list[Contour]:
        m = self.measure
        return create_vertical_bridge_contours(
            self.inner,
            self.outer,
            m.center_x if center_x is None else center_x,
            m.half_width,
            *(extent or (m.inner_min_y, m.inner_max_y)),
            all_contours=self.all_contours,
            processed_nested=self.processed_nested,
            epsilon=self.geometry.get_line_epsilon(self.upm),
            bridge_tolerance=self.geometry.get_contour_gap(self.upm),
            point_tolerance=self.geometry.get_point_dedup_tolerance(self.upm),
        )

    def forced_horizontal(self) -> list[Contour]:
        m = self.measure
        if m.can_horizontal:
            result = self.horizontal()
            if result != [self.outer, self.inner]:
                return result
        if m.can_vertical:
            return self.vertical()
        return [self.outer, self.inner]

    def forced_vertical(self) -> list[Contour]:
        m = self.measure
        if m.can_vertical:
            result = self.vertical()
            if result != [self.outer, self.inner]:
                return result
        if m.can_horizontal:
            return self.horizontal()
        return [self.outer, self.inner]

    def preferred(self) -> list[Contour]:
        m = self.measure
        floor = self.geometry.scaled("asymmetry_floor", self.upm)
        vertical_asymmetry = max(m.stroke_top, m.stroke_bottom) / max(
            min(m.stroke_top, m.stroke_bottom), floor
        )
        horizontal_asymmetry = max(m.stroke_left, m.stroke_right) / max(
            min(m.stroke_left, m.stroke_right), floor
        )
        force_horizontal_for_asymmetry = (
            m.can_horizontal
            and vertical_asymmetry > 2.5
            and vertical_asymmetry > horizontal_asymmetry
        )
        force_horizontal_for_off_center = False
        prefer_vertical = (
            m.can_vertical
            and not force_horizontal_for_asymmetry
            and not force_horizontal_for_off_center
            and (not m.can_horizontal or m.vertical_stroke <= m.horizontal_stroke)
        )
        if prefer_vertical:
            result = self.vertical()
            if result == [self.outer, self.inner] and m.can_horizontal:
                result = self.horizontal()
        else:
            result = self.horizontal()
            if result == [self.outer, self.inner] and m.can_vertical:
                result = self.vertical()
        if result == [self.outer, self.inner] and m.can_horizontal:
            bottom_center_y = (
                m.inner_min_y
                + m.half_width
                + self.geometry.scaled("bottom_bridge_offset", self.upm)
            )
            result = self.horizontal(bottom_center_y)
        return result
