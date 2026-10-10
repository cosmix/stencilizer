"""Read variable glyphs: default outline, variation supports and a master at each peak."""

from typing import TYPE_CHECKING

from fontTools.pens.basePen import NullPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.exceptions import VariationDataError
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.variable.model import Support, VariableGlyph

if TYPE_CHECKING:
    from stencilizer.domain.glyph import Glyph


def is_variable(font: TTFont) -> bool:
    """True when the font has an ``fvar`` table."""
    return "fvar" in font


def cff2_vsindex(font: TTFont, name: str) -> int | None:
    """The VarData index the glyph's charstring blends with, or None when it never blends.

    The charstring is drawn with a recording blender so subroutines are followed: blend
    operators of subroutinized fonts live only in subrs. Drawing costs a full charstring pass,
    so the CFF2 writer calls this once and passes the index to ``cff2_region_supports``.
    """
    top_dict = font["CFF2"].cff.topDictIndex[0]
    recorded: list[int] = []

    def blender(vs_index: int, _deltas: list[float]) -> float:
        recorded.append(vs_index)
        return 0

    top_dict.CharStrings[name].draw(NullPen(), blender)
    distinct = set(recorded)
    if len(distinct) > 1:
        raise VariationDataError(name, "more than one vsindex in one charstring")
    return recorded[0] if recorded else None


def _gvar_supports(font: TTFont, name: str) -> tuple[Support, ...]:
    variations = font["gvar"].variations.get(name, []) if "gvar" in font else []
    return tuple(
        Support(tuple(sorted((tag, s, p, e) for tag, (s, p, e) in tv.axes.items())))
        for tv in variations
    )


def cff2_supports(font: TTFont, name: str) -> tuple[Support, ...]:
    """The supports of the VarData regions the glyph's charstring blends with, in region order.

    Empty when the glyph never blends.
    """
    vs_index = cff2_vsindex(font, name)
    if vs_index is None:
        return ()
    return cff2_region_supports(font, name, vs_index)


def cff2_region_supports(font: TTFont, name: str, vs_index: int) -> tuple[Support, ...]:
    """Supports of VarData ``vs_index``'s regions, after checking the FD Private agrees."""
    top_dict = font["CFF2"].cff.topDictIndex[0]
    _, fd_index = top_dict.CharStrings.getItemAndSelector(name)
    private_index = getattr(top_dict.FDArray[fd_index or 0].Private, "vsindex", None)
    if private_index is not None and private_index != vs_index:
        raise VariationDataError(name, "Private vsindex not honoured by fontTools glyph sets")
    store = top_dict.VarStore.otVarStore
    if vs_index >= len(store.VarData):
        raise VariationDataError(name, "vsindex outside the variation store")
    tags = [axis.axisTag for axis in font["fvar"].axes]
    regions = store.VarRegionList.Region
    supports = []
    for region_index in store.VarData[vs_index].VarRegionIndex:
        axes = regions[region_index].VarRegionAxis
        supports.append(
            Support(
                tuple(
                    sorted(
                        (tag, a.StartCoord, a.PeakCoord, a.EndCoord)
                        for tag, a in zip(tags, axes, strict=True)
                        if a.PeakCoord != 0
                    )
                )
            )
        )
    return tuple(supports)


def read_variable_glyph(
    font: TTFont, name: str, unicode_by_name: dict[str, int] | None = None
) -> VariableGlyph | None:
    """Read ``name`` with masters at every support peak; None for composite or empty glyf glyphs.

    Pass ``unicode_by_name`` (glyph name to code point, built once from the cmap) when reading
    many glyphs; without it the converter rebuilds the cmap reverse map for this glyph.
    """
    if "glyf" in font and font["glyf"][name].isComposite():
        return None
    default = fonttools_glyph_to_domain(name, font.getGlyphSet()[name], font, unicode_by_name)
    cff2 = "CFF2" in font
    # The frozen contract reads Cantarell's empty ``.notdef`` as a glyph with no supports, so
    # only glyf fonts report an empty glyph as None.
    if default.is_empty() and not cff2:
        return None
    supports = cff2_supports(font, name) if cff2 else _gvar_supports(font, name)
    code_point = default.metadata.unicode
    master_unicode = {name: code_point} if code_point is not None else {}
    masters: list[Glyph] = []
    for support in supports:
        glyph_set = font.getGlyphSet(location=support.peak(), normalized=True)
        masters.append(fonttools_glyph_to_domain(name, glyph_set[name], font, master_unicode))
    return VariableGlyph(
        default=default,
        supports=supports,
        masters=tuple(masters),
        axis_tags=tuple(axis.axisTag for axis in font["fvar"].axes),
        cff2=cff2,
    )
