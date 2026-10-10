"""Variable-font engine: model, solver, reader, compatible flattening and glyph stencilling.

``transform_variable_glyph`` stencils one glyph (overlap removal via
``remove_overlaps_compatible``, then surgery replayed on every master) and returns a
``VariableOutcome``.
"""

from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.overlaps import remove_overlaps_compatible
from stencilizer.variable.reader import is_variable, read_variable_glyph
from stencilizer.variable.solver import solve_deltas
from stencilizer.variable.transform import VariableOutcome, transform_variable_glyph

__all__ = [
    "Support",
    "VariableGlyph",
    "VariableOutcome",
    "flatten_compatible",
    "is_variable",
    "read_variable_glyph",
    "remove_overlaps_compatible",
    "solve_deltas",
    "transform_variable_glyph",
]
