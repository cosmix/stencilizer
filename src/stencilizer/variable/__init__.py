"""Variable-font engine: model, solver, reader and compatible flattening."""

from stencilizer.variable.flatten import flatten_compatible
from stencilizer.variable.model import Support, VariableGlyph
from stencilizer.variable.reader import is_variable, read_variable_glyph
from stencilizer.variable.solver import solve_deltas

__all__ = [
    "Support",
    "VariableGlyph",
    "flatten_compatible",
    "is_variable",
    "read_variable_glyph",
    "solve_deltas",
]
