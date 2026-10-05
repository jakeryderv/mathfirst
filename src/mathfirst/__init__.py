"""Mathematical objects and operations backed by SymPy."""

from .core import (
    ApproximateScalarWarning,
    Expression,
    Point,
    Root,
    Scalar,
    Set,
    Variable,
)
from .function import Function
from .operations import derivative, limit, real_roots
from .polynomial import Polynomial

__all__ = [
    "ApproximateScalarWarning",
    "Expression",
    "Function",
    "Point",
    "Polynomial",
    "Root",
    "Scalar",
    "Set",
    "Variable",
    "derivative",
    "limit",
    "real_roots",
]
