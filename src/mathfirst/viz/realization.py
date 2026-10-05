"""Visual interpretations of mathematical objects, independent of rendering."""

from dataclasses import dataclass

from ..function import Function


@dataclass(frozen=True, slots=True)
class FunctionGraph:
    """Interpret a one-variable function as a graph in two dimensions.

    The source retains its mathematical metadata. Numerical evaluation, viewing
    ranges, and rendering belong to the viewer.
    """

    function: Function

    def __post_init__(self) -> None:
        if not isinstance(self.function, Function):
            raise TypeError("FunctionGraph requires a Function or Polynomial")
        if len(self.function.variables) != 1:
            raise ValueError("FunctionGraph requires exactly one variable")
