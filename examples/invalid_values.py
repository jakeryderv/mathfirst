"""Graph sqrt(x) across negative and positive inputs.

Run: uv run --locked --extra viz python examples/invalid_values.py

The viewer starts at (-5, 5). Negative inputs have no real square-root values,
so the left side is blank; the real curve starts at the origin. Zoom or pan into
negative inputs to see an empty range, then reset to restore the initial view.
"""

from mathfirst import Function, Scalar, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
half = Scalar.exact("1/2")
# Leave the domain unspecified to exercise invalid-value handling while viewing.
function = Function(x**half, (x,))
graph = FunctionGraph(function)
viewer = Viewer(graph)

if __name__ == "__main__":
    viewer.show()
