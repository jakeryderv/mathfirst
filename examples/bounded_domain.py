"""Graph x² on the open interval (-2, 2).

Run: uv run --locked --extra viz python examples/bounded_domain.py

The initial horizontal range comes from the function's domain. Pan past either
boundary: samples outside the interval remain blank. The endpoints are excluded;
this viewer does not draw special markers for open endpoints.
"""

from mathfirst import Function, Set, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
domain = Set.interval(-2, 2, left_open=True, right_open=True)
function = Function(x**2, (x,), domain=domain, codomain=Set.reals())
graph = FunctionGraph(function)
viewer = Viewer(graph)

if __name__ == "__main__":
    viewer.show()
