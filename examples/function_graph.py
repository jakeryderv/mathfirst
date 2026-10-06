"""Run with: uv run --locked --extra viz python examples/function_graph.py."""

from mathfirst import Polynomial, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
p = Polynomial(x**3 - 2 * x + 1, x)
graph = FunctionGraph(p)
viewer = Viewer(graph)

if __name__ == "__main__":
    viewer.show()
