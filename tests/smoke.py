"""Validate installed distributions without importing the source checkout."""

import sys
from importlib import resources
from importlib.metadata import version

from mathfirst import Polynomial, Scalar, Variable, derivative, real_roots
from mathfirst.viz import FunctionGraph, Viewer

assert not {"numpy", "fastapi", "pydantic", "uvicorn", "websockets"}.intersection(
    sys.modules
)
assert (
    resources.files("mathfirst.viz").joinpath("static/vendor/plotly.min.js").is_file()
)

x = Variable("x")
polynomial = Polynomial((x - 1) ** 2 * (x + 2), x)
assert polynomial(Scalar(3)) == 20
assert derivative(polynomial)(Scalar(1)) == 0
roots = real_roots(polynomial)
assert tuple((root.value, root.multiplicity) for root in roots) == ((-2, 1), (1, 2))
viewer = Viewer(FunctionGraph(polynomial))
assert viewer.url is None
print("Installed mathfirst", version("mathfirst"))
