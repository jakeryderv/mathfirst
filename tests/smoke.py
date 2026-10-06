"""Validate installed distributions without importing the source checkout."""

import sys
from fractions import Fraction
from importlib import resources
from importlib.metadata import version

import numpy as np

from mathfirst import Polynomial, Scalar, Variable, derivative, real_roots
from mathfirst.numerical import as_numeric_array
from mathfirst.viz import FunctionGraph, Viewer

assert not {"fastapi", "pydantic", "uvicorn", "websockets"}.intersection(sys.modules)
assert (
    resources.files("mathfirst.viz").joinpath("static/vendor/plotly.min.js").is_file()
)

x = Variable("x")
polynomial = Polynomial((x - 1) ** 2 * (x + 2), x)
assert polynomial(Scalar(3)) == 20
assert isinstance(Scalar.exact("0.1").to_numpy(), np.float64)
assert Scalar.exact(Fraction(71, 179)).to_numpy() == float(Fraction(71, 179))
assert Scalar(2**63 - 1).to_numpy(dtype=np.int64) == np.iinfo(np.int64).max
assert as_numeric_array([1, 2]).dtype == np.dtype(np.float64)
assert derivative(polynomial)(Scalar(1)) == 0
roots = real_roots(polynomial)
assert tuple((root.value, root.multiplicity) for root in roots) == ((-2, 1), (1, 2))
viewer = Viewer(FunctionGraph(polynomial))
assert viewer.url is None
print("Installed mathfirst", version("mathfirst"))
