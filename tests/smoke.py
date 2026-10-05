"""Validate installed distributions without importing the source checkout."""

from importlib.metadata import version

from mathfirst import Polynomial, Scalar, Variable, derivative, real_roots

x = Variable("x")
polynomial = Polynomial((x - 1) ** 2 * (x + 2), x)
assert polynomial(Scalar(3)) == 20
assert derivative(polynomial)(Scalar(1)) == 0
roots = real_roots(polynomial)
assert tuple((root.value, root.multiplicity) for root in roots) == ((-2, 1), (1, 2))
print("Installed mathfirst", version("mathfirst"))
