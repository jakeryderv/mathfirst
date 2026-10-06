"""Exact mathematics and explicit NumPy evaluation, without a browser.

Run: uv run --locked python examples/numerical_evaluation.py

The numerical results use MathFirst's declared float64/complex128 defaults or
an explicitly requested dtype. The source objects retain their mathematics.
"""

import numpy as np

from mathfirst import Function, Point, Polynomial, Scalar, Variable

x = Variable("x")
offset = Scalar.exact("0.1")
p = Polynomial(x**2 + offset, x)

# Calling a function substitutes mathematically; numerical evaluation is separate.
exact_result = p(2)  # Exact 41/10.
grid = np.linspace(-2, 2, 5, dtype=np.float64)
values = p.evaluate_numpy(grid)
values_float32 = p.evaluate_numpy(grid, dtype=np.float32)
numerical_offset = offset.to_numpy()

# Coordinates are copied into a writable numerical array; the Point stays exact.
point = Point(offset, exact_result)
coordinates = point.to_numpy()
coordinates[0] = 99
original_coordinates = point.to_numpy()  # Still [0.1, 4.1].

# Multivariable arguments broadcast: (3, 1) and (1, 2) produce (3, 2).
y = Variable("y")
function = Function(x + y, (x, y))
rows = np.arange(3, dtype=np.float64)[:, None]
columns = np.array([10, 20], dtype=np.float64)[None, :]
broadcast_values = function.evaluate_numpy(rows, columns)

# Real invalid values require opt-in. Complex arithmetic requires a complex dtype.
square_root = Function(x ** Scalar.exact("1/2"), (x,))
real_square_roots = square_root.evaluate_numpy([-1, 0, 1], allow_nonfinite=True)
complex_square_roots = square_root.evaluate_numpy([-1, 0, 1], dtype=np.complex128)

if __name__ == "__main__":
    print("Exact p(2):", exact_result)
    print("Concrete offset:", numerical_offset)
    print("Grid:", grid)
    print("float64 values:", values)
    print("float32 values:", values_float32)
    print("Point after editing the numerical copy:", point)
    print("Fresh coordinates:", original_coordinates)
    print("Broadcast values:\n", broadcast_values)
    print("Real square roots (NaN for -1):", real_square_roots)
    print("Complex square roots:", complex_square_roots)
