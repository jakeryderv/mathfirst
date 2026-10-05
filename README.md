# mathfirst

`mathfirst` is a math-native layer for representing and working with mathematics in Python.

The core idea is simple:

> **Model mathematics according to its mathematical meaning first, and treat symbolic, numerical, visualization, and other libraries as implementation backends.**

Instead of building directly around library-specific objects such as SymPy expressions, NumPy arrays, or plotting primitives, `mathfirst` defines its own mathematical concepts and relationships:

```text
Scalar
Variable
Expression
Set
Function
Polynomial
Point
Root
...
```

These objects should preserve the structure and semantics of the mathematics they represent, including concepts such as exactness, assumptions, domains, codomains, dimensions, and algebraic relationships.

SymPy is the current symbolic backend, and NumPy supports numerical sampling in the optional browser viewer. SciPy or mpmath may support future computational needs, while the public mathematical model remains defined by `mathfirst`.

The guiding principles are:

- **Mathematical meaning before implementation details**
- **Exact mathematics where possible; approximation should be explicit**
- **Objects represent mathematical concepts**
- **Operations transform mathematical objects**
- **Mathematical structure and assumptions should be explicit**
- **Backend-specific behavior should remain behind deliberate boundaries**
- **Downstream systems consume the common math model**

## Current behavior and usage

- `Variable("x")` represents a real variable by default; equal names represent the same variable.
- `Scalar` preserves exact integers, fractions, and decimal strings. `Scalar.exact(...)` rejects approximate inputs; `Scalar.approx(...)` explicitly requests approximation. Passing a Python float directly to `Scalar` emits an `ApproximateScalarWarning` with both alternatives.
- `Scalar.is_exact` describes the current representation, rather than the provenance of previous arithmetic.
- Function domains and codomains are optional, descriptive metadata. Construction validates expression and argument structure, and evaluation checks argument count. Set membership is available explicitly through `Set.contains()`; Cartesian products use `Set.product()`.
- General derivatives leave domain and codomain unspecified unless supplied. Polynomial derivatives remain polynomials with real default sets.
- Expression equality is structural. `real_roots()` returns exact real roots with multiplicities; `limit()` defaults to a two-sided approach and accepts an explicit direction.
- `to_sympy()` provides a deliberate backend escape hatch, including SymPy lambdas for functions and SymPy polynomials for polynomials.

```python
from fractions import Fraction

from mathfirst import Polynomial, Scalar, Variable, derivative

x = Variable("x")
exact_decimal = Scalar.exact("0.1")  # Exact 1/10
exact_fraction = Scalar(Fraction(1, 3))
approximate = Scalar.approx(0.1)  # Explicit approximation, without a warning

p = Polynomial(x**2 - exact_decimal, x)
print(p(Scalar(2)))  # 39/10
print(derivative(p))  # 2*x
backend_polynomial = p.to_sympy()
```

Install with `uv add mathfirst` or `pip install mathfirst`. Requires Python
3.12–3.14. SymPy is the current required backend.

Package tests live in [`tests/`](tests/). See the
[development guide](docs/development.md) for setup, checks, and releases.

## Browser visualization

Install the optional viewer with `uv add "mathfirst[viz]"` or
`pip install "mathfirst[viz]"`. In this checkout, use `uv sync --locked --extra viz`.

```python
from mathfirst import Polynomial, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
p = Polynomial(x**3 - 2 * x + 1, x)
viewer = Viewer(FunctionGraph(p))
viewer.show()  # Opens a local browser viewer and blocks until stopped.
```

The first slice displays one function graph with pan, zoom, and hover. `FunctionGraph`
is declarative; `Viewer` owns sampling, rendering, the server, and internal view state.
Use `viewer.show(block=False)` explicitly for notebooks or background use, then
`viewer.close()` when finished. See [visualization usage and limits](docs/viz.md)
and [the runnable example](examples/function_graph.py).

CLI/watch mode, figures/exports, multiple realizations, and additional backends
are future work.

Visualization is therefore not the purpose of the math layer itself; it is one consumer of it.

The long-term goal is for code using `mathfirst` to read and behave more like the mathematics being expressed, rather than like the particular libraries used to compute it.
