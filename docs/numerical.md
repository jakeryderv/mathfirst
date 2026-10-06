# Numerical contracts

MathFirst separates mathematical objects from concrete numerical values, following
[the design philosophy](../DESIGN.md). `Scalar`, `Expression`, `Function`, and
`Polynomial` retain their SymPy mathematical representations. Numerical conversion
is an explicit operation and does not replace or mutate the mathematical source.

## Domain defaults

These are MathFirst's numerical conversion defaults, not assumptions imposed on
every mathematical domain or numerical backend:

| API/domain | Canonical representation |
| --- | --- |
| Exact and symbolic mathematics | SymPy expressions, preserving exact integers, fractions, and constants |
| `Scalar.to_numpy()` | `np.float64` for known real values; `np.complex128` otherwise; NaN and signed infinity use the real default |
| `as_numeric_array()` | `NDArray[np.float64]`, unless the caller declares another supported dtype |
| `Function.evaluate_numpy()` / `Polynomial.evaluate_numpy()` | `np.float64` or `NDArray[np.float64]`; explicit floating/complex dtype applies to inputs and results |
| `Point.to_numpy()` | `NDArray[np.float64]` of shape `(dimension,)`; explicit complex or integer output is available |
| Viewer bounds, sampling grids, and ordinates | `NDArray[np.float64]`; intermediate function values use `complex128` for validity checks |
| Numerical validity masks | `NDArray[np.bool_]`, distinct from numerical magnitudes |

`DEFAULT_REAL_DTYPE` and `DEFAULT_COMPLEX_DTYPE` in `mathfirst.numerical` declare
the defaults. Other domains can declare their own dtypes. There is no package-wide
fixed-width integer default: exact integers remain unbounded, and callers requesting
concrete integer output choose its width and signedness. Names, control booleans,
dimensions, multiplicities, and implementation counters retain their existing
semantic or exact representations.

NumPy and SymPy are core dependencies. The optional `viz` extra supplies the web
runtime. `Function.__call__()` continues to perform symbolic
substitution and return mathematical objects, including for NumPy scalar arguments.

## Scalar input and intent

- Python and NumPy integers enter the mathematical model exactly, including values
  outside signed 64-bit range.
- Fractions, exact decimal strings, and existing SymPy expressions preserve their
  mathematical representation.
- Direct Python/NumPy floating and complex inputs to `Scalar` emit
  `ApproximateScalarWarning`. `Scalar.approx(...)` explicitly accepts approximation;
  `Scalar.exact(...)` rejects these input types, including complex values whose
  imaginary component is zero.
- Existing approximate SymPy expressions remain approximate without repeated
  warnings during algebra. `Scalar.approx(...)` produces an approximate mathematical
  expression using SymPy's evaluation precision; `to_numpy()` produces concrete
  NumPy values. Increasing precision cannot restore information lost before input.
- Python/NumPy booleans and arrays are not scalar mathematical expressions.

`is_exact` continues to describe the current representation, not input provenance.
Nonfinite symbolic values retain the existing representation-based behavior.

## Explicit NumPy scalar conversion

```python
import numpy as np
import sympy as sp

from mathfirst import Scalar

ratio = Scalar.exact("0.1")
real = ratio.to_numpy()  # np.float64
small = ratio.to_numpy(dtype=np.float32)  # Explicit precision choice
imaginary = Scalar.exact(2 * sp.I).to_numpy()  # np.complex128
integer = Scalar(255).to_numpy(dtype=np.uint8)
assert ratio.is_exact  # Source remains exact
```

`dtype` accepts NumPy scalar classes, dtype objects, or dtype strings. Supported
representations are signed/unsigned integers of 8, 16, 32, or 64 bits;
`float16`, `float32`, `float64`; and `complex64`, `complex128`. Boolean, object,
structured, datetime, and extended-precision output dtypes are outside this
conversion contract. Explicit aliases such as `dtype=int` follow NumPy's dtype
resolution; choose an explicit width when portability matters.

Integer conversion requires a proven mathematical integer and checks its range
without passing through floating point. Fractional values and approximate values
such as `Scalar.approx(3.0)` are rejected rather than truncated or reclassified as
exact integers. Real output rejects nonzero imaginary components.

Floating/complex scalar conversion evaluates through binary64 real/complex values
and then rounds to the requested dtype. This can lose exactness and precision;
underflow to zero is permitted. Finite values overflowing the evaluation or output
representation raise `OverflowError`, even when nonfinite values are allowed.
This API does not provide arbitrary-precision evaluation or correctly rounded
direct conversion to every narrower dtype.

NaN and signed infinity require `allow_nonfinite=True`. Complex infinity (`sp.zoo`)
and unevaluable numerical expressions are rejected. Control options require native
Python booleans. Unsupported types/dtypes raise `TypeError`; invalid values raise
`ValueError`; representational overflow raises `OverflowError`.

Returned NumPy scalars are immutable, independent numerical results. There is no
mutable array view or numerical cache attached to the mathematical source.

## Numerical function evaluation

```python
import numpy as np

from mathfirst import Polynomial, Scalar, Variable

x = Variable("x")
p = Polynomial(x**2 + Scalar.exact("0.1"), x)
exact = p(2)  # Scalar containing exact 41/10
value = p.evaluate_numpy(2)  # np.float64
values = p.evaluate_numpy([0, 1, 2])  # Writable float64 array, shape (3,)
smaller = p.evaluate_numpy([0, 1], dtype=np.float32)
```

`Function.evaluate_numpy(*values, dtype=..., allow_nonfinite=...)` also applies to
`Polynomial`. It checks argument count and canonicalizes concrete inputs before
numerical evaluation. Mathematical `Scalar` inputs must be converted explicitly
with `to_numpy()` first; unresolved expressions and object arrays are rejected.
Domain and codomain sets remain descriptive metadata, just as for exact substitution.

Arguments follow [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
For example, two arguments of shapes `(3, 1)` and `(1, 4)` produce shape `(3, 4)`.
The output always has the common broadcast shape, including when an argument is
unused, an expression is constant, or an input is empty. Incompatible argument
shapes and unexpected backend result shapes raise `ValueError`. All scalar or
zero-dimensional inputs produce a NumPy scalar, including a constant function
with no arguments. Otherwise the result is an independent, writable array;
constant broadcasting does not expose NumPy's read-only broadcast view.

The default input/output dtype is MathFirst's `float64`. Callers may select
`float16/32/64` or `complex64/128`. Integer output is excluded from function
evaluation, which can involve approximate arithmetic. Complex input requires an
explicit complex dtype. A complex dtype also changes backend arithmetic: for
example, `sqrt(-1)` produces `1j` with `complex128`, while real evaluation produces
NaN. An expression already simplified using real-variable assumptions is not
guaranteed to describe a complex continuation.

The backend is [SymPy's NumPy lambdify](https://docs.sympy.org/latest/modules/utilities/lambdify.html);
unsupported expressions raise `ValueError`. Attached custom SymPy implementations
are not used. Numerical constants use the scalar conversion policy, including
large exact integers that fit a floating representation. The operation creates
no numerical cache or mutable numerical state on the mathematical object.

Nonfinite inputs and results raise `ValueError` unless `allow_nonfinite=True`.
With that opt-in, NaN/infinity from floating arithmetic, such as division by zero
or arithmetic overflow, are retained. Conversion of a finite value that overflows
its representation still raises `OverflowError`. Nonzero imaginary components
cannot be discarded by real output. Floating arithmetic, rounding, and underflow
follow the selected backend dtype; these results carry no exactness guarantee.

## Numerical point coordinates

```python
from fractions import Fraction

from mathfirst import Point

point = Point(Fraction(1, 3), 2)
coordinates = point.to_numpy()  # Writable float64 array, shape (2,)
coordinates[0] = 0  # The mathematical point still contains exact 1/3.
```

`Point` retains ordered mathematical coordinates. `to_numpy()` returns a new
one-dimensional array of shape `(point.dimension,)` and defaults to MathFirst's
`float64`. Complex coordinates require an explicit complex dtype. Each coordinate
follows `Scalar.to_numpy()` validation and precision policies, including proven
integer/range checks for explicit integer output and nonfinite opt-in.
Unresolved symbolic coordinates must be substituted before conversion; unevaluable
coordinates are rejected. Repeated calls produce independent arrays and do not
attach mutable numerical state to the point.

## Array canonicalization and ownership

```python
import numpy as np

from mathfirst.numerical import as_numeric_array

owned = as_numeric_array([1, 2, 3])  # Independent, writable float64 array
source = np.array([1.0, 2.0], dtype=np.float64)
borrowed = as_numeric_array(source, copy=False)
assert np.shares_memory(borrowed, source)
```

`as_numeric_array()` accepts numerical Python scalars/sequences and NumPy arrays,
normalizing them to an explicitly selected dtype. Boolean, string, object, and
masked data are rejected; masks require a separate validity contract. Ragged
sequences are not supported. Empty numerical collections are valid.

Inputs whose NumPy dtype inference produces object arrays, such as unbounded
integer sequences, require deliberate scalar conversion first. This helper does
not infer exact-value or symbolic semantics from object arrays.

The default `copy=True` creates independent, writable storage, including when the
source is read-only or a view. Callers may mutate the result without changing the
source. `copy=False` requires an existing NumPy array with the requested dtype;
it rejects inputs requiring conversion or allocation. Returned data may share
storage and mutations with the source and preserves its writeability. Read-only
access does not freeze other aliases to that storage.

Integer output requires integer input and checked range. Real output rejects
nonzero imaginary components. Floating casts may round or underflow. Nonfinite
inputs require opt-in, and finite-component overflow is always rejected. Conversion
cannot recover precision already lost through input construction or dtype inference.

The helper establishes representation, not application-specific shape, rank, units,
positivity, or other domain invariants. Validate these at the relevant domain boundary.
`numpy.typing` aliases and overloads support static checking with ty; explicit runtime
validation enforces the conversion contracts. No jaxtyping/beartype dependency is
needed for this slice.

## Viewer boundaries and ownership

The viewer owns sampling resolution and runtime state. Bounds are copied into
read-only `float64` arrays of shape `(2,)`, then checked for finite, increasing
limits with a finite width. An unbounded declared interval may have infinite
domain endpoints; the actual viewport must remain finite. Caller mutations to
input bounds cannot change the viewer's current or initial range.

Each sampling call produces fresh, read-only `float64` x/y arrays and a separate
boolean validity mask, all of shape `(513,)`. Invalid y entries contain NaN;
the mask determines which entries are usable. These arrays are not cached or
shared between calls. Read-only flags prevent accidental writes through the
normal interface; they are not a deep-immutability guarantee. Compiled viewer
evaluators are cached, with real input grids and complex output conversion used
only to validate real, finite plotted values.

The WebSocket boundary converts NumPy arrays/scalars into Python/JSON lists and
numbers. Invalid entries become JSON `null`, so NaN/infinity never leak into
messages. Incoming JSON viewport values are validated and canonicalized before
they enter viewer state. Failed evaluation preserves the previous successful state.

[`numpy.typing`](https://numpy.org/doc/stable/reference/typing.html) describes
scalar/array dtypes, and ty checks the API overloads. Argument counts, broadcast
shapes, bounds, dtypes, finiteness, representability, and ownership are enforced
explicitly at the relevant runtime boundary. Type annotations alone establish
neither shape guarantees nor mathematical domain membership.
