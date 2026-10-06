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
| Numerical validity masks | `NDArray[np.bool_]`, distinct from numerical magnitudes |

`DEFAULT_REAL_DTYPE` and `DEFAULT_COMPLEX_DTYPE` in `mathfirst.numerical` declare
the defaults. Other domains can declare their own dtypes. There is no package-wide
fixed-width integer default: exact integers remain unbounded, and callers requesting
concrete integer output choose its width and signedness. Names, control booleans,
dimensions, multiplicities, and implementation counters retain their existing
semantic or exact representations.

NumPy and SymPy are core dependencies. The optional `viz` extra supplies the web
runtime. This slice does not add numerical function evaluation or change the viewer's
sampling implementation. `Function.__call__()` continues to perform symbolic
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
