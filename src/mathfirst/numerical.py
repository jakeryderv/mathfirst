"""MathFirst's concrete NumPy contracts and explicit canonicalization.

The defaults belong to MathFirst's numerical conversion domain. Other domains
may declare different dtypes. Exact and symbolic objects retain their source
representations until a numerical conversion is requested.
"""

from typing import Any, Final, cast, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = [
    "DEFAULT_COMPLEX_DTYPE",
    "DEFAULT_REAL_DTYPE",
    "ComplexArray",
    "MaskArray",
    "NumericArray",
    "NumericInput",
    "NumericScalar",
    "RealArray",
    "as_numeric_array",
]

type NumericScalar = np.integer[Any] | np.floating[Any] | np.complexfloating[Any, Any]
type NumericInput = int | float | complex | NumericScalar
type NumericArray = NDArray[NumericScalar]
type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type MaskArray = NDArray[np.bool_]

DEFAULT_REAL_DTYPE: Final = np.dtype(np.float64)
DEFAULT_COMPLEX_DTYPE: Final = np.dtype(np.complex128)

_SUPPORTED_DTYPES = tuple(
    np.dtype(scalar)
    for scalar in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
)


def _numeric_dtype(dtype: DTypeLike) -> np.dtype[NumericScalar]:
    result = np.dtype(dtype)
    if result.newbyteorder("=") not in _SUPPORTED_DTYPES:
        raise TypeError(
            "dtype must be a supported integer, float16/32/64, or complex64/128"
        )
    return cast(np.dtype[NumericScalar], result)


def _check_options(*, copy: bool, allow_nonfinite: bool) -> None:
    if not isinstance(copy, bool) or not isinstance(allow_nonfinite, bool):
        raise TypeError("copy and allow_nonfinite must be Python booleans")


def _reject_booleans(value: object) -> None:
    if isinstance(value, (bool, np.bool_)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == "b"
    ):
        raise TypeError("Boolean values are not numerical magnitudes")
    if isinstance(value, (list, tuple)):
        for element in value:
            _reject_booleans(element)


@overload
def as_numeric_array(
    values: ArrayLike,
    *,
    dtype: None = None,
    copy: bool = True,
    allow_nonfinite: bool = False,
) -> RealArray: ...


@overload
def as_numeric_array[NumericT: NumericScalar](
    values: ArrayLike,
    *,
    dtype: type[NumericT] | np.dtype[NumericT],
    copy: bool = True,
    allow_nonfinite: bool = False,
) -> NDArray[NumericT]: ...


@overload
def as_numeric_array(
    values: ArrayLike,
    *,
    dtype: DTypeLike,
    copy: bool = True,
    allow_nonfinite: bool = False,
) -> NumericArray: ...


def as_numeric_array(
    values: ArrayLike,
    *,
    dtype: DTypeLike | None = None,
    copy: bool = True,
    allow_nonfinite: bool = False,
) -> NumericArray:
    """Normalize numerical inputs to an explicitly typed NumPy array.

    The default dtype is MathFirst's float64. Floating casts may round or
    underflow; overflow raises OverflowError. Integer output requires integer
    input and a representable range. Real output rejects nonzero imaginary
    components. Nonfinite input requires explicit opt-in. Boolean, string,
    object, and masked data are rejected.

    The default returns an independent, writable array. With copy=False, the
    input must already be an ndarray of the requested dtype: returned data may
    share storage and mutations, and retains the input's writeability. No shape
    or domain-specific invariant is established by this conversion.
    """
    _check_options(copy=copy, allow_nonfinite=allow_nonfinite)
    target = _numeric_dtype(DEFAULT_REAL_DTYPE if dtype is None else dtype)
    _reject_booleans(values)
    if np.ma.isMaskedArray(values):
        raise TypeError("Masked arrays require an explicit validity policy")
    source = np.asarray(values)
    if source.dtype.kind not in "iufc":
        raise TypeError("Inputs must contain numerical values, excluding booleans")
    if not copy and (not isinstance(values, np.ndarray) or source.dtype != target):
        raise ValueError("copy=False requires an ndarray with the requested dtype")
    if not allow_nonfinite and not np.isfinite(source).all():
        raise ValueError("Nonfinite values require allow_nonfinite=True")
    if target.kind in "iu":
        if source.size and source.dtype.kind not in "iu":
            raise ValueError("Integer output requires integer input")
        limits = np.iinfo(target.str)
        if source.size and (
            int(source.min()) < limits.min or int(source.max()) > limits.max
        ):
            raise OverflowError(f"Values are outside the range of {target}")
    if target.kind == "f" and source.dtype.kind == "c":
        if np.any(source.imag != 0):
            raise ValueError("Real output cannot discard imaginary components")
        source = source.real
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(source, dtype=target, copy=copy)
    for before, after in ((source.real, result.real), (source.imag, result.imag)):
        if np.any(np.isfinite(before) & ~np.isfinite(after)):
            raise OverflowError(f"Finite values overflow {target}")
    return result
