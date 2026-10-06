"""Verify concrete representation, conversion, and array ownership contracts."""

from typing import assert_type

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from mathfirst import Scalar
from mathfirst.numerical import (
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_REAL_DTYPE,
    as_numeric_array,
)


def test_mathfirst_defaults_are_explicit():
    assert np.dtype(np.float64) == DEFAULT_REAL_DTYPE
    assert np.dtype(np.complex128) == DEFAULT_COMPLEX_DTYPE
    result = as_numeric_array([1, 2, 3])
    assert result.dtype == DEFAULT_REAL_DTYPE
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_static_return_contracts():
    assert_type(as_numeric_array([1, 2]), NDArray[np.float64])
    assert_type(as_numeric_array([1, 2], dtype=np.float32), NDArray[np.float32])
    assert_type(as_numeric_array([1j], dtype=np.complex128), NDArray[np.complex128])
    assert_type(Scalar(1).to_numpy(), np.float64 | np.complex128)
    assert_type(Scalar(1).to_numpy(dtype=np.int64), np.int64)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        "float32",
        np.dtype(">f8"),
    ],
)
def test_explicit_floating_dtype(dtype):
    result = as_numeric_array([1.25, -2.5], dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, [1.25, -2.5])


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_limits_remain_exact(dtype):
    limits = np.iinfo(dtype)
    values = np.array([limits.min, limits.max], dtype=dtype)
    result = as_numeric_array(values, dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([-1], np.uint8),
        ([256], np.uint8),
        ([128], np.int8),
        (np.array([2**64 - 1], dtype=np.uint64), np.int64),
    ],
)
def test_integer_overflow_is_rejected(values, dtype):
    with pytest.raises(OverflowError, match="range"):
        as_numeric_array(values, dtype=dtype)


@pytest.mark.parametrize("values", [[1.0], [1.5], [1 + 0j]])
def test_integer_conversion_does_not_truncate_or_exactify(values):
    with pytest.raises(ValueError, match="integer input"):
        as_numeric_array(values, dtype=np.int64)


@pytest.mark.parametrize(
    "values",
    [
        [True, 1],
        [np.bool_(False), 2],
        [[1, True]],
        np.array([True]),
        ["1"],
        np.array([1], dtype=object),
        np.ma.array([1.0], mask=[True]),
    ],
)
def test_non_numeric_or_ambiguous_input_is_rejected(values):
    with pytest.raises(TypeError):
        as_numeric_array(values)


@pytest.mark.parametrize(
    "dtype", [bool, object, str, "datetime64[ns]", np.longdouble, np.clongdouble]
)
def test_unsupported_output_dtypes_are_rejected(dtype):
    with pytest.raises(TypeError, match="dtype"):
        as_numeric_array([1], dtype=dtype)


def test_complex_conversion_preserves_components():
    result = as_numeric_array([1 + 2j, 3 - 4j], dtype=DEFAULT_COMPLEX_DTYPE)
    assert result.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(result, [1 + 2j, 3 - 4j])
    with pytest.raises(ValueError, match="imaginary"):
        as_numeric_array([1 + 2j])
    np.testing.assert_array_equal(as_numeric_array([1 + 0j]), [1.0])


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_values_require_opt_in(value):
    with pytest.raises(ValueError, match="allow_nonfinite"):
        as_numeric_array([value])
    result = as_numeric_array([value], allow_nonfinite=True)
    np.testing.assert_array_equal(result, [value])


@pytest.mark.parametrize("allow_nonfinite", [False, True])
@pytest.mark.parametrize(
    "value, dtype",
    [
        (1e100, np.float32),
        (1e100j, np.complex64),
        (complex(float("inf"), 1e100), np.complex64),
    ],
)
def test_float_overflow_is_distinct_from_allowed_nonfinite_input(
    value, dtype, allow_nonfinite
):
    expected = OverflowError if np.isfinite(value) or allow_nonfinite else ValueError
    with pytest.raises(expected):
        as_numeric_array([value], dtype=dtype, allow_nonfinite=allow_nonfinite)


def test_default_copy_is_independent_and_writable():
    original = np.array([1.0, 2.0])
    original.flags.writeable = False
    result = as_numeric_array(original)
    assert not np.shares_memory(result, original)
    assert result.flags.owndata and result.flags.writeable
    result[0] = 9
    assert original[0] == 1


def test_no_copy_preserves_sharing_and_writeability():
    original = np.array([1.0, 2.0, 3.0])
    view = original[::2]
    result = as_numeric_array(view, copy=False)
    assert np.shares_memory(result, original)
    result[0] = 9
    assert original[0] == 9
    view.flags.writeable = False
    readonly = as_numeric_array(view, copy=False)
    assert not readonly.flags.writeable


@pytest.mark.parametrize("values", [[1.0], np.array([1], dtype=np.int64)])
def test_no_copy_rejects_inputs_requiring_allocation(values):
    with pytest.raises(ValueError, match="copy=False"):
        as_numeric_array(values, copy=False)


@pytest.mark.parametrize("options", [{"copy": np.bool_(True)}, {"allow_nonfinite": 1}])
def test_control_options_require_python_booleans(options):
    with pytest.raises(TypeError, match="Python booleans"):
        as_numeric_array([1.0], **options)


def test_empty_arrays_and_underflow_have_defined_behavior():
    assert as_numeric_array([], dtype=np.int64).shape == (0,)
    assert as_numeric_array([1e-100], dtype=np.float32)[0] == 0


def test_multidimensional_arrays_preserve_shape_and_signed_zero():
    values = np.array([[0.0, -0.0], [1.0, 2.0]])
    result = as_numeric_array(values)
    assert result.shape == values.shape
    np.testing.assert_array_equal(np.signbit(result), np.signbit(values))


@given(
    st.lists(st.floats(width=64, allow_nan=False, allow_infinity=False), max_size=30)
)
def test_float64_canonicalization_preserves_input_values(values):
    result = as_numeric_array(values)
    assert result.dtype == DEFAULT_REAL_DTYPE
    np.testing.assert_array_equal(result, values)
