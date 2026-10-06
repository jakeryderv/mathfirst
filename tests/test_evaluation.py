"""Explicit numerical boundaries preserve mathematics and array ownership."""

from fractions import Fraction
from typing import assert_type

import numpy as np
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from mathfirst import Expression, Function, Point, Polynomial, Scalar, Set, Variable
from mathfirst.numerical import RealArray


def test_numerical_evaluation_is_separate_from_exact_substitution():
    x = Variable("x")
    function = Polynomial(x**2 + Scalar(Fraction(1, 3)), x)
    exact = function(2)
    assert isinstance(exact, Scalar) and exact.is_exact
    assert exact.to_sympy() == sp.Rational(13, 3)
    numerical = function.evaluate_numpy(2)
    assert_type(numerical, np.float64 | RealArray)
    assert isinstance(numerical, np.float64)
    assert numerical == pytest.approx(13 / 3)
    assert function(2) == exact
    assert isinstance(function.evaluate_numpy(np.array(2)), np.float64)


def test_multivariable_broadcasting_and_constant_result_shape():
    x, y = Variable("x"), Variable("y")
    left = np.arange(3)[:, None]
    right = np.arange(4)[None, :]
    function = Function(x - 2 * y, (x, y))
    result = function.evaluate_numpy(left, right)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4) and result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, left - 2 * right)
    constant = Function(Scalar(7), (x, y)).evaluate_numpy(left, right)
    assert isinstance(constant, np.ndarray)
    assert constant.shape == (3, 4) and constant.flags.writeable
    np.testing.assert_array_equal(constant, np.full((3, 4), 7))
    # An unused argument still contributes to the requested broadcast shape.
    np.testing.assert_array_equal(
        Function(x, (x, y)).evaluate_numpy(left, right),
        np.broadcast_to(left, (3, 4)),
    )


@pytest.mark.parametrize("constant", [False, True])
def test_empty_arrays_keep_broadcast_shape(constant):
    x, y = Variable("x"), Variable("y")
    function = Function(Scalar(2) if constant else x + y, (x, y))
    result = function.evaluate_numpy(np.empty((0, 1)), np.ones((1, 3)))
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3) and result.dtype == np.dtype(np.float64)


def test_zero_argument_numerical_function():
    assert Function(Scalar(3), ()).evaluate_numpy() == np.float64(3)


def test_large_exact_constant_uses_explicit_scalar_conversion():
    x = Variable("x")
    function = Function(Scalar(2**100), (x,))
    np.testing.assert_array_equal(function.evaluate_numpy([0, 1]), [float(2**100)] * 2)
    with pytest.raises(OverflowError):
        Function(Scalar(10**400), (x,)).evaluate_numpy(0, allow_nonfinite=True)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128, ">f8"]
)
def test_requested_dtype_is_used_for_inputs_and_results(dtype):
    x = Variable("x")
    function = Function(x / 2, (x,))
    result = function.evaluate_numpy([1, 3], dtype=dtype)
    assert isinstance(result, np.ndarray) and result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, [0.5, 1.5])
    assert isinstance(function.evaluate_numpy(3, dtype=dtype), np.dtype(dtype).type)


def test_complex_evaluation_is_explicit():
    x = Variable("x")
    function = Function(Expression(sp.sqrt(x.to_sympy())), (x,))
    with pytest.raises(ValueError, match="Nonfinite"):
        function.evaluate_numpy(-1)
    assert np.isnan(function.evaluate_numpy(-1, allow_nonfinite=True))
    assert function.evaluate_numpy(-1, dtype=np.complex128) == np.complex128(1j)
    with pytest.raises(ValueError, match="imaginary"):
        Function(x, (x,)).evaluate_numpy(1j)
    with pytest.raises(ValueError, match="imaginary"):
        Function(Expression(sp.I), (x,)).evaluate_numpy(0)
    assert Function(Expression(sp.I), (x,)).evaluate_numpy(0, dtype=np.complex128) == 1j


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_nonfinite_input_requires_opt_in_even_for_constant_function(value):
    x = Variable("x")
    function = Function(Scalar(2), (x,))
    with pytest.raises(ValueError, match="Nonfinite"):
        function.evaluate_numpy(value)
    assert function.evaluate_numpy(value, allow_nonfinite=True) == 2


def test_nonfinite_output_and_arithmetic_overflow_policy():
    x = Variable("x")
    pole = Function(1 / x, (x,))
    with pytest.raises(ValueError, match="Nonfinite"):
        pole.evaluate_numpy([0, 1])
    np.testing.assert_array_equal(
        pole.evaluate_numpy([0, 1], allow_nonfinite=True), [np.inf, 1]
    )
    square = Function(x**2, (x,))
    with pytest.raises(ValueError, match="Nonfinite"):
        square.evaluate_numpy(1e200)
    assert np.isinf(square.evaluate_numpy(1e200, allow_nonfinite=True))
    # Conversion overflow is rejected even when arithmetic nonfinites are allowed.
    with pytest.raises(OverflowError):
        square.evaluate_numpy(1e100, dtype=np.float32, allow_nonfinite=True)


def test_argument_count_shapes_dtypes_and_options_are_validated():
    x, y = Variable("x"), Variable("y")
    function = Function(x + y, (x, y))
    with pytest.raises(ValueError, match="Expected 2"):
        function.evaluate_numpy(1)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        function.evaluate_numpy(np.zeros(2), np.zeros(3))
    for dtype in (np.int64, np.bool_, object):
        with pytest.raises(TypeError, match="dtype"):
            function.evaluate_numpy(1, 2, dtype=dtype)
    with pytest.raises(TypeError, match="Python boolean"):
        function.evaluate_numpy(1, 2, allow_nonfinite=np.bool_(True))  # ty: ignore[invalid-argument-type] -- runtime control validation


@pytest.mark.parametrize(
    "value",
    [
        True,
        [1, False],
        ["1"],
        np.array([1], dtype=object),
        np.ma.array([1], mask=[True]),
    ],
)
def test_invalid_numerical_inputs_are_rejected(value):
    x = Variable("x")
    with pytest.raises(TypeError):
        Function(x, (x,)).evaluate_numpy(value)


def test_symbolic_inputs_require_explicit_conversion():
    x = Variable("x")
    with pytest.raises(TypeError):
        Function(x, (x,)).evaluate_numpy(Scalar(2))  # ty: ignore[invalid-argument-type] -- explicit mathematical conversion required
    assert Function(x, (x,)).evaluate_numpy(Scalar(2).to_numpy()) == 2


def test_unsupported_symbolic_functions_report_backend_error():
    x = Variable("x")
    unknown = Expression(sp.Function("unknown")(x.to_sympy()))  # ty: ignore[call-non-callable] -- SymPy callable function class
    with pytest.raises(ValueError, match="NumPy backend"):
        Function(unknown, (x,)).evaluate_numpy(1)


def test_unexpected_backend_result_shape_is_rejected(monkeypatch):
    x = Variable("x")
    monkeypatch.setattr(
        sp, "lambdify", lambda *args, **kwargs: lambda *values: np.ones((2, 3))
    )
    with pytest.raises(ValueError, match=r"result.*shape"):
        Function(x, (x,)).evaluate_numpy(np.ones(3))


def test_function_results_are_owned_writable_and_uncached():
    x = Variable("x")
    source = np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::2]
    source.setflags(write=False)
    function = Function(x, (x,))
    result = function.evaluate_numpy(source)
    assert isinstance(result, np.ndarray)
    assert result.flags.writeable and not np.shares_memory(source, result)
    result[:] = -10
    np.testing.assert_array_equal(function.evaluate_numpy(source), [[0, 2], [3, 5]])
    np.testing.assert_array_equal(source, [[0, 2], [3, 5]])
    assert function.expression is x


def test_numerical_evaluation_does_not_enforce_descriptive_sets():
    x = Variable("x")
    function = Function(x + 1, (x,), Set.interval(0, 1), Set.integers())
    assert function.evaluate_numpy(2.5) == 3.5


@settings(max_examples=40, deadline=None)
@given(
    a=st.integers(-100, 100),
    b=st.integers(-100, 100),
    c=st.integers(-100, 100),
    values=st.lists(st.integers(-100, 100), max_size=20),
)
def test_polynomial_numerical_evaluation_matches_arithmetic(a, b, c, values):
    x = Variable("x")
    result = Polynomial(a * x**2 + b * x + c, x).evaluate_numpy(values)
    np.testing.assert_array_equal(
        result, [a * value**2 + b * value + c for value in values]
    )


def test_point_conversion_shape_exact_source_and_ownership():
    point = Point(Fraction(1, 3), 2)
    source = point.to_sympy()
    result = point.to_numpy()
    assert_type(result, RealArray)
    assert result.shape == (point.dimension,) and result.dtype == np.dtype(np.float64)
    assert result.flags.writeable
    np.testing.assert_array_equal(result, [1 / 3, 2])
    result[:] = 0
    assert point.to_sympy() == source
    np.testing.assert_array_equal(point.to_numpy(), [1 / 3, 2])
    assert point.to_numpy(dtype=np.float32).dtype == np.dtype(np.float32)


def test_point_conversion_handles_complex_integer_and_nonfinite_values():
    point = Point(sp.I, 2)
    with pytest.raises(ValueError, match="imaginary"):
        point.to_numpy()
    np.testing.assert_array_equal(point.to_numpy(dtype=np.complex128), [1j, 2])
    largest = 2**64 - 1
    assert Point(largest).to_numpy(dtype=np.uint64)[0] == largest
    with pytest.raises(OverflowError):
        Point(largest).to_numpy(dtype=np.int64)
    with pytest.raises(ValueError, match="proven integer"):
        Point(Fraction(1, 3)).to_numpy(dtype=np.int64)
    with pytest.raises(ValueError, match="Nonfinite"):
        Point(sp.oo).to_numpy()
    assert np.isinf(Point(sp.oo).to_numpy(allow_nonfinite=True)[0])


def test_point_conversion_rejects_unresolved_coordinates_and_invalid_options():
    with pytest.raises(ValueError, match="free variables"):
        Point(Variable("x")).to_numpy()
    with pytest.raises(TypeError, match="boolean"):
        Point(1).to_numpy(allow_nonfinite=1)  # ty: ignore[invalid-argument-type] -- runtime option validation
    with pytest.raises(TypeError, match="dtype"):
        Point(1).to_numpy(dtype=object)
