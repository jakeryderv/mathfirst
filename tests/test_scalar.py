import warnings
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from mathfirst.core import ApproximateScalarWarning, Scalar, Variable
from mathfirst.function import Function
from mathfirst.polynomial import Polynomial


@pytest.mark.parametrize(
    "value", [np.int8(-3), np.int64(2**63 - 1), np.uint64(2**64 - 1)]
)
def test_numpy_integers_preserve_exact_values(value):
    scalar = Scalar.exact(value)
    assert scalar.is_exact and scalar == value
    assert scalar.to_sympy() == sp.Integer(int(value))


@pytest.mark.parametrize(
    "value",
    [
        np.float16("0.1"),
        np.float32("0.1"),
        np.float64("0.1"),
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        1 + 2j,
    ],
)
def test_approximate_scalar_inputs_have_consistent_intent(value):
    with pytest.warns(ApproximateScalarWarning):
        scalar = Scalar(value)
    assert not scalar.is_exact
    with pytest.raises(TypeError, match="cannot infer intent"):
        Scalar.exact(value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        approximate = Scalar.approx(value)
    assert not approximate.is_exact
    assert complex(approximate.to_sympy()) == complex(scalar.to_sympy())


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact, Scalar.approx])
def test_boolean_scalar_inputs_are_rejected(value, constructor):
    with pytest.raises(TypeError, match="Boolean"):
        constructor(value)


@pytest.mark.parametrize("value", [Fraction(1, 3), "0.1", sp.pi, sp.sqrt(2)])
def test_numpy_conversion_is_explicit_and_preserves_source(value):
    scalar = Scalar.exact(value)
    source = scalar.to_sympy()
    result = scalar.to_numpy()
    assert isinstance(result, np.float64)
    assert result == pytest.approx(float(source))
    assert scalar.to_sympy() is source and scalar.is_exact


def test_numpy_conversion_defaults_to_complex128_for_complex_values():
    scalar = Scalar.exact(sp.Rational(1, 3) + 2 * sp.I)
    assert isinstance(scalar.to_numpy(), np.complex128)
    assert scalar.to_numpy() == pytest.approx(1 / 3 + 2j)
    with pytest.raises(ValueError, match="imaginary"):
        scalar.to_numpy(dtype=np.float64)


def test_complex_conversion_rounds_rational_components_directly():
    value = Scalar.exact(sp.Rational(71, 179) + sp.I / 3).to_numpy()
    assert value == complex(float(Fraction(71, 179)), float(Fraction(1, 3)))


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.complex64, np.complex128, "float32"],
)
def test_numpy_scalar_conversion_respects_explicit_dtype(dtype):
    result = Scalar.exact("0.1").to_numpy(dtype=dtype)
    assert isinstance(result, np.dtype(dtype).type)
    assert result == np.dtype(dtype).type(0.1)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_numpy_integer_conversion_checks_exact_range(dtype):
    limits = np.iinfo(dtype)
    for value in (limits.min, limits.max):
        result = Scalar(value).to_numpy(dtype=dtype)
        assert isinstance(result, dtype)
        assert int(result) == value
    for value in (limits.min - 1, limits.max + 1):
        with pytest.raises(OverflowError, match="range"):
            Scalar(value).to_numpy(dtype=dtype)


@pytest.mark.parametrize(
    "scalar", [Scalar.exact("0.5"), Scalar.approx(3.0), Scalar.exact(sp.I)]
)
def test_integer_conversion_does_not_truncate_or_exactify(scalar):
    with pytest.raises(ValueError, match="proven integer"):
        scalar.to_numpy(dtype=np.int64)


@pytest.mark.parametrize("value", [sp.oo, -sp.oo, sp.nan])
def test_scalar_nonfinite_conversion_requires_opt_in(value):
    scalar = Scalar(value)
    with pytest.raises(ValueError, match="allow_nonfinite"):
        scalar.to_numpy()
    result = scalar.to_numpy(allow_nonfinite=True)
    assert isinstance(result, np.float64)
    np.testing.assert_equal(result, float(value))


def test_scalar_conversion_rejects_unevaluable_values_and_complex_infinity():
    unknown = sp.Function("unknown")()  # ty: ignore[call-non-callable] -- SymPy creates a callable function class dynamically
    for value in (sp.zoo, unknown):
        with pytest.raises(ValueError, match="numerical"):
            Scalar(value).to_numpy(allow_nonfinite=True)


@pytest.mark.parametrize("dtype", [bool, object, "datetime64[ns]", np.longdouble])
def test_scalar_conversion_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="dtype"):
        Scalar(1).to_numpy(dtype=dtype)


@pytest.mark.parametrize("allow_nonfinite", [False, True])
def test_scalar_overflow_is_rejected_even_with_nonfinite_opt_in(allow_nonfinite):
    with pytest.raises(OverflowError):
        Scalar(2**2000).to_numpy(allow_nonfinite=allow_nonfinite)
    with pytest.raises(OverflowError):
        Scalar(10**100).to_numpy(dtype=np.float32, allow_nonfinite=allow_nonfinite)


def test_scalar_conversion_underflow_and_control_options():
    assert Scalar.exact("1e-100").to_numpy(dtype=np.float32) == 0
    with pytest.raises(TypeError, match="Python boolean"):
        Scalar(1).to_numpy(allow_nonfinite=np.bool_(True))  # ty: ignore[invalid-argument-type] -- runtime control validation


def test_numpy_inputs_work_in_symbolic_arithmetic_and_substitution():
    x = Variable("x")
    function = Function(x + np.int64(1), (x,))
    result = function(np.int64(2))
    assert isinstance(result, Scalar) and result.is_exact and result == 3
    assert isinstance(function(x), type(x + 1))
    assert Polynomial(np.int64(3) * x, x)(np.int64(2)) == 6


@pytest.mark.parametrize("value", [0, 3, -7, 2**80 + 1])
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_integers_are_exact(value, constructor):
    scalar = constructor(value)

    assert isinstance(scalar.to_sympy(), sp.Integer)
    assert scalar.to_sympy() == sp.Integer(value)
    assert scalar.is_exact


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(-7, 12), Fraction(2, 4)])
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_fractions_are_exact(value, constructor):
    scalar = constructor(value)

    assert scalar.to_sympy() == sp.Rational(value.numerator, value.denominator)
    assert scalar == value
    assert scalar.is_exact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.1", sp.Rational(1, 10)),
        ("-2.50", sp.Rational(-5, 2)),
        ("1e-3", sp.Rational(1, 1000)),
        ("0.123456789012345678901", sp.Rational(123456789012345678901, 10**21)),
    ],
)
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_decimal_strings_are_exact(text, expected, constructor):
    scalar = constructor(text)

    assert scalar.to_sympy() == expected
    assert scalar.is_exact


def test_exact_decimal_arithmetic_stays_exact():
    result = Scalar("0.1") + Scalar.exact("0.2")

    assert isinstance(result, Scalar)
    assert result.to_sympy() == sp.Rational(3, 10)
    assert result.is_exact


@pytest.mark.parametrize("value", [2, Fraction(1, 3), "0.1", sp.pi, sp.Float("0.1")])
def test_unambiguous_or_existing_backend_inputs_do_not_warn(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Scalar(value)


def test_implicit_float_warns_at_the_call_site():
    with pytest.warns(ApproximateScalarWarning) as captured:
        scalar = Scalar(0.1)

    assert len(captured) == 1
    warning = captured[0]
    assert warning.filename == __file__
    assert "Python floats are approximate" in str(warning.message)
    assert 'Scalar.exact("0.1")' in str(warning.message)
    assert "Scalar.approx(0.1)" in str(warning.message)
    assert isinstance(scalar.to_sympy(), sp.Float)
    assert scalar.to_sympy() == sp.Float(0.1)
    assert scalar.to_sympy() != sp.Rational(1, 10)
    assert not scalar.is_exact


@pytest.mark.parametrize("value", [0.0, 0.1, -2.5])
def test_explicit_float_approximation_preserves_input_without_warning(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scalar = Scalar.approx(value)

    assert isinstance(scalar.to_sympy(), sp.Float)
    assert scalar.to_sympy() == sp.Float(value)
    assert not scalar.is_exact


@pytest.mark.parametrize("value", [0, 2, Fraction(1, 3), "0.1", sp.sqrt(2)])
def test_exact_values_can_be_explicitly_approximated(value):
    scalar = Scalar.approx(value)

    assert isinstance(scalar.to_sympy(), sp.Float)
    assert not scalar.is_exact
    assert float(scalar.to_sympy()) == pytest.approx(float(Scalar(value).to_sympy()))


@pytest.mark.parametrize("value", [0.1, 1.0, 1 + 2j])
def test_exact_rejects_python_approximate_inputs(value):
    with pytest.raises(TypeError, match="cannot infer intent"):
        Scalar.exact(value)


@pytest.mark.parametrize(
    "value",
    [sp.Float("0.1"), Scalar.approx(0.1), sp.sqrt(2) + sp.Float("0.1")],
)
def test_exact_rejects_existing_approximate_expressions(value):
    with pytest.raises(ValueError, match="approximate SymPy Floats"):
        Scalar.exact(value)


def test_exact_symbolic_constants_remain_exact():
    assert Scalar.exact(sp.pi).to_sympy() == sp.pi
    assert Scalar.exact(sp.sqrt(2)).is_exact
    assert Scalar.exact("sqrt(2) + 0.1").to_sympy() == sp.sqrt(2) + sp.Rational(1, 10)


@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact, Scalar.approx])
def test_free_variables_are_rejected(constructor):
    with pytest.raises(ValueError, match="free variables"):
        constructor(Variable("x"))


def test_explicit_approximation_does_not_repeat_warnings_in_math_operations():
    x = Variable("x")
    coefficient = Scalar.approx(0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        function_result = Function(coefficient * x, (x,))(Scalar(2))
        polynomial = Polynomial(coefficient * x, x)
        polynomial_result = polynomial(Scalar(2))
        arithmetic_result = coefficient + Scalar(1)

    for result in (function_result, polynomial_result, arithmetic_result):
        assert isinstance(result, Scalar)
        assert not result.is_exact
    assert not polynomial.coefficients[0].is_exact
