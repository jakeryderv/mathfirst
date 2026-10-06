"""Check exact mathematics against independent Python arithmetic."""

from collections import Counter
from fractions import Fraction

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mathfirst import Expression, Polynomial, Scalar, Variable, real_roots

RATIONALS = st.builds(
    Fraction,
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=1, max_value=1000),
)


@given(value=RATIONALS)
@example(value=Fraction(71, 179))
def test_scalar_numpy_conversion_matches_fraction(value: Fraction) -> None:
    scalar = Scalar.exact(value)
    source = scalar.to_sympy()
    result = scalar.to_numpy(dtype=np.float64)
    assert isinstance(result, np.float64)
    assert result == float(value)
    assert scalar.is_exact and scalar.to_sympy() is source


def _as_fraction(expression: Expression) -> Fraction:
    assert isinstance(expression, Scalar)
    assert expression.is_exact
    numerator, denominator = expression.to_sympy().as_numer_denom()
    return Fraction(int(numerator), int(denominator))


@given(left=RATIONALS, right=RATIONALS)
def test_exact_scalar_arithmetic_matches_fraction(
    left: Fraction, right: Fraction
) -> None:
    a, b = Scalar.exact(left), Scalar.exact(right)
    assert _as_fraction(a + b) == left + right
    assert _as_fraction(a - b) == left - right
    assert _as_fraction(a * b) == left * right
    if right != 0:
        assert _as_fraction(a / b) == left / right


# Bound symbolic work rather than imposing a machine-dependent time deadline.
@settings(max_examples=50, deadline=None)
@given(coefficients=st.lists(RATIONALS, min_size=1, max_size=5), value=RATIONALS)
@example(coefficients=[Fraction(0), Fraction(0)], value=Fraction(2))
def test_polynomial_evaluation_matches_horner(
    coefficients: list[Fraction], value: Fraction
) -> None:
    x = Variable("x")
    expression: Expression = Scalar(0)
    for power, coefficient in enumerate(reversed(coefficients)):
        expression += Scalar(coefficient) * x**power
    polynomial = Polynomial(expression, x)

    expected = Fraction(0)
    for coefficient in coefficients:
        expected = expected * value + coefficient

    assert _as_fraction(polynomial(Scalar(value))) == expected


@settings(max_examples=50, deadline=None)
@given(
    factors=st.lists(st.integers(min_value=-5, max_value=5), max_size=5),
    leading=st.one_of(st.integers(-5, -1), st.integers(1, 5)),
)
@example(factors=[], leading=1)
@example(factors=[0, 0, -2], leading=-3)
def test_factored_polynomial_roots_and_multiplicities(
    factors: list[int], leading: int
) -> None:
    x = Variable("x")
    expression: Expression = Scalar(leading)
    for root in factors:
        expression *= x - root
    polynomial = Polynomial(expression, x)

    actual = tuple(
        (_as_fraction(root.value), root.multiplicity) for root in real_roots(polynomial)
    )
    expected = tuple(sorted(Counter(factors).items()))
    assert actual == expected
