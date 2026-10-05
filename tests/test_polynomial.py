import pytest
import sympy as sp

from mathfirst import Polynomial, Scalar, Set, Variable


def test_polynomial_properties_and_evaluation():
    x = Variable("x")
    polynomial = Polynomial(3 * x**2 - 2 * x + 1, x)
    assert polynomial.variable is x
    assert polynomial.coefficients == (3, -2, 1)
    assert tuple(term.to_sympy() for term in polynomial.terms) == (
        3 * x.to_sympy() ** 2,
        -2 * x.to_sympy(),
        sp.Integer(1),
    )
    assert polynomial.degree == 2
    assert polynomial.leading_coefficient == 3
    assert polynomial(2) == 9
    assert polynomial.to_sympy() == sp.Poly(
        3 * x.to_sympy() ** 2 - 2 * x.to_sympy() + 1, x.to_sympy()
    )
    assert polynomial.domain is not None
    assert polynomial.codomain is not None
    assert polynomial.domain.to_sympy() == sp.S.Reals
    assert polynomial.codomain.to_sympy() == sp.S.Reals


def test_constant_and_zero_polynomials():
    x = Variable("x")
    assert Polynomial(7, x).degree == 0
    assert Polynomial(7, x).coefficients == (7,)
    zero = Polynomial(0, x)
    assert zero.degree == Scalar(-sp.oo)
    assert zero.coefficients == (0,)
    assert zero(2) == 0


def test_polynomial_preserves_supplied_sets():
    x = Variable("x")
    domain, codomain = Set.interval(0, 1), Set.interval(0, 2)
    polynomial = Polynomial(x + 1, x, domain=domain, codomain=codomain)
    assert polynomial.domain is domain
    assert polynomial.codomain is codomain


@pytest.mark.parametrize("kind", ["reciprocal", "complex", "other_variable"])
def test_polynomial_rejects_invalid_expressions(kind):
    x = Variable("x")
    expressions = {
        "reciprocal": 1 / x,
        "complex": sp.I * x.to_sympy(),
        "other_variable": x + Variable("y"),
    }
    with pytest.raises(ValueError):
        Polynomial(expressions[kind], x)
