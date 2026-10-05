import pytest
import sympy as sp

from mathfirst import (
    Function,
    Polynomial,
    Scalar,
    Set,
    Variable,
    derivative,
    limit,
    real_roots,
)


def test_derivative_preserves_polynomial_structure():
    x = Variable("x")
    result = derivative(Polynomial(x**3 - 2 * x + 7, x))
    assert isinstance(result, Polynomial)
    assert result.coefficients == (3, 0, -2)
    assert result.variable is x
    assert result.domain is not None
    assert result.domain.to_sympy() == sp.S.Reals


def test_general_partial_derivative_and_explicit_sets():
    x, y = Variable("x"), Variable("y")
    function = Function(
        x**2 * y, (x, y), Set.product(Set.reals(), Set.reals()), Set.reals()
    )
    result = derivative(function, x)
    assert result(2, 3) == 12
    assert result.variables == (x, y)
    assert result.domain is None
    assert result.codomain is None
    domain, codomain = Set.product(Set.reals(), Set.reals()), Set.reals()
    explicit = derivative(function, y, domain=domain, codomain=codomain)
    assert explicit(2, 3) == 4
    assert explicit.domain is domain
    assert explicit.codomain is codomain
    with pytest.raises(ValueError, match="multivariable"):
        derivative(function)
    with pytest.raises(ValueError, match="does not belong"):
        derivative(function, Variable("z"))


def test_limit_removable_discontinuity_and_infinity():
    x = Variable("x")
    assert limit(Function((x**2 - 1) / (x - 1), (x,)), 1) == 2
    assert limit(Function(1 / x, (x,)), Scalar(sp.oo)) == 0


def test_limit_sides_and_nonexistent_two_sided_limit():
    x = Variable("x")
    function = Function(1 / x, (x,))
    assert limit(function, 0, direction="+") == sp.oo
    assert limit(function, 0, direction="-") == -sp.oo
    with pytest.raises(ValueError):
        limit(function, 0)
    with pytest.raises(ValueError, match="direction"):
        limit(function, 0, direction="invalid")  # ty: ignore[invalid-argument-type] -- invalid direction


def test_exact_real_roots_are_sorted_with_multiplicity():
    x = Variable("x")
    roots = real_roots(Polynomial((x - 1) ** 3 * (x + 2) ** 2 * (x**2 + 1), x))
    assert tuple((root.value, root.multiplicity) for root in roots) == ((-2, 2), (1, 3))
    irrational = real_roots(Polynomial(x**2 - 2, x))
    assert tuple(root.value.to_sympy() for root in irrational) == (
        -sp.sqrt(2),
        sp.sqrt(2),
    )
    for root in irrational:
        assert isinstance(root.value, Scalar)
        assert root.value.is_exact


def test_constant_zero_and_non_polynomial_root_inputs():
    x = Variable("x")
    assert real_roots(Polynomial(3, x)) == ()
    assert real_roots(Polynomial(x**2 + 1, x)) == ()
    with pytest.raises(ValueError, match="infinitely many"):
        real_roots(Polynomial(0, x))
    with pytest.raises(TypeError, match="Polynomial"):
        real_roots(Function(x, (x,)))  # ty: ignore[invalid-argument-type] -- require polynomial
