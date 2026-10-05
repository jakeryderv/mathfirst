import pytest
import sympy as sp

from mathfirst import Function, Scalar, Set, Variable


def test_function_uses_simultaneous_substitution():
    x, y = Variable("x"), Variable("y")
    function = Function(x - y, (x, y))
    assert function(y, x).to_sympy() == y.to_sympy() - x.to_sympy()
    assert function(3, 1) == 2
    assert isinstance(function(3, 1), Scalar)
    assert function.to_sympy() == sp.Lambda(
        (x.to_sympy(), y.to_sympy()), x.to_sympy() - y.to_sympy()
    )


def test_function_sets_are_descriptive_metadata():
    x = Variable("x")
    domain, codomain = Set.interval(0, 1), Set.reals()
    function = Function(x + 1, (x,), domain, codomain)
    assert function.domain is domain
    assert function.codomain is codomain
    assert function(2) == 3
    assert domain.contains(2) is False


def test_function_rejects_duplicate_and_undeclared_variables():
    x, y = Variable("x"), Variable("y")
    with pytest.raises(ValueError, match="distinct"):
        Function(x, (x, Variable("x")))
    with pytest.raises(ValueError, match="undeclared"):
        Function(x + y, (x,))


@pytest.mark.parametrize("values", [(), (1, 2)])
def test_function_checks_argument_count(values):
    x = Variable("x")
    with pytest.raises(ValueError, match="Expected 1"):
        Function(x, (x,))(*values)


def test_function_validates_construction():
    x = Variable("x")
    with pytest.raises(TypeError, match="Expression"):
        Function(1, (x,))  # ty: ignore[invalid-argument-type] -- invalid expression
    with pytest.raises(TypeError, match="tuple"):
        Function(x, [x])  # ty: ignore[invalid-argument-type] -- require tuple
    with pytest.raises(TypeError, match="domain"):
        Function(x, (x,), domain=1)  # ty: ignore[invalid-argument-type] -- invalid set
    with pytest.raises(TypeError, match="codomain"):
        Function(x, (x,), codomain=1)  # ty: ignore[invalid-argument-type] -- invalid set
