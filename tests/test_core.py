import pytest
import sympy as sp

from mathfirst import Expression, Point, Root, Scalar, Set, Variable


def test_expression_arithmetic_and_structural_equality():
    x = Variable("x")
    assert (2 + x - 1).to_sympy() == x.to_sympy() + 1
    assert (3 * x / 2).to_sympy() == 3 * x.to_sympy() / 2
    assert (2 / x).to_sympy() == 2 / x.to_sympy()
    assert (2**x).to_sympy() == 2 ** x.to_sympy()
    assert (-x).to_sympy() == -x.to_sympy()
    assert isinstance(x - x, Scalar)
    assert (x + 1) ** 2 != x**2 + 2 * x + 1
    assert Scalar(1) != True  # noqa: E712 -- bool must not compare as a scalar
    with pytest.raises(TypeError, match="unhashable"):
        hash(x)


@pytest.mark.parametrize("name", ["", None, 1])
def test_variable_rejects_invalid_names(name):
    with pytest.raises((TypeError, ValueError)):
        Variable(name)


def test_variable_identity_is_name_based():
    assert Variable("x") == Variable("x")
    assert Variable("x") != Variable("y")
    assert Variable("x").to_sympy().is_real is True


def test_point_preserves_order_and_symbolic_coordinates():
    x = Variable("x")
    point = Point(x, 2, Scalar.exact("0.1"))
    assert point.dimension == 3
    assert point.coordinates == (x, 2, sp.Rational(1, 10))
    assert point.to_sympy() == sp.Tuple(x.to_sympy(), 2, sp.Rational(1, 10))
    with pytest.raises(ValueError, match="at least one"):
        Point()


def test_set_membership_and_open_endpoints():
    interval = Set.interval(0, 1, left_open=True)
    assert interval.contains(0) is False
    assert interval.contains(1) is True
    assert interval.contains(Scalar.exact("0.5")) is True
    assert interval.contains(Variable("x")) is None
    assert Set.integers().contains(Scalar.exact("0.5")) is False
    assert Set.reals().contains(Variable("x")) is True


def test_cartesian_product_membership():
    product = Set.product(Set.integers(), Set.interval(0, 1))
    assert product.contains((2, Scalar.exact("0.5"))) is True
    assert product.contains((2, 3)) is False
    with pytest.raises(TypeError, match="factors"):
        Set.product(Set.reals(), sp.S.Reals)


@pytest.mark.parametrize("multiplicity", [0, -1, True, 1.5])
def test_root_rejects_invalid_multiplicity(multiplicity):
    with pytest.raises((TypeError, ValueError)):
        Root(Scalar(1), multiplicity)


def test_root_representation_and_value_validation():
    assert Root(Scalar(2), 3).to_sympy() == sp.Tuple(2, 3)
    with pytest.raises(TypeError, match="Expression"):
        Root(2)  # ty: ignore[invalid-argument-type] -- test runtime validation
    with pytest.raises(TypeError, match="scalar expression"):
        Expression(sp.S.Reals)
    with pytest.raises(TypeError, match="SymPy Set"):
        Set(1)  # ty: ignore[invalid-argument-type] -- test runtime validation
