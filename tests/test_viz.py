"""Realizations and sampling without a running server or browser."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from mathfirst import Expression, Function, Polynomial, Scalar, Set, Variable
from mathfirst.viz import FunctionGraph, Viewer


def test_graph_preserves_the_source_and_has_no_runtime_state():
    x = Variable("x")
    function = Function(x, (x,), Set.interval(-2, 2), Set.reals())
    graph = FunctionGraph(function)
    assert graph.function is function
    with pytest.raises(FrozenInstanceError):
        graph.function = function  # ty: ignore[invalid-assignment] -- immutable
    viewer = Viewer(graph)
    assert viewer.graph is graph
    with pytest.raises(AttributeError):
        viewer.graph = graph  # ty: ignore[invalid-assignment] -- source is read-only
    assert viewer.url is None
    np.testing.assert_array_equal(viewer._xlim, [-2, 2])
    viewer.close()


def test_graph_validates_source_and_dimension():
    x, y = Variable("x"), Variable("y")
    with pytest.raises(TypeError, match="Function"):
        FunctionGraph(x)  # ty: ignore[invalid-argument-type] -- invalid source
    for function in (Function(x + y, (x, y)), Function(Scalar(2), ())):
        with pytest.raises(ValueError, match="one variable"):
            FunctionGraph(function)
    with pytest.raises(TypeError, match="FunctionGraph"):
        Viewer(x)  # ty: ignore[invalid-argument-type] -- invalid realization


@pytest.mark.parametrize(
    "bounds",
    [(2, 2), (3, 1), (0, sp.oo), (0, float("nan")), (-1e308, 1e308), (True, 2)],
)
def test_viewer_rejects_invalid_bounds(bounds):
    x = Variable("x")
    with pytest.raises(ValueError, match="Bounds"):
        Viewer(FunctionGraph(Polynomial(x, x)), xlim=bounds)


def test_viewer_range_and_supported_domains():
    x = Variable("x")
    np.testing.assert_array_equal(
        Viewer(FunctionGraph(Function(x, (x,))))._xlim, [-5, 5]
    )
    graph = FunctionGraph(Function(x, (x,), Set.interval(0, sp.oo)))
    np.testing.assert_array_equal(Viewer(graph)._xlim, [-5, 5])
    np.testing.assert_array_equal(Viewer(graph, xlim=(1, 2))._xlim, [1, 2])
    with pytest.raises(ValueError, match="domains"):
        Viewer(FunctionGraph(Function(x, (x,), Set.integers())))
    symbolic = Set.interval(sp.Symbol("a", real=True), sp.oo)
    with pytest.raises(ValueError, match="numerically evaluable"):
        Viewer(FunctionGraph(Function(x, (x,), symbolic)))
    with pytest.raises(ValueError, match="numerically evaluable"):
        Viewer(FunctionGraph(Function(x, (x,), symbolic)), xlim=(-1, 1))


@settings(max_examples=30, deadline=None)
@given(a=st.integers(-10, 10), b=st.integers(-10, 10), c=st.integers(-10, 10))
def test_polynomial_samples_match_python_arithmetic(a: int, b: int, c: int):
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Polynomial(a * x**2 + b * x + c, x)))
    xs, ys, valid = viewer._sample((-2, 2))
    assert valid.all()
    assert xs[0] == -2 and xs[-1] == 2
    assert len(xs) == len(ys) == 513
    assert ys == pytest.approx([a * value**2 + b * value + c for value in xs])


@pytest.mark.parametrize("expression", [sp.oo, sp.nan, sp.I])
def test_constant_invalid_values_become_gaps(expression):
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Function(Expression(expression), (x,))))
    samples = viewer._sample((-1, 1))
    assert not samples.valid.any()
    assert np.isnan(samples.y).all()


def test_pole_and_nonreal_values_become_gaps():
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Function(1 / x, (x,))))
    xs, ys, valid = viewer._sample((-1, 1))
    assert xs[256] == 0
    assert not valid[256]
    assert np.isnan(ys[256])
    assert ys[0] == -1 and ys[-1] == 1
    sqrt = Function(Expression(sp.sqrt(x.to_sympy())), (x,))
    xs, ys, valid = Viewer(FunctionGraph(sqrt))._sample((-1, 1))
    assert not valid[:256].any()
    assert np.isnan(ys[:256]).all()
    assert ys[256] == 0 and ys[-1] == 1


def test_open_domain_is_clipped_and_empty_range_is_supported():
    x = Variable("x")
    function = Function(x**2, (x,), Set.interval(0, 2, left_open=True, right_open=True))
    viewer = Viewer(FunctionGraph(function))
    xs, ys, valid = viewer._sample((-1, 3))
    np.testing.assert_array_equal(valid, (xs > 0) & (xs < 2))
    np.testing.assert_array_equal(ys[valid], xs[valid] ** 2)
    assert np.isnan(ys[~valid]).all()
    assert not viewer._sample((3, 4)).valid.any()
    assert function.domain is not None and function.domain.contains(0) is False
    assert function(3) == 9  # Viewing does not change math evaluation semantics.


def test_unsupported_evaluation_reports_a_useful_error():
    x = Variable("x")
    undefined = Expression(sp.Function("unimplemented")(x.to_sympy()))  # ty: ignore[call-non-callable] -- SymPy dynamically creates a callable function class
    viewer = Viewer(FunctionGraph(Function(undefined, (x,))))
    with pytest.raises(ValueError, match="NumPy backend"):
        viewer._sample((-1, 1))
    assert viewer.url is None


def test_missing_extra_reports_installation_instructions(monkeypatch):
    import mathfirst.viz.viewer as runtime

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(runtime.importlib, "import_module", unavailable)
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Polynomial(x, x)))
    with pytest.raises(ImportError, match=r"mathfirst\[viz\]"):
        viewer.show(block=False, open_browser=False)
    assert viewer.url is None


@pytest.mark.parametrize("kwargs", [{"block": 1}, {"open_browser": "yes"}])
def test_show_requires_explicit_boolean_options(kwargs):
    x = Variable("x")
    with pytest.raises(TypeError, match="booleans"):
        Viewer(FunctionGraph(Polynomial(x, x))).show(**kwargs)


def test_bounds_are_owned_numpy_state_with_shape_validation():
    x = Variable("x")
    bounds = np.array([-1.0, 1.0], dtype=np.float32)
    viewer = Viewer(FunctionGraph(Polynomial(x, x)), xlim=bounds)
    bounds[:] = [10, 20]
    np.testing.assert_array_equal(viewer._xlim, [-1, 1])
    assert viewer._xlim.dtype == np.dtype(np.float64)
    assert not viewer._xlim.flags.writeable
    assert not np.shares_memory(viewer._xlim, bounds)
    with pytest.raises(ValueError, match="Bounds"):
        viewer._sample(np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="Bounds"):
        viewer._sample(np.array([False, True]))


def test_sample_arrays_have_consistent_dtypes_shapes_and_ownership():
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Polynomial(Scalar(3), x)))
    samples = viewer._sample((-1, 1))
    assert samples.x.dtype == samples.y.dtype == np.dtype(np.float64)
    assert samples.valid.dtype == np.dtype(np.bool_)
    assert samples.x.shape == samples.y.shape == samples.valid.shape == (513,)
    assert samples.valid.all()
    np.testing.assert_array_equal(samples.y, np.full(513, 3))
    repeated = viewer._sample((-1, 1))
    for first, second in zip(samples, repeated, strict=True):
        assert not first.flags.writeable
        assert not np.shares_memory(first, second)
        with pytest.raises(ValueError, match="read-only"):
            first[0] = 0


def test_large_exact_constant_can_be_sampled_as_float64():
    x = Variable("x")
    samples = Viewer(FunctionGraph(Function(Scalar(2**100), (x,))))._sample((-1, 1))
    assert samples.valid.all()
    np.testing.assert_array_equal(samples.y, np.full(513, float(2**100)))


@pytest.mark.parametrize("value", [np.ones((2, 513)), np.array(["1"]), True])
def test_invalid_backend_output_is_reported_without_corrupting_state(value):
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Polynomial(x, x)))
    viewer._evaluate = lambda values: value
    with pytest.raises(ValueError, match="NumPy backend"):
        viewer._sample((-1, 1))
    np.testing.assert_array_equal(viewer._xlim, [-5, 5])
