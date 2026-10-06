from __future__ import annotations

from typing import cast, overload

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .core import Expression, Scalar, Set, Variable, _to_sympy, _wrap_sympy
from .numerical import (
    DEFAULT_REAL_DTYPE,
    NumericArray,
    NumericScalar,
    RealArray,
    _numeric_dtype,
    as_numeric_array,
)


class Function:
    """A scalar expression with distinct arguments and descriptive sets.

    Multivariable domains contain argument tuples, typically constructed with
    Set.product(). None means the domain or codomain is unspecified. Declared
    sets express an intended mathematical contract, not a proof of validity.
    Evaluation validates argument structure; membership can be checked explicitly
    using Set.contains().
    """

    def __init__(
        self,
        expression: Expression,
        variables: tuple[Variable, ...],
        domain: Set | None = None,
        codomain: Set | None = None,
    ) -> None:
        if not isinstance(expression, Expression):
            raise TypeError("Function expression must be an Expression")
        if not isinstance(variables, tuple) or not all(
            isinstance(variable, Variable) for variable in variables
        ):
            raise TypeError("Function variables must be a tuple of Variables")
        if domain is not None and not isinstance(domain, Set):
            raise TypeError("Function domain must be a Set or None")
        if codomain is not None and not isinstance(codomain, Set):
            raise TypeError("Function codomain must be a Set or None")

        allowed_symbols = {variable.to_sympy() for variable in variables}

        if len(allowed_symbols) != len(variables):
            raise ValueError("Function variables must have distinct symbols")

        unknown_symbols = expression.to_sympy().free_symbols - allowed_symbols

        if unknown_symbols:
            raise ValueError(
                f"Expression contains undeclared variables: {unknown_symbols}"
            )

        self._expression = expression
        self._variables = variables
        self._domain = domain
        self._codomain = codomain

    @property
    def expression(self) -> Expression:
        return self._expression

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._variables

    @property
    def domain(self) -> Set | None:
        return self._domain

    @property
    def codomain(self) -> Set | None:
        return self._codomain

    def __call__(self, *values: object) -> Expression:
        """Evaluate with simultaneous substitution and descriptive set metadata."""
        if len(values) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} arguments, got {len(values)}"
            )

        arguments = tuple(_to_sympy(value) for value in values)

        substitutions = {
            variable.to_sympy(): argument
            for variable, argument in zip(
                self._variables,
                arguments,
                strict=True,
            )
        }

        result = self._expression.to_sympy().subs(
            substitutions.items(), simultaneous=True
        )

        return _wrap_sympy(result)

    @overload
    def evaluate_numpy(
        self,
        *values: ArrayLike,
        dtype: None = None,
        allow_nonfinite: bool = False,
    ) -> np.float64 | RealArray: ...

    @overload
    def evaluate_numpy[NumericT: NumericScalar](
        self,
        *values: ArrayLike,
        dtype: type[NumericT] | np.dtype[NumericT],
        allow_nonfinite: bool = False,
    ) -> NumericT | NDArray[NumericT]: ...

    @overload
    def evaluate_numpy(
        self,
        *values: ArrayLike,
        dtype: DTypeLike,
        allow_nonfinite: bool = False,
    ) -> NumericScalar | NumericArray: ...

    def evaluate_numpy(
        self,
        *values: ArrayLike,
        dtype: DTypeLike | None = None,
        allow_nonfinite: bool = False,
    ) -> NumericScalar | NumericArray:
        """Evaluate numerically with NumPy broadcasting and owned results.

        Inputs and output use MathFirst's float64 default, or an explicitly
        requested floating/complex dtype. Complex input evaluation requires a
        complex dtype. Zero-dimensional inputs return a NumPy scalar; otherwise
        the result is an independent, writable array of the broadcast shape,
        including for constant expressions and empty inputs.

        Nonfinite inputs/results require allow_nonfinite=True. Real output
        cannot discard imaginary components. Unsupported backend expressions
        raise ValueError. Sets and symbolic assumptions remain descriptive:
        this operation does not enforce membership or complex continuation of
        formulas simplified using real-variable assumptions.
        """
        if len(values) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} arguments, got {len(values)}"
            )
        if not isinstance(allow_nonfinite, bool):
            raise TypeError("allow_nonfinite must be a Python boolean")
        target = _numeric_dtype(DEFAULT_REAL_DTYPE if dtype is None else dtype)
        if target.kind not in "fc":
            raise TypeError("Numerical evaluation requires a floating or complex dtype")
        arguments = tuple(
            as_numeric_array(value, dtype=target, allow_nonfinite=allow_nonfinite)
            for value in values
        )
        try:
            shape = np.broadcast_shapes(*(argument.shape for argument in arguments))
        except ValueError as exc:
            raise ValueError(
                "Numerical arguments must have broadcast-compatible shapes"
            ) from exc
        if self._expression.to_sympy().is_number:
            # Preserve the scalar conversion policy for exact constants whose
            # Python integers would otherwise infer an object array.
            raw = Scalar(self._expression).to_numpy(
                dtype=target, allow_nonfinite=allow_nonfinite
            )
        else:
            try:
                evaluate = sp.lambdify(
                    tuple(variable.to_sympy() for variable in self._variables),
                    self._expression.to_sympy(),
                    modules="numpy",
                    dummify=True,
                    use_imps=False,
                )
                with np.errstate(all="ignore"):
                    raw = evaluate(*arguments)
            except (
                TypeError,
                ValueError,
                NameError,
                ZeroDivisionError,
                OverflowError,
                NotImplementedError,
            ) as exc:
                raise ValueError(
                    "This function cannot be evaluated with the NumPy backend"
                ) from exc
        result = as_numeric_array(raw, dtype=target, allow_nonfinite=allow_nonfinite)
        if result.shape != shape:
            try:
                result = np.broadcast_to(result, shape).copy()
            except ValueError as exc:
                raise ValueError(
                    "Numerical result does not match the argument shape"
                ) from exc
        if not shape:
            return cast(NumericScalar, result[()])
        return result

    def to_sympy(self) -> sp.Basic:
        """Return a SymPy Lambda; declared sets remain metadata on this object.

        Subclasses may return a more specific mathematical representation.
        """
        return sp.Lambda(
            tuple(variable.to_sympy() for variable in self._variables),
            self._expression.to_sympy(),
        )

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        return (
            f"Function("
            f"expression={self.expression!r}, "
            f"variables={self.variables!r}, "
            f"domain={self.domain!r}, "
            f"codomain={self.codomain!r}"
            f")"
        )
