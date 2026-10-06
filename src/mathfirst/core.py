from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import cast, overload

import numpy as np
import sympy as sp
from numpy.typing import DTypeLike, NDArray
from sympy.parsing.sympy_parser import (
    parse_expr,
    rationalize,
    standard_transformations,
)

from .numerical import (
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_REAL_DTYPE,
    NumericArray,
    NumericInput,
    NumericScalar,
    RealArray,
    _numeric_dtype,
    as_numeric_array,
)


class Expression:
    """A scalar symbolic expression with structural equality.

    Equality compares SymPy representations without algebraic simplification.
    Expressions are unhashable, including Scalar and Variable instances.
    """

    __slots__ = ("_expr",)

    def __init__(
        self,
        value: Expression | NumericInput | Fraction | sp.Expr,
    ) -> None:
        self._expr = _to_sympy(value)

    def to_sympy(self) -> sp.Expr:
        """Return the underlying SymPy expression for backend-specific work."""
        return self._expr

    def __add__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr + _to_sympy(other))

    def __radd__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) + self._expr)

    def __sub__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr - _to_sympy(other))

    def __rsub__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) - self._expr)

    def __mul__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr * _to_sympy(other))

    def __rmul__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) * self._expr)

    def __truediv__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr / _to_sympy(other))

    def __rtruediv__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) / self._expr)

    def __pow__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr ** _to_sympy(other))

    def __rpow__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) ** self._expr)

    def __neg__(self) -> Expression:
        return _wrap_sympy(-self._expr)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bool, np.bool_)) or not isinstance(
            other, (Expression, int, Fraction, float, complex, np.number, sp.Expr)
        ):
            return NotImplemented

        return self._expr == _to_sympy(other)

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"Expression({self._expr!r})"


class ApproximateScalarWarning(UserWarning):
    """A floating-point input was passed without explicit approximate intent."""


class Scalar(Expression):
    """A scalar whose numeric representation distinguishes exact and approximate.

    Integers and Fractions remain exact. Decimal strings are parsed as exact
    rationals. Python/NumPy floating and complex inputs remain approximate and
    warn unless passed through approx(). Existing SymPy expressions retain
    their representation. to_numpy() is the concrete numerical boundary.
    """

    def __init__(
        self,
        value: Expression | NumericInput | Fraction | str | sp.Expr,
    ) -> None:
        super().__init__(_to_scalar_sympy(value))

        if self._expr.free_symbols:
            raise ValueError("Scalar cannot contain free variables")

        if isinstance(value, (float, complex, np.floating, np.complexfloating)):
            message = (
                "Python floats are approximate. "
                f'Use Scalar.exact("{value}") for exact decimal intent or '
                if isinstance(value, float) and not isinstance(value, np.floating)
                else "Floating-point inputs are approximate. "
                "Use Scalar.exact with an exact expression or decimal string, or "
            )
            warnings.warn(
                message
                + f"Scalar.approx({value!r}) to explicitly accept approximation.",
                ApproximateScalarWarning,
                stacklevel=2,
            )

    @classmethod
    def exact(
        cls,
        value: Expression | NumericInput | Fraction | str | sp.Expr,
    ) -> Scalar:
        """Construct an exact scalar without guessing intent from approximate input.

        Python/NumPy floating or complex values and expressions containing Floats
        are rejected. Use decimal strings or Fractions for exact rational intent.
        """
        if isinstance(value, (float, complex, np.floating, np.complexfloating)):
            raise TypeError(
                "Exact scalars cannot infer intent from floating or complex "
                "values; use a string, integer, Fraction, or exact expression"
            )

        scalar = cls(value)

        if not scalar.is_exact:
            raise ValueError("Exact scalars cannot contain approximate SymPy Floats")

        return scalar

    @classmethod
    def approx(
        cls,
        value: Expression | NumericInput | Fraction | str | sp.Expr,
    ) -> Scalar:
        """Explicitly approximate a numeric scalar using SymPy's default precision.

        Python floats keep their existing approximation. Increasing evaluation
        precision later cannot recover information absent from a float input.
        """
        scalar = cls(_to_scalar_sympy(value))

        if not scalar._expr.is_number:
            raise ValueError("Approximation requires a numeric scalar")

        result = scalar._expr.evalf()

        # evalf() can keep exact zero; an explicit approximation should be Float.
        if result.is_Rational:
            result = sp.Float(result)

        return cls(result)

    @overload
    def to_numpy(
        self, *, dtype: None = None, allow_nonfinite: bool = False
    ) -> np.float64 | np.complex128: ...

    @overload
    def to_numpy[NumericT: NumericScalar](
        self,
        *,
        dtype: type[NumericT] | np.dtype[NumericT],
        allow_nonfinite: bool = False,
    ) -> NumericT: ...

    @overload
    def to_numpy(
        self, *, dtype: DTypeLike, allow_nonfinite: bool = False
    ) -> NumericScalar: ...

    def to_numpy(
        self, *, dtype: DTypeLike | None = None, allow_nonfinite: bool = False
    ) -> NumericScalar:
        """Evaluate into a NumPy scalar without changing the mathematical source.

        MathFirst defaults to float64 for real values and complex128 otherwise.
        Explicit dtypes support fixed-width integers, float16/32/64, and
        complex64/128. Integer output requires a proven integer and checks its
        range. Real output cannot discard an imaginary component. Floating
        output uses binary64 evaluation followed by the requested dtype's
        rounding; underflow is permitted and overflow raises OverflowError.

        NaN and signed infinity require allow_nonfinite=True. Complex infinity
        and unevaluable expressions are rejected. Returned NumPy scalars are
        immutable; conversion creates no mutable view or cache of this object.
        """
        if not isinstance(allow_nonfinite, bool):
            raise TypeError("allow_nonfinite must be a Python boolean")
        if not self._expr.is_number or self._expr.has(sp.zoo):
            raise ValueError("Conversion requires an evaluable numerical scalar")
        if dtype is None:
            dtype = (
                DEFAULT_REAL_DTYPE
                if self._expr.is_real is True or self._expr in (sp.nan, sp.oo, -sp.oo)
                else DEFAULT_COMPLEX_DTYPE
            )
        target = _numeric_dtype(dtype)
        if target.kind in "iu":
            if self._expr.is_integer is not True:
                raise ValueError("Integer output requires a proven integer value")
            value = int(self._expr)
            limits = np.iinfo(target.str)
            if value < limits.min or value > limits.max:
                raise OverflowError(f"Value is outside the range of {target}")
            return target.type(value)
        try:
            # SymPy's __complex__ first evalf()s at default decimal precision.
            # Convert components directly to avoid that extra rounding step.
            real, imaginary = self._expr.as_real_imag()
            numeric = complex(float(real), float(imaginary))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Scalar cannot be evaluated numerically") from exc
        if self._expr.is_finite is True and not np.isfinite(numeric):
            raise OverflowError("Finite scalar overflows binary64 evaluation")
        if target.kind == "f":
            if numeric.imag != 0 and self._expr is not sp.nan:
                raise ValueError("Real output cannot discard imaginary components")
            converted = as_numeric_array(
                numeric.real, dtype=target, allow_nonfinite=allow_nonfinite
            )
        else:
            converted = as_numeric_array(
                numeric, dtype=target, allow_nonfinite=allow_nonfinite
            )
        # A scalar input guarantees a zero-dimensional array and scalar indexing.
        return cast(NumericScalar, converted[()])

    @property
    def is_exact(self) -> bool:
        """Whether the current representation contains no approximate Float atoms.

        This describes the representation, not the provenance of earlier input
        or arithmetic. Nonfinite symbolic values retain their representation.
        """
        return not self._expr.has(sp.Float)

    def __repr__(self) -> str:
        return f"Scalar({self._expr!r})"


class Variable(Expression):
    """A named real variable; equal names represent the same variable."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Variable name must be a string")
        if not name:
            raise ValueError("Variable name cannot be empty")

        self._name = name
        super().__init__(sp.Symbol(name, real=True))

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Variable({self._name!r})"


class Set:
    """A mathematical set with definite or unresolved membership."""

    __slots__ = ("_set",)

    def __init__(self, value: sp.Set) -> None:
        if not isinstance(value, sp.Set):
            raise TypeError("value must be a SymPy Set")

        self._set = value

    @classmethod
    def reals(cls) -> Set:
        return cls(sp.S.Reals)

    @classmethod
    def integers(cls) -> Set:
        return cls(sp.S.Integers)

    @classmethod
    def interval(
        cls,
        start: object,
        end: object,
        *,
        left_open: bool = False,
        right_open: bool = False,
    ) -> Set:
        return cls(
            sp.Interval(
                _to_sympy(start),
                _to_sympy(end),
                left_open=left_open,
                right_open=right_open,
            )
        )

    @classmethod
    def product(cls, *sets: Set) -> Set:
        """Build a Cartesian product whose elements are coordinate tuples."""
        if not all(isinstance(value, Set) for value in sets):
            raise TypeError("Cartesian product factors must be Sets")

        return cls(sp.ProductSet(*(value.to_sympy() for value in sets)))

    def contains(self, value: object) -> bool | None:
        """Return True or False for known membership, or None if unresolved.

        Tuples are interpreted as elements of a Cartesian product.
        """
        if isinstance(value, (tuple, sp.Tuple)):
            element = sp.Tuple(*(_to_sympy(coordinate) for coordinate in value))
        else:
            element = _to_sympy(value)

        membership = self._set.contains(element)

        if membership is sp.S.true:
            return True
        if membership is sp.S.false:
            return False

        return None

    def to_sympy(self) -> sp.Set:
        """Return the underlying SymPy set for backend-specific operations."""
        return self._set

    def __str__(self) -> str:
        return str(self._set)

    def __repr__(self) -> str:
        return f"Set({self._set!r})"


class Point:
    """Ordered mathematical coordinates; numerical conversion is explicit."""

    __slots__ = ("_coordinates",)

    def __init__(self, *coordinates: object) -> None:
        if not coordinates:
            raise ValueError("Point requires at least one coordinate")

        self._coordinates = tuple(
            _wrap_sympy(_to_sympy(coordinate)) for coordinate in coordinates
        )

    @property
    def coordinates(self) -> tuple[Expression, ...]:
        return self._coordinates

    @property
    def dimension(self) -> int:
        return len(self._coordinates)

    def to_sympy(self) -> sp.Tuple:
        """Return the ordered coordinates as a SymPy tuple."""
        return sp.Tuple(*(coordinate.to_sympy() for coordinate in self._coordinates))

    @overload
    def to_numpy(
        self, *, dtype: None = None, allow_nonfinite: bool = False
    ) -> RealArray: ...

    @overload
    def to_numpy[NumericT: NumericScalar](
        self,
        *,
        dtype: type[NumericT] | np.dtype[NumericT],
        allow_nonfinite: bool = False,
    ) -> NDArray[NumericT]: ...

    @overload
    def to_numpy(
        self, *, dtype: DTypeLike, allow_nonfinite: bool = False
    ) -> NumericArray: ...

    def to_numpy(
        self, *, dtype: DTypeLike | None = None, allow_nonfinite: bool = False
    ) -> NumericArray:
        """Return independent, writable coordinates of shape (dimension,).

        The point domain defaults to float64. Complex coordinates require an
        explicit complex dtype. Scalar conversion policies apply to each
        coordinate, including exact integer conversion and nonfinite opt-in.
        Unresolved symbolic coordinates must be substituted before conversion.
        Mutating the returned array never changes the mathematical point.
        """
        if not isinstance(allow_nonfinite, bool):
            raise TypeError("allow_nonfinite must be a Python boolean")
        target = _numeric_dtype(DEFAULT_REAL_DTYPE if dtype is None else dtype)
        result = np.empty(self.dimension, dtype=target)
        for index, coordinate in enumerate(self._coordinates):
            if coordinate.to_sympy().free_symbols:
                raise ValueError(
                    "Point conversion requires coordinates without free variables"
                )
            result[index] = Scalar(coordinate).to_numpy(
                dtype=target, allow_nonfinite=allow_nonfinite
            )
        return result

    def __repr__(self) -> str:
        return f"Point({', '.join(map(str, self._coordinates))})"


@dataclass(frozen=True, slots=True)
class Root:
    value: Expression
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.value, Expression):
            raise TypeError("Root value must be an Expression")
        if isinstance(self.multiplicity, bool) or not isinstance(
            self.multiplicity, int
        ):
            raise TypeError("Root multiplicity must be an integer")
        if self.multiplicity < 1:
            raise ValueError("Root multiplicity must be at least 1")

    def to_sympy(self) -> sp.Tuple:
        """Return the root value and its multiplicity as a SymPy tuple."""
        return sp.Tuple(self.value.to_sympy(), self.multiplicity)


def _to_sympy(value: object) -> sp.Expr:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Boolean values cannot represent a scalar expression")
    if isinstance(value, Expression):
        return value.to_sympy()

    result = sp.sympify(value)

    if not isinstance(result, sp.Expr):
        raise TypeError(f"Cannot represent {value!r} as a scalar expression")

    return result


def _wrap_sympy(value: sp.Expr) -> Expression:
    if value.free_symbols:
        return Expression(value)

    return Scalar(value)


def _to_scalar_sympy(value: object) -> sp.Expr:
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    if isinstance(value, str):
        return _to_sympy(
            parse_expr(value, transformations=(*standard_transformations, rationalize))
        )

    return _to_sympy(value)
