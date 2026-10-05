# Design Philosophy

## Overview

This project uses a clear separation between **Python's semantic/object model** and **NumPy's numerical representation model**.

The core rule is:

> Use native Python for structure, semantics, relationships, and control.

> Use NumPy for numerical values and homogeneous numerical data.

The goal is to make numerical APIs more explicit, strongly typed, and representation-aware while keeping the surrounding Python code idiomatic and composable.

---

## Native Python Layer

Native Python types define the program's semantic structure.

Prefer:

- `str` for names, labels, and text
- `bool` for program state and control
- `None` for absence
- `dict` for mappings and relationships
- `list` for heterogeneous or semantic sequences
- `tuple` for structural grouping
- `set` / `frozenset` for membership
- `Enum` / `Literal` for symbolic choices
- classes and dataclasses for domain objects
- protocols, generics, and type aliases for API structure

These types answer questions such as:

- What is this object?
- How does it relate to other objects?
- What state is the program in?
- How is the API organized?

---

## NumPy Numerical Layer

NumPy is the canonical representation for numerical values.

Prefer:

- `np.int8` ... `np.int64`
- `np.uint8` ... `np.uint64`
- `np.float16` ... `np.float64`
- NumPy complex types
- `np.bool_` for numerical boolean data and masks
- `ndarray` for vectors, matrices, tensors, images, samples, and other homogeneous data
- `np.datetime64` and `np.timedelta64` where time is represented numerically
- structured dtypes where fixed numerical records are appropriate

These types make properties such as the following explicit:

- width
- precision
- signedness
- dtype
- shape
- memory representation

---

## Scalar Convention

Numerical scalars should generally use NumPy types when they form part of the public numerical model.

Examples:

```text
mass        → np.float64
temperature → np.float32
index       → np.int64
count       → np.uint32
object_id   → np.uint64
```

This provides a consistent numerical type system instead of mixing Python `int` / `float` with NumPy types arbitrarily.

---

## Boolean Convention

Use `bool` for semantic program state:

```text
enabled
visible
debug
```

Use `np.bool_` or boolean arrays when the value is part of numerical data:

```text
valid_sample
collision_mask
selection_mask
```

---

## Collections

Container choice follows semantic meaning.

Use Python containers for program structure:

```text
dict[str, Body]
list[Body]
set[Entity]
tuple[Frame, Frame]
```

Use NumPy arrays for homogeneous numerical collections:

```text
float64[3]
float64[4, 4]
float32[N, 3]
uint8[H, W, 3]
```

A domain object may therefore combine both layers:

```text
Body
├── name: str
├── active: bool
├── id: np.uint64
├── mass: np.float64
├── position: float64[3]
├── velocity: float64[3]
└── vertices: float32[N, 3]
```

---

## Typing

The intended typing stack is:

```text
Python typing
    +
NumPy dtypes
    +
numpy.typing
    +
jaxtyping
    +
static type checker
```

This allows APIs to express both semantic meaning and numerical representation.

Examples:

```text
Position → float64[3]
Matrix4  → float64[4, 4]
Vertices → float32[N, 3]
Image    → uint8[H, W, 3]
Mask     → bool[N]
```

Domain-specific classes should wrap these representations where additional semantics are required.

---

## Design Boundary

The decision rule is:

### Use Python when the value represents

- semantics
- structure
- identity
- ownership
- relationships
- configuration
- control flow

### Use NumPy when the value represents

- numerical magnitude
- numerical state
- coordinates
- indices with explicit representation
- vectors
- matrices
- tensors
- images
- signals
- measurements
- numerical datasets

---

## Rationale

This design provides:

- consistent numerical representation
- stronger and more expressive type contracts
- explicit precision and width
- cleaner interoperability with numerical/native systems
- direct compatibility with NumPy's optimized computational model
- a clear boundary between semantic objects and numerical data

The intent is not to replace Python's object model with NumPy.

Instead:

> **Python defines what things are and how they relate.**

> **NumPy defines how numerical values are represented and computed.**
