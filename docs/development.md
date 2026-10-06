# Development

Run from the repository root:

```sh
uv sync --locked --extra viz
uv run --extra viz ruff format --check .
uv run --extra viz ruff check .
uv run --extra viz ty check
uv run --extra viz pytest
uv build --no-sources
```

Core runtime dependencies (NumPy and SymPy) belong in `[project.dependencies]`;
optional viewer backends belong in `[project.optional-dependencies].viz`, and
development tools belong in the `dev` dependency group. Development commands enable the `viz` extra
so the full suite includes viewer unit and server tests. Commit `uv.lock` after
changing dependencies.
Use `uv run --extra viz ruff format .` to apply formatting.

Implementation modules live directly in `src/mathfirst/`. The package root
exports the public objects and operations. Tests live in `tests/`, import the
installed package, and use pytest's `importlib` import mode. Viewer runtime and
packaged static assets live in `src/mathfirst/viz/`; see [visualization usage](viz.md).

CI runs on pushes and pull requests, testing Python 3.12, 3.13, and 3.14.
The publishing workflow runs the same checks before building and uploading.

## Coverage and generated tests

`uv run --extra viz pytest` reports branch coverage for `mathfirst`, including
missing lines and branch destinations. Pre-push, CI, and release checks use the same pytest
configuration. Coverage currently reports a baseline without a minimum threshold;
use the gaps to guide useful tests before choosing a required percentage.

Generate a browsable report or run without coverage while debugging:

```sh
uv run --extra viz pytest --cov-report=html
uv run --extra viz pytest --no-cov
```

The HTML report is written to `htmlcov/index.html`. Coverage data and reports
are ignored by Git.

Hypothesis tests in `tests/test_properties.py` check exact scalar arithmetic
against Python's `Fraction`, polynomial evaluation against Horner's method,
real roots against known factors and multiplicities, and scalar NumPy conversion
against Python's `Fraction` conversion. Numerical canonicalization tests also cover
finite float64 inputs, and generated viewer tests compare sampled quadratics with
Python arithmetic. Numerical function tests check polynomial values, broadcasting,
empty and constant results, dtype/validity contracts, and independent array
ownership. Server tests verify NumPy data crosses the JSON boundary as numbers
and `null`, with viewport recovery after errors. Symbolic inputs are bounded integers
and rationals; the more expensive symbolic tests use 50 examples and disable per-example time deadlines
to avoid failures caused by machine speed.

Hypothesis runs through pytest and stores useful examples locally in the ignored
`.hypothesis/` directory. If it finds a failure, preserve a focused regression test
alongside the generated test. To inspect example statistics:

```sh
uv run --extra viz pytest tests/test_properties.py --hypothesis-show-statistics
```

## Git hooks

After cloning and running `uv sync --locked --extra viz`, install both local hooks:

```sh
uv run --locked --extra viz pre-commit install
```

The configuration installs both `pre-commit` and `pre-push`. Commits check
formatting and linting on staged Python files and run project-wide type checking.
Pushes run the full test suite once, including pushes without Python changes.
Hook commands use `uv run --locked --extra viz` to use the project's locked tool
versions. Formatting checks report changes to make; run
`uv run --extra viz ruff format .` to apply them.

Run either stage manually against all tracked files:

```sh
uv run --locked --extra viz pre-commit run --all-files --hook-stage pre-commit
uv run --locked --extra viz pre-commit run --all-files --hook-stage pre-push
```

Hook installation is local to each clone. CI and release validation continue to
run independently of local hooks.

## Releases

Update `[project.version]` in `pyproject.toml`, run `uv lock`, and run the checks
above. Commit and push the changes, then wait for CI to pass for that commit.
Create a GitHub release with a tag matching the version, such as `v0.2.0`,
targeting that exact commit.

`.github/workflows/publish.yml` checks the version, runs validation, builds and
checks both distributions with and without the visualization extra, and publishes
through the `pypi` environment using Trusted Publishing. No PyPI token secret is
needed. Confirm the workflow
succeeds and the release appears on PyPI before considering publication complete.
