# Development

Run from the repository root:

```sh
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv build --no-sources
```

Runtime dependencies belong in `[project.dependencies]`; development tools belong
in the `dev` dependency group. Commit `uv.lock` after changing dependencies.
Use `uv run ruff format .` to apply formatting.

Implementation modules live directly in `src/mathfirst/`. The package root
exports the public objects and operations. Tests live in `tests/`, import the
installed package, and use pytest's `importlib` import mode.

CI runs on pushes and pull requests, testing Python 3.12, 3.13, and 3.14.
The publishing workflow runs the same checks before building and uploading.

## Coverage and generated tests

`uv run pytest` reports branch coverage for `mathfirst`, including missing lines
and branch destinations. Pre-push, CI, and release checks use the same pytest
configuration. Coverage currently reports a baseline without a minimum threshold;
use the gaps to guide useful tests before choosing a required percentage.

Generate a browsable report or run without coverage while debugging:

```sh
uv run pytest --cov-report=html
uv run pytest --no-cov
```

The HTML report is written to `htmlcov/index.html`. Coverage data and reports
are ignored by Git.

Hypothesis tests in `tests/test_properties.py` check exact scalar arithmetic
against Python's `Fraction`, polynomial evaluation against Horner's method,
and real roots against known factors and multiplicities. Inputs are bounded
integers and rationals; symbolic tests use 50 generated examples each and disable
per-example time deadlines to avoid failures caused by machine speed.

Hypothesis runs through pytest and stores useful examples locally in the ignored
`.hypothesis/` directory. If it finds a failure, preserve a focused regression test
alongside the generated test. To inspect example statistics:

```sh
uv run pytest tests/test_properties.py --hypothesis-show-statistics
```

## Git hooks

After cloning and running `uv sync --locked`, install both local hooks:

```sh
uv run --locked pre-commit install
```

The configuration installs both `pre-commit` and `pre-push`. Commits check
formatting and linting on staged Python files and run project-wide type checking.
Pushes run the full test suite once, including pushes without Python changes.
Hook commands use `uv run --locked` to use the project's locked tool versions.
Formatting checks report changes to make; run `uv run ruff format .` to apply them.

Run either stage manually against all tracked files:

```sh
uv run --locked pre-commit run --all-files --hook-stage pre-commit
uv run --locked pre-commit run --all-files --hook-stage pre-push
```

Hook installation is local to each clone. CI and release validation continue to
run independently of local hooks.

## Releases

Update `[project.version]` in `pyproject.toml`, run `uv lock`, and run the checks
above. Commit and push the changes, then wait for CI to pass for that commit.
Create a GitHub release with a tag matching the version, such as `v0.2.0`,
targeting that exact commit.

`.github/workflows/publish.yml` checks the version, runs validation, builds and
checks both distributions, and publishes through the `pypi` environment using
Trusted Publishing. No PyPI token secret is needed. Confirm the workflow
succeeds and the release appears on PyPI before considering publication complete.
