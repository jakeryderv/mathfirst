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

## Releases

Update `[project.version]` in `pyproject.toml`, run `uv lock`, and run the checks
above. Commit and push the changes, then wait for CI to pass for that commit.
Create a GitHub release with a tag matching the version, such as `v0.2.0`,
targeting that exact commit.

`.github/workflows/publish.yml` checks the version, runs validation, builds and
checks both distributions, and publishes through the `pypi` environment using
Trusted Publishing. No PyPI token secret is needed. Confirm the workflow
succeeds and the release appears on PyPI before considering publication complete.
