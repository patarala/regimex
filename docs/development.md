# Working on Regimex

Run commands from the checkout root. The project pins Python in
[`.python-version`](../.python-version) and uv in [`pyproject.toml`](../pyproject.toml).
Install the locked development dependencies and run the core checks.

```sh
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
```

To work on one behavior, pass a test file or a pytest node ID. For example,
`uv run --locked pytest tests/test_exports.py` exercises the export contract.
Optional integrations report skips when their dependencies or services are
unavailable. The [architecture guide](architecture.md#where-to-find-the-code)
maps each part of the service to its source files.

## Checks by change

| Change | What to run |
| --- | --- |
| Model features, training or inference | `uv run --locked --extra ml pytest tests/test_baseline_backend.py tests/test_tcn_backend.py tests/test_fixed_update_tcn.py`. For interval decoding, include `tests/test_segmentation.py`. |
| Python types, including optional dependencies | `uv run --locked --all-extras --all-groups mypy src tests` |
| Workspace or report controls | `node --experimental-vm-modules --test tests/*.test.cjs`. Exercise changed controls at desktop and narrow widths. For reports, check offline loading, linked zoom and missing values. |
| PostgreSQL, queue, backup or object storage | Use the [integration test setup](storage.md#integration-test-setup) for a disposable database, recovery tools and the HTTPS S3 fixture. |
| Dependencies or packaging | After changing scientific dependencies, run the scientific imports and CPU tensor check in CI. After changing package contents, run `uv build --no-build-isolation` and test the installed wheel outside the checkout. [CI commands](../.github/workflows/ci.yml) |
| Container startup or deployment | With Docker and Compose available, run `python tests/support/compose_service_smoke.py`. |
| Documentation | Follow the changed links from the README and run the examples with their stated prerequisites. Check model parameters against the backend and scores against the named evaluation. |

## What CI runs

Every push checks Python formatting and lint, runs the core Python tests and
JavaScript checks, builds the distributions and installs the wheel outside the
checkout. Pull requests into `main` also exercise scientific dependencies,
PostgreSQL, HTTPS S3, Chromium and the Compose deployment.

For a direct push to `main`, run **Actions → CI → Run workflow** on that
revision with **full_checks** enabled before releasing. The workflow lists the
dependency groups, service versions and commands for each job.
[CI workflow](../.github/workflows/ci.yml)
