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

When [running from source](usage.md#start-locally), edit the workspace HTML,
JavaScript or CSS under `src/regimex/web/` and reload the browser. The API
reads these files on request, including newly added top-level modules. Changes
to Python handlers take effect when you restart the API process. In a packaged
Compose deployment, use the [component update procedure](deployment.md#update-individual-components)
to deploy the corresponding image.

## Checks by change

| Change | What to run |
| --- | --- |
| Model features, training or inference | Run the affected backend tests, e.g. `uv run --locked --extra ml pytest tests/test_adaptive_backend.py`. The baseline and TCN suites are `tests/test_baseline_backend.py` and `tests/test_tcn_backend.py`. For update-based TCN training or interval decoding, include `tests/test_fixed_update_tcn.py` or `tests/test_segmentation.py`. |
| Python types, including optional dependencies | `uv run --locked --all-extras --all-groups mypy src tests` |
| Workspace or report controls | `node --experimental-vm-modules --test tests/*.test.cjs`. Run the affected `tests/*_browser.py` suite with Chromium to check layout, navigation or offline reports. |
| PostgreSQL, queue, backup or object storage | Use the [integration test setup](storage.md#integration-test-setup) for a disposable database, recovery tools and the HTTPS S3 fixture. |
| Dependencies or packaging | After changing scientific dependencies, run the scientific imports and CPU tensor check in CI. After changing package contents, run `uv build --no-build-isolation` and test the installed wheel outside the checkout. [CI commands](../.github/workflows/ci.yml) |
| Container startup or deployment | Follow the [container smoke test](deployment.md#verify-a-container-installation) to run an analysis and restore it into a fresh deployment. |
| Documentation | Follow the changed links from the README and run the examples with their stated prerequisites. Check model parameters against the backend and scores against the named evaluation. |

For a browser test, install Chromium once in the Playwright environment.
On Linux with the browser's system libraries installed, run e.g.

```sh
uv run --locked --group browser-test playwright install --only-shell chromium
REGIMEX_REQUIRE_BROWSER_TESTS=1 uv run --locked --all-extras --group browser-test \
  pytest tests/test_web_startup_browser.py
```

## What CI runs

Every push checks Python formatting and lint, runs the core Python tests and
JavaScript checks, builds the distributions and installs the wheel outside the
checkout. Pull requests into `main` also exercise scientific dependencies,
PostgreSQL, HTTPS S3, Chromium and the Compose deployment.

To run the full checks before releasing a direct push to `main`, open
**Actions → CI → Run workflow**, choose the branch and enable **Run scientific,
storage and container checks**. The manual run checks the branch tip at dispatch.
The workflow lists the dependency groups, service versions and commands for each job.
[CI workflow](../.github/workflows/ci.yml)
