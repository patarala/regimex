# Working on Regimex

Regimex is a Python service for retrospective analysis of completed market
experiments. Read [README.md](README.md) for setup and
[docs/architecture.md](docs/architecture.md) for component responsibilities.

## Repository layout

- `src/regimex/`: contracts, scientific code, HTTP/CLI entry points and web assets.
- `src/regimex/application/`: authorization, resource lifecycle and workflows.
- `src/regimex/backends/`: baseline and temporal model implementations.
- `src/regimex/infrastructure/`: PostgreSQL, queue, migrations and object storage.
- `tests/`: behavior tests and small deliberate fixtures.
- `docs/`: product, model and operator documentation.
- `deploy/`, `Dockerfile`, `compose*.yaml`: deployment configuration.

## Development

Use Python from `.python-version` and uv from `pyproject.toml`. Use `uv` and
`--locked` for checks; retain intentional lockfile changes. Write code, comments
and repository documentation in English. Keep commands in the README executable.
Inspect the current diff before editing and preserve unrelated work.

Prefer small, explicit functions and modules organized by responsibility. Keep
HTTP and CLI handlers thin. Scientific functions receive data and configuration
as arguments; they do not read service settings or perform storage operations.
Do not add compatibility layers, speculative abstractions or placeholder success
responses. Validate configuration and reject settings that have no effect.

Scientific dependencies belong to `ml`, plotting to `viz`, and integration test
tools to explicit development groups. PyTorch uses the explicit CPU index.
Ordinary CLI/API startup and backend discovery must not import optional scientific
libraries. Package web/report assets and migrations with the wheel. Pin GitHub
Actions to reviewed commit SHAs and keep CI permissions read-only.

## Models and data

[docs/models.md](docs/models.md) defines the interval semantics. Preserve the
original time grid, complete interval coverage, and the distinction between
missing values and observed zeros. Direction, activity, variability and data
availability are separate concepts. Do not add jump-event annotations.

Backends own feature preparation, fitting and weight/preprocessing codecs.
Prediction accepts original observations, timing, units and missingness; never
pass targets, reference boundaries, seeds or generator settings. Discover only
trusted installed implementations through `regimex.models`; reject incompatible
interfaces, devices, class order, output grids and codecs.

Keep raw observations, ordered dataset membership, split assignments, training
labels and evaluation references separately identified. Split whole experiments
or seed groups before windows, keeping instruments together. Fit preprocessing
on training data only. Select settings and checkpoints on decoded validation
segment F1. Freeze choices before opening a final benchmark; any subsequent
search requires a fresh benchmark checked against prior exposures.

Record seeds, data/configuration hashes and environment versions. Restore
preprocessing with weights; prediction must not retrain. Automatic labels are a
training convention, not independent human truth. Report measured quality,
uncertainty and boundary weaknesses. Mocks cannot establish transfer to another
simulator.

Simulated prices, volumes and quotes come from executions and resting orders.
Preserve price/time priority, partial fills, reservation release, asset
conservation and shared quote balances. Do not schedule regimes to raise scores.

## Persistence and security

PostgreSQL is authoritative for owners, lifecycle, jobs and publication. Immutable
object storage holds observations, models and reports. Enqueue the operation and
its task in one supported transaction. Keep digest-bound JSON numbers faithful;
never replace a committed digest to hide a mismatch.

Assume retries and worker death. Preserve attempt fencing, heartbeats,
cancellation and bounded recovery. Stale attempts cannot commit. Verify objects
before making them visible. Resolve active models to immutable versions at intake.
Keep training, fixed-model evaluation and explicit publication separate.

Enforce owner/run/source checks on resources and downloads. Scope idempotency by
owner and operation. Keep integration credentials on the backend. Preserve session
and CSRF checks; never fetch arbitrary user URLs or persist presigned URLs.
Apply Torch/BLAS settings and computation limits in isolated worker processes.

Deletion requires authorization and reference checks. Preserve registered
objects during cleanup and fence exact orphan generations. Backup/restore must
coordinate database and objects; partial restoration blocks readiness and intake.
See [docs/storage.md](docs/storage.md) for provider and maintenance guarantees.

## Verification

Run checks appropriate to the change:

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked --extra ml --extra viz --extra service --extra s3 --group service-test mypy src tests
uv run --locked --extra ml --extra viz --extra service --extra s3 --group service-test pytest
uv build --no-build-isolation
```

Database/queue tests additionally use `--group queue-test` and a disposable
`REGIMEX_TEST_DATABASE_DSN`. Run the real HTTPS S3 provider with
`tests/support/run_storage_tests.py --tls`; recovery tests need compatible
`PGDUMP_BIN` and `PGRESTORE_BIN`. Disclose skipped or unavailable checks.

After dependency changes, verify imports and a CPU tensor operation. After
packaging changes, install the wheel outside the source tree. After container
changes, build and run the Compose smoke when Docker is available.

Test observable behavior and critical invariants, not implementation details.
After UI/report changes, check JavaScript syntax and actual browser controls at
desktop and narrow widths, including offline reports, missing data and linked
zoom. Keep errors explicit; do not silently skip failed instruments.

## Files

Keep credentials, generated data, models, reports, local inputs and test evidence
out of Git and container build contexts. Runtime data belongs in `var/`; generated
release assets belong in `release/`. Keep only small intentional fixtures in tests.
Do not publish artifacts or change remotes during local development.
