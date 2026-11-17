# Regimex

Use the Python and uv versions pinned in `.python-version` and `pyproject.toml`.
Check the current branch and diff before editing. Work on topic branches,
preserve unrelated edits, and write code, comments and docs in English.
Use comments for constraints a maintainer could miss, such as why an upload
must finish before a database commit. Omit narration of the next line.

## Change boundaries

- Keep HTTP and CLI handlers thin. Put workflows in `src/regimex/application/`,
  SQL and object storage in `src/regimex/infrastructure/`, model-specific code
  in `src/regimex/backends/`, and browser controls in `src/regimex/web/`.
  See [architecture](docs/architecture.md) for the analysis flow.
- Scientific functions take data and configuration as arguments. CLI startup
  and backend discovery must work without NumPy, Torch or Plotly. Use the
  `ml` and `viz` extras for those dependencies and development groups for tests.
- Keep runtime data and credentials in `var/` and generated release assets in
  `release/`. Exclude both from Git and container build contexts. Keep only
  small, intentional fixtures in `tests/`.

## Invariants

- Preserve the original **observation grid**, complete interval coverage,
  and the difference between missing and zero. Keep availability, direction,
  activity and variability separate. Do not add jump-event annotations.
  Preserve price/time priority, partial fills, reservations and shared quote
  balances when changing the simulator.
- For **model training**, split by whole experiment or seed group before
  windowing. Fit preprocessing on training data only. Select settings by
  decoded validation segment F1. Freeze choices before opening a final
  benchmark and use a fresh benchmark for further selection. Restore saved preprocessing
  and weights at inference. Prediction inputs exclude labels, reference boundaries,
  seeds and generator settings. See [model contracts](docs/models.md).
- Resolve the active model when accepting a **job**. Commit the operation
  and queue task together. Scope idempotency keys by owner and operation. Verify
  objects before publication, and check the attempt token and cancellation
  state in the commit transaction. Renew the lease during I/O.
- Check **access** by owner, source and run on resources and downloads.
  Keep integration keys on the backend and enforce session and CSRF checks.
  Preserve digest-bound JSON numbers. Check references and exact object
  generations before cleanup. A partial restore blocks readiness and intake.
  See [storage guarantees](docs/storage.md).

## Verification

Run the checks that cover the change and report any skipped integrations.

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked --all-extras --all-groups mypy src tests
uv run --locked --all-extras --all-groups pytest
```

- Use a disposable `REGIMEX_TEST_DATABASE_DSN` for database and queue tests.
  For HTTPS S3 and recovery, follow the [provider test setup](docs/storage.md#integration-test-setup).
- For browser and report changes, run `node --experimental-vm-modules --test tests/*.test.cjs`.
  Try changed controls at desktop and narrow widths. For reports, check offline
  loading, linked zoom and missing data.
- Preserve the CPU PyTorch index. Test scientific imports and a CPU tensor
  operation after dependency changes. After packaging changes run
  `uv build --no-build-isolation` and test the installed wheel outside the checkout.
- Run `python tests/support/compose_service_smoke.py` after container
  changes. Pin CI actions to reviewed SHAs and keep permissions
  read-only. Use full CI for release verification.
