# Regimex agent instructions

Check the current branch and diff before editing. Work on a topic branch and
preserve unrelated edits. Use the Python and uv versions pinned in
`.python-version` and `pyproject.toml`. Write code, comments and docs in English.
Use comments to explain constraints such as why an upload must finish before
its database transaction commits.

## Where to make the change

- Keep HTTP and CLI handlers thin. Put workflows in `src/regimex/application/`,
  SQL and object storage in `src/regimex/infrastructure/`, and browser controls
  in `src/regimex/web/`. For report rendering and worker entry points, see the
  [code map](docs/architecture.md#where-to-find-the-code).
- Put model-specific preparation, training and weight codecs in `src/regimex/backends/`.
  Pass data and configuration to scientific functions explicitly. Keep CLI
  startup and backend discovery usable without NumPy, Torch or Plotly. Keep
  scientific dependencies in `ml`, plotting in `viz` and test tools in development
  groups. Preserve the explicit CPU index for PyTorch.
- Keep credentials and runtime data in `var/`, generated release assets in
  `release/`, and both out of Git and the container build context. Use small,
  intentional fixtures in `tests/`.

## Contracts to preserve

- **Observations and simulation.** Preserve the original time grid and complete
  interval coverage. Distinguish missing values from observed zeros, and keep
  direction, activity, variability and availability separate. Omit jump-event
  annotations. Preserve price/time priority, partial fills, reservations and
  shared quote balances. [Models and interval semantics](docs/models.md)
- **Training and inference.** Split by whole experiment or seed group before
  windowing. Fit preprocessing on training data only. Select settings by decoded
  validation interval macro-F1. Freeze choices before opening the final benchmark
  and use a fresh benchmark for further selection. At inference load only
  trusted installed backends, verify the bundle's codec and class order, and
  restore both weights and preprocessing. Keep labels, reference boundaries,
  seeds and generator settings out of prediction inputs. [Evaluation protocol and results](docs/quality.md)
- **Jobs.** Pin the model version at intake. Commit the operation and queue task
  together. Scope idempotency by owner and operation. Renew the lease during
  I/O, verify output objects before publication, and check the attempt token and
  cancellation state in the transaction that publishes the result.
  [Analysis lifecycle](docs/architecture.md#from-upload-to-report)
- **Access and storage.** Enforce owner, source and run checks on resources and
  downloads. Keep integration keys on the backend and preserve session and CSRF
  checks. Preserve digest-bound JSON numbers. Check references and exact object
  generations before cleanup. Block readiness and intake during a partial restore.
  [Integration contract](docs/integration.md) · [Storage and recovery](docs/storage.md)

## Verify the change

Start with the affected test file. For example, after a TCN backend change run
`uv run --locked --extra ml pytest tests/test_tcn_backend.py`. For documentation,
check changed links and headings, run changed command examples where possible,
and verify model settings against the backend or the named release bundle.

Use the [development guide](docs/development.md) for local checks and the full CI
setup. Report any skipped integrations. When editing the [CI workflow](.github/workflows/ci.yml),
pin actions to reviewed commit SHAs and keep permissions read-only.
