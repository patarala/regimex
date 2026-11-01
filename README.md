# Regimex

Regimex analyses completed market experiments and marks intervals of price growth,
decline and sideways movement. Upload a JSON export, run a saved model, and view
or download an interactive report. Reports also work offline.

The service includes a baseline model, a temporal convolutional network (TCN),
a reproducible market generator and an administrative API for training,
evaluation and model publication. It runs independently of an external simulator.

## Start here

- [Run the service and analyse a file](docs/usage.md)
- [Architecture and design decisions](docs/architecture.md)
- [API and model administration](docs/api.md)
- [Deployment](docs/deployment.md)

## Installation

Use Python **3.14.4** and uv **0.13.0**:

```sh
uv sync --locked --extra ml --extra viz --extra service --extra s3
uv run --locked regimex --help
```

The service needs PostgreSQL and separate API and worker processes. The
[usage guide](docs/usage.md) walks through database setup, access tokens,
loading a model, starting the processes and using the browser interface.
[Docker Compose](docs/deployment.md) provides the container setup.

Release assets contain saved models and sample exports. Unpack them into
`release/`, or create data and train models through the administrative API.
Training is an explicit action; uploading an experiment never retrains a model.
Runtime data and private credentials belong in the ignored `var/` directory.

## Input and output

The current JSON export contract supports one or two instruments, at most 4,096
time steps and a 16 MiB file. It records prices, trades, orders, quotes, time units
and explicit missingness. Validate an export before using it:

```sh
uv run --locked regimex validate --input experiment.json
uv run --locked regimex schema --output var/schemas
```

Each analysis returns direction intervals, class probabilities, activity and
variability statistics, plus a saved HTML report. Missing observations and
observed zeros keep their distinct meanings. Intervals cover the original time
grid; direction and data availability are separate.

To build an offline report from an export and a downloaded analysis result:

```sh
uv run --locked --extra viz regimex report \
  --input experiment.json --result result.json --output var/report
```

Open `var/report/index.html` and keep its directory together. The browser also
provides a self-contained HTML download for each saved analysis.

## Models and limitations

Both models restore saved preprocessing and weights. Their evaluation uses whole
experiment splits and interval macro-F1. The current TCN scores **0.640**, versus
**0.470** for the baseline on 64 held-out experiments with automatic references.
Boundary placement remains a weakness.
These measurements do not establish independent human accuracy or transfer to a
different simulator. See [evaluation](docs/quality.md) and
[model and interval definitions](docs/models.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Terminal commands, browser controls and common errors |
| [Architecture](docs/architecture.md) | Component diagram, data flow and reasons for the design |
| [API](docs/api.md) | Endpoints, authentication, datasets and model lifecycle |
| [Models](docs/models.md) | Generation, interval semantics, features, training and evaluation |
| [Quality](docs/quality.md) | Current measurements, methodology and limitations |
| [Deployment](docs/deployment.md) | Docker, TLS, resources and local/S3 configuration |
| [Storage](docs/storage.md) | Integrity, retention, backup and restoration |
| [Integration](docs/integration.md) | External export and delegated access contract |

## Development

Production code is in `src/regimex`, behavior tests in `tests`. See
[AGENTS.md](AGENTS.md) for engineering conventions.

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked --extra ml --extra viz --extra service --extra s3 --group service-test mypy src tests
uv run --locked --extra ml --extra viz --extra service --extra s3 --group service-test pytest
uv build --no-build-isolation
```

PostgreSQL and queue tests additionally require `--group queue-test` and a
disposable `REGIMEX_TEST_DATABASE_DSN`. The real S3 and restoration test setup is
described in [storage](docs/storage.md). Tests requiring unavailable services are
reported as skipped; they do not verify those integrations.
