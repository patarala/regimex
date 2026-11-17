# Regimex

Regimex labels periods of growth, decline and sideways movement in completed
market experiments. Upload an experiment, choose a saved model and inspect its
predictions on the price chart. You can compare two models on the same
observations or download a report to open offline.

## Getting started

- **Try a report.** Open a saved HTML report from the release assets in your
  browser. The reference TCN pack includes `examples/1-instrument/tcn-report.html`.
- **Run the service.** The Linux installer sets up a private PostgreSQL
  database, imports the bundled models and starts the API and workers.
  See [local installation](docs/installation.md). For containers, use the
  [Compose deployment guide](docs/deployment.md).
- **Analyze your first file.** Sign in with the token supplied by your
  administrator. Under **New analysis**, choose a JSON file and select
  **Analyze**. The [usage guide](docs/usage.md) covers reports, comparisons,
  saved experiments and running the service from source.

From a source checkout you can validate the included example without starting
a server. To generate your own reproducible experiment, use
[`regimex generate`](docs/usage.md#terminal-tools). The project uses Python
3.14.4 and uv 0.13.0.

```sh
uv run --locked regimex validate --input tests/fixtures/experiment-v2.json
```

## Data and models

A version 2.1 export contains one completed run with one or two instruments,
up to 4,096 time steps and a maximum file size of 16 MiB. An analysis preserves
that grid and returns direction intervals, class probabilities and trading
statistics. The report distinguishes steps with missing observations from
steps with known zero activity. See the [export contract](docs/integration.md)
and [model and interval definitions](docs/models.md).

Regimex includes a market generator and two model backends, logistic regression
and a temporal convolutional network (TCN). On the same 72 held-out synthetic
experiments with profile `direction-32/2.1`, interval macro-F1 against automatic
labels was 0.408 for logistic regression and 0.669 for the TCN. See
[model quality](docs/quality.md) for results by sequence length and the separate
evaluation of the reference TCN distributed with the release assets.

## Running and maintaining Regimex

| Task | Guide |
| --- | --- |
| Understand the data flow and design choices | [Architecture](docs/architecture.md) |
| Automate analysis, train a model or publish a version | [API workflows](docs/api.md) |
| Connect a simulator and manage its keys | [Platform access](docs/access.md) |
| Configure storage, back up data or recover an installation | [Storage and recovery](docs/storage.md) |
| Package the service and its optional assets | [Building a release](docs/releasing.md) |

## Development

The code lives in `src/regimex/`. See [AGENTS.md](AGENTS.md) for development
conventions and local test commands.

Every push runs formatting, lint, core tests, JavaScript checks and packaging.
Pull requests into `main` also run the scientific, PostgreSQL, HTTPS S3 and
container tests. For a direct push to `main`, run **Actions → CI → Run workflow**
with **full_checks** enabled before releasing.

## License

[MIT](LICENSE). When sharing a report directory, include its `LICENSES.md`
with the HTML and JavaScript.
