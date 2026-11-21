# Regimex

Regimex labels periods of growth, decline and sideways movement in completed
market experiments. Upload a run, choose a saved model and see its intervals
on the price chart. You can compare two models on the same observations and
download an HTML report that works offline.

## Getting started

**With a running service**, sign in with your access token. Under **New
analysis**, choose a JSON export, select a model and click **Analyze**. Once
the job finishes, open the report or download it. [Using Regimex](docs/usage.md#analyze-an-experiment)

**From a checkout**, use Python 3.14.4 and uv 0.13.0. This command validates
the included export locally and prints `Valid snapshot: snapshot-1`.

```sh
uv run --locked regimex validate --input tests/fixtures/experiment-v2.json
```

To host the service, use [Docker Compose](docs/deployment.md).
For development, [run the API and workers from source](docs/usage.md#start-locally).
To create an experiment to analyze, use [`regimex generate`](docs/usage.md#terminal-tools).
If you have the reference TCN asset pack, you can preview a classification
directly in a browser by opening `examples/1-instrument/tcn-report.html` inside the pack.

## Data and models

A version 2.1 export contains one completed run with one or two instruments,
up to 4,096 time steps and a maximum file size of 16 MiB. An analysis preserves
that grid and returns direction intervals, class probabilities and trading
statistics. Steps with missing observations and steps with known zero activity
have different meanings. [Export contract](docs/integration.md#export-contract) ·
[Interval definitions](docs/models.md#observations-and-intervals)

| Model | How it uses an instrument's history |
| --- | --- |
| [Logistic baseline](docs/baseline-context.md) | Classifies each step using summaries of price, trading activity and quotes over centered windows. |
| [Temporal convolutional network (TCN)](docs/tcn.md) | Combines price and return features across several time scales with convolutions that look before and after each step. |

Both models feed the same interval decoder. On 72 held-out synthetic
experiments with profile `direction-32/2.1`, interval macro-F1 against automatic
labels was 0.408 for the baseline and 0.669 for the TCN. The quality guide
also reports how closely the reference TCN placed interval boundaries.
[Model quality](docs/quality.md)

## Guides

| Task | Guide |
| --- | --- |
| Understand the data flow and design choices | [Architecture](docs/architecture.md) |
| Train, evaluate and publish a model | [Training screen](docs/usage.md#train-a-model) |
| Automate analysis or model training | [API workflows](docs/api.md) |
| Connect a simulator and manage its keys | [Integration protocol](docs/integration.md) · [Platform access](docs/access.md) |
| Configure storage, back up data or recover an installation | [Storage and recovery](docs/storage.md) |
| Compare worker capacity and measure queueing under sustained load | [Load testing](docs/load-testing.md) |
| Work on the code and run the relevant tests | [Development](docs/development.md) |
| Package the service and its optional assets | [Building a release](docs/releasing.md) |

## License

[MIT](LICENSE). When sharing a report directory, include its `LICENSES.md`
with the HTML and JavaScript.
