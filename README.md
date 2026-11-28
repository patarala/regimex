# Regimex

Regimex finds periods of growth, decline and sideways movement in completed
market simulations. Upload a JSON export to see the intervals alongside the
price chart, compare models on the same run or download an interactive HTML report.

## Analyze your first experiment

1. Sign in to the service with your access token.
2. In **Analyses → New analysis**, choose your export or select **Try an example**
   to load a 256-step ADA/USD market simulation. Choose a published model.
3. Select **Analyze**. The report appears when the job finishes. Use
   **Download JSON** for predictions and statistics on the original time grid.

For a new installation, [start the service with Docker Compose](docs/deployment.md)
and register or train its first model. To try the export reader offline from
a checkout, use Python 3.14.4 and uv 0.13.0.

```sh
uv run --locked regimex validate --input tests/fixtures/experiment-v2.json
```

A successful check prints `Valid snapshot: snapshot-1`.

## Documentation

- [Analyze an experiment and read its report](docs/usage.md)
- [Train, evaluate and publish a model](docs/training.md)
- [Connect a simulator](docs/integration.md)
- [Deploy across application nodes](docs/multi-host.md)
- [Browse all guides and references](docs/README.md)

## License

[MIT](LICENSE). Include the generated `LICENSES.md` when sharing an offline report directory.
