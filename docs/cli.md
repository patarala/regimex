# Command-line workflows

Run commands from the checkout root. For a remote service, use its HTTPS origin
and the token file for your account.

## Generate and analyze an experiment

To try a market of your own, generate one completed experiment locally. The seed,
step count and instrument count determine its contents. With the same generator
version and locked dependencies, repeating the command gives the same snapshot
ID and checksum. Choose a new output filename for each invocation.

```sh
uv run --locked --extra ml regimex generate --seed 42 --steps 512 --instruments 2 --output var/run.json
```

Generation runs offline and supports 1–4,096 steps and one or two instruments.
With Regimex running, submit the generated file and save its HTML report:

```sh
uv run --locked regimex analyze \
  --input var/run.json --output var/analysis.html
```

The default service is `http://127.0.0.1:8000` and the token file is
`var/credentials/user.token`. For a remote deployment, pass its HTTPS
`--origin` and your `--token-file`. Use `--model` to select a published
version. The default is the model active when the service accepts the job.
Choose a new output filename.

The command prints an idempotency key before submission and an analysis ID
once accepted. If the connection drops before you receive an ID, repeat the
submission with the same input, model and `--idempotency-key`. Once you have
an ID, resume the download without submitting again:

```sh
uv run --locked regimex analyze --analysis-id ANALYSIS_UUID --output var/analysis.html
```

The CLI retries a status check or download up to twice after a connection
failure or a temporary HTTP response (408, 429, 502, 503 or 504). It honors
`Retry-After` within the overall time limit. Waiting stops after 300 seconds
by default. Use `--timeout SECONDS` to change it. An accepted analysis
continues on the server after you stop waiting.

To use the predictions in another program or a multi-experiment report, choose
`--format json`. You can request JSON on the initial submission or download it
later using the analysis ID printed by the command. Before saving JSON, the CLI
checks the source run, snapshot and model against the job. It also verifies
the inference settings against the pinned policy digest and checks the instruments
and step count when the snapshot records them.
The offline report shows the SHA-256 of the canonical
result JSON. You can reformat the file while preserving that digest. Keep
numeric values as written, including the distinction between `1` and `1.0`.

```sh
uv run --locked regimex analyze --analysis-id ANALYSIS_UUID --format json --output var/result.json
uv run --locked --extra viz regimex report --input var/run.json --result var/result.json --output var/run-report
```

To validate an export or write JSON schemas locally:

```sh
uv run --locked regimex validate --input tests/fixtures/experiment-v2.json
uv run --locked regimex schema --output var/schemas
```

To build an offline report, supply one saved result per export, all from
the same model and inference policy. This example uses the reference TCN
pack unpacked under `release/`.

```sh
uv run --locked --extra viz regimex report \
  --input release/examples/1-instrument/snapshot.json --result release/examples/1-instrument/tcn-result.json \
  --input release/examples/2-instrument/snapshot.json --result release/examples/2-instrument/tcn-result.json \
  --output var/report
```

Choose a new or empty output directory. Open `var/report/index.html` and keep
the directory together when sharing it. For larger sets, `--samples` controls
how many experiments appear in the report (default 12, maximum 100).

## Choose where training runs

| How you start | Where it runs | Where to find it |
| --- | --- | --- |
| **Start training** in the browser, or `regimex submit-training` | Service worker | **Training** shows the queue, progress and outcome from submission onward. |
| A local fitting or comparison script | Your machine | The files stay in the chosen output directory. You can run offline. |
| Local fitting with bundle registration enabled | Your machine | **Training** shows the verified result under the administrator account that imported it. |

To submit the same request as the web form, save its complete configuration
as `var/training.json`, including the dataset, references and benchmark IDs.
The CLI prints the run link as soon as the service accepts it. Reuse the
printed `--idempotency-key` if the response is lost. [Request example](api.md#3-train-and-publish)

```sh
uv run --locked regimex submit-training --input var/training.json \
  --origin http://127.0.0.1:8000 --token-file var/credentials/admin.token
```

To fit on your machine, run the [offline forest workflow](large-direction-forest.md#reproduce)
or a local baseline or TCN comparison. In the current checkout, the scripts
that produce supported bundles can also record the result in your service.
Load its operator environment and set `REGIMEX_TRAINING_TOKEN_FILE` to the
administrator token file in that shell. Use both `--extra ml` and
`--extra service` with `uv run` so the same environment can fit the model
and register its bundle. Add `--extra s3` for
an S3-backed deployment. Unset the variable to return to offline-only fitting.
Training uses local resources in both cases.

An existing bundle can be added later with `regimex service register-model`.
Supply its recorded manifest SHA-256 and the administrator token for the
intended workspace. The command prints the link to its **Imported** run.
The page shows the saved settings, validation score, history and weight
identity. For the forest comparisons you can also attach the
[final scores and inference timings](large-direction-forest.md#import-a-completed-comparison).
Repeating registration recovers the same run. Use `--activate` only when
you intend to change the default model.

To add a chart to a forest with only a final score in its history, measure
the original validation examples with increasing numbers of trees from the
saved bundle. Pass the resulting JSON to `regimex service register-model`
with `--forest-validation-curve PATH`, using the bundle's original manifest
digest and inference policy. The run displays these measurements as coming
from saved weights. [Curve format](models.md#adaptive-direction-forest)

## Compare training and independent scores

Open a completed run in **Training**. When the run has a direction quality
report, **Interval macro-F1 by reference** shows the score and number of groups
in each measured split. **Simulator agreement** compares predictions with
automatic labels on simulator observations. **Known direction** compares them
with the drift specified when generating controlled prices. Read across one
row to compare training, validation and fresh groups using the same kind of reference.

For a saved model with its original automatic reference sets, the following
command scores the validation split offline. Set the paths to the files from
the fit. In the locations JSON, group exports by `source_namespace` and use
each manifest member's `snapshot_id` as the key. Paths are relative to the
locations file. A source with one snapshot in the measured split can also
point directly to its file.

```json
{
  "trade": {
    "snapshot-17": "exports/run-17.json",
    "snapshot-18": "exports/run-18.json"
  }
}
```

Add a second `--cohort` with the training references and locations to get
both scores in one report.

```sh
uv run --locked --extra ml python scripts/measure_direction_quality.py \
  --bundle "$BUNDLE" --manifest-sha256 "$BUNDLE_SHA256" --protocol "$PLAN" \
  --cohort "$DATASET" "$VALIDATION_REFERENCES" "$LOCATIONS" --output var/quality.json
```

To display the report, run the following in the service's operator environment.
For a bundle registered with a custom inference policy, also pass the original
`--runtime-file`. The measurements appear on the existing run.

```sh
uv run --locked --extra ml --extra service regimex service register-model \
  --bundle "$BUNDLE" --manifest-sha256 "$BUNDLE_SHA256" \
  --token-file "$REGIMEX_TRAINING_TOKEN_FILE" --direction-quality var/quality.json
```

Independent measurements come from an evaluator using groups reserved outside
training and selection. Keep its evaluation plan, reference manifest and full
JSON report alongside the weights. [Direction quality report format](models.md#direction-quality-reports)

## Find a run started in the terminal

Open the link printed by `regimex submit-training` or `regimex analyze`, or
sign in as the same owner and open **Training** or **Analyses**. In the
training run, open **Run details and settings → Started via**. In an analysis,
open **Details → Launch source**. The source identifies the web interface, an
API call, a simulator integration or the name of the CLI command. Opening
the job from another client keeps its original source. Older runs may have
no recorded source.

For an offline model, **Imported via** identifies the command that registered
the saved bundle. The training settings and validation score come from the
bundle. Direct local commands such as `regimex generate`, `validate`
and `report` work without a service. To use a generated experiment in the
workspace, submit its export for analysis.

Use `REGIMEX_CLI_ORIGIN` and `REGIMEX_CLI_TOKEN_FILE` to select the service
and account for analysis and training submissions in your shell.
Explicit `--origin` and `--token-file` options take precedence.
