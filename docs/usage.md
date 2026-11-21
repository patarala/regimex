# Using Regimex

With a running service, all you need is its address, your access token and
the JSON export of a completed experiment. The [Docker Compose guide](deployment.md)
uses `https://localhost:8443`. The [source setup](#start-locally) below
serves `http://127.0.0.1:8000`.

## Analyze an experiment

1. Paste your token into **Access token** and select **Sign in**. The
   setup guides save the user token in `var/credentials/user.token`.
2. Under **New analysis**, select **Choose JSON files** or drop an export there.
   For a first run from a checkout, [generate a 512-step example](#terminal-tools)
   and choose `var/run.json`.
3. Check the run and instruments, select a published model, then choose
   **Analyze**. The analysis keeps that model version.
4. When the job finishes, read the preview or select **Download HTML** for
   an offline copy. **Download JSON** saves the predictions and statistics.

To analyze several exports with the same model, select or drop the files together.
Select **Analyze files** (the button includes the count). Keep the batch open while
files are waiting to be submitted. Each row shows its status and opens the accepted
analysis. The browser keeps up to two analyses in progress. If a quota or a service
interruption pauses submissions, use **Resume batch**. Choose **Retry analysis**
on a failed file to run it again using its saved snapshot and original model.
Once the batch finishes, **Retry failed** retries all failed jobs together.
Accepted analyses stay in **Experiments** when you close the batch.

You can sign out and return to **Experiments** later. A session lasts 30 minutes
by default. Select an experiment to open its latest matching analysis, or
expand the analysis count to pick a particular run. Search by name, instrument,
run, model or analysis ID. **Rename** changes the display name in your workspace.

## Read a report

To try a saved classification without starting the service, open
`examples/index.html` inside the reference TCN asset pack. In an unpacked
source release containing those assets, the same page is at `release/examples/index.html`.

For an experiment with two instruments, the report shows both sets of
predicted intervals on a shared timeline. In the service and its downloaded HTML,
select an interval on the timeline or activate its step range in the table to
zoom all charts to that interval. Choose an instrument in the dropdown to switch
prices and trading activity while keeping the current zoom.
The other lane stays visible, so you can see when the two directions diverged.
For a single instrument the report opens directly on its price history.
Drag to zoom or use **Reset zoom** to return to the full run.

The HTML downloaded from the service also shows the model's probability for
each direction at each step. Its **Download JSON** button saves the original
snapshot under `export` and the predictions under `result`. In a multi-experiment
CLI report, select an experiment, then an instrument. Select an interval in the
table to zoom in on it.

A direction applies from the start step up to, but excluding, the stop step.
`no_trades` marks a trading pause and `insufficient_data` marks a period with
too little information to assign a direction. Return standard deviation
measures the spread of adjacent log price changes within an interval. The
missing-data plot shows fields with gaps. For quotes, an empty book side means
there was no resting order on that side. See [interval definitions](models.md).

When absolute timestamps are too close together to plot accurately, the analysis
and comparison reports use bins from the start. The instrument details show the
origin and bin width in the original time unit. If a partial final bin is too
small even on the relative scale, the axis shows equal-width bins and is labelled
“Bin index”.

## Run again and compare

Open a finished analysis, choose a model under **Analyze this experiment
again**, and select **Run again**. This creates another analysis of the saved
snapshot. To compare the two, expand the experiment's analysis count, select
two successful runs and choose **Compare selected**. You can also open one
result and choose a second under **Compare with another analysis**.

The price chart shows both sets of intervals on the same grid. The table
reports where the models disagree about direction or data availability.
Select a step range in the table to zoom both predictions and the price chart.
**Download comparison** saves a self-contained HTML file. Both results must
belong to the same saved snapshot. To compare the supplied baseline and TCN,
both versions must have been published or activated.

## Train a model

Sign in with an administrator token and open **Training**. The service needs a
running `admin` worker. If this is your first training run, use the API to
create a development dataset with training and validation partitions, generate
a reference set for each, and register the dataset as development history.
Reserve a separate final dataset with sealed test references, then start
selection on that reservation. Keep its benchmark ID for the form.
[Data preparation](api.md#1-create-observations-and-references) ·
[Benchmark reservation](api.md#2-reserve-final-data-before-selection)

1. Choose **New run**. Select the development dataset, its training labels and
   validation references. New training runs use the current analysis profile
   (`direction-32/2.2`). Enter the **Selection benchmark ID**.
2. Choose **baseline** or **tcn**. Under **Training settings**, you can edit
   the architecture parameters, training policy and optional CPU runtime
   overrides as JSON. Use the training runtime to set a time or memory budget
   within your server's limits. Select **Start
   training** to queue the run on CPU. You can close the page and return later.
3. Select the run to see its stage and attempt, or use **Cancel training** while
   it is queued or running. After it succeeds, check the selected validation
   interval macro-F1. The baseline plots this score against regularization.
   The TCN plots loss and validation scores by epoch or optimizer update.
   Use **Training seed** to switch between seeds when a run used several.
4. Choose **Evaluate candidate** to score its saved weights on the same
   validation references. Open **Evaluation JSON** for the full metrics.
   To make this model the default for new analyses, choose **Publish candidate**
   and confirm the active model it will replace.

You can train and compare several candidates under one reservation. The
training screen keeps the final test references sealed. Once you have chosen the
model for the final evaluation, freeze it and open the benchmark through the
API. [Final evaluation](api.md#4-freeze-and-evaluate-final-data)

## Delete saved work

Use **Delete analysis** under **Details** to remove one result, or **Delete
experiment** in history to remove the export and its analyses. The preview
lists the affected resources and any jobs or datasets that block deletion.
Download anything you want to keep before confirming **Delete**.

Files awaiting cleanup still count toward your quota. If removal was
interrupted, reopen the receipt under **Pending deletions** and select **Retry
cleanup**. If the server response was lost, select **Check deletion** to find
out whether it accepted the request.

## Terminal tools

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

Waiting stops after 300 seconds by default. Use `--timeout SECONDS` to change it.
An accepted analysis continues on the server after you stop waiting.

To use the predictions in another program or a multi-experiment report, choose
`--format json`. You can request JSON on the initial submission or download it
later using the analysis ID printed by the command. Both formats come from the
same saved analysis.

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

## Start locally

To run from source, use Linux with Python 3.14.4, uv 0.13.0 and PostgreSQL 16.
A PostgreSQL administrator can create the role and database on a running server.

```sh
sudo -u postgres createuser --pwprompt regimex
sudo -u postgres createdb --owner=regimex regimex
```

Put the password in a private `~/.pgpass` file with mode `0600`. For the
connection below the entry is `127.0.0.1:5432:regimex:regimex:YOUR_PASSWORD`.
Prepare the local configuration and issue the tokens from the checkout root.

```sh
uv sync --locked --all-extras
umask 077
mkdir -p var/credentials var/objects var/work
cat > var/environment.sh <<'EOF'
export REGIMEX_DATABASE_DSN='host=127.0.0.1 port=5432 dbname=regimex user=regimex'
export REGIMEX_OBJECT_ROOT="$PWD/var/objects"
export REGIMEX_WORK_ROOT="$PWD/var/work"
export REGIMEX_PUBLIC_ORIGIN='http://127.0.0.1:8000'
export REGIMEX_ALLOW_LOOPBACK_HTTP=true
EOF
. var/environment.sh
uv run --locked --all-extras regimex service migrate
uv run --locked --all-extras regimex service token-create --owner admin --role admin --output var/credentials/admin.token
uv run --locked --all-extras regimex service token-create --owner researcher --role user --output var/credentials/user.token
```

If you have the release assets in `release/`, verify their checksums and import
the supplied TCN. For a source-only installation, the [administrative
API](api.md#administration) can train and publish a model.

```sh
(cd release && sha256sum -c SHA256SUMS)
```

Once all files pass verification, register and activate the model.

```sh
MODEL_MANIFEST_SHA256=$(sha256sum release/models/tcn/manifest.json | cut -d ' ' -f 1)
uv run --locked --all-extras regimex service register-model \
  --bundle release/models/tcn --manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --token-file var/credentials/admin.token --activate
```

Open four terminals in the checkout root and run `. var/environment.sh` in
each. Start one process per terminal.

```sh
uv run --locked --all-extras regimex service serve
uv run --locked --all-extras regimex service worker --queue analysis
uv run --locked --all-extras regimex service worker --queue admin
uv run --locked --all-extras regimex service worker --queue maintenance
```

Open `http://127.0.0.1:8000/` to sign in. The API reference is at `/docs`.
Ctrl+C stops each process. Keep the PostgreSQL database and `var/objects` to
retain saved work.

## If something fails

| Symptom | What to do |
| --- | --- |
| Sign-in fails | Use the full user token and the configured origin. If the session expired, sign in again. |
| Export rejected | Check the field named in the error. A completed version 2.1 export may have up to two instruments, 4,096 steps and a file size of 16 MiB. |
| No active model | Register and activate a bundle, or publish an evaluated candidate. |
| Analysis stays queued | Check the worker status beside the job. In Compose, run `docker compose --profile ops run --rm ops service doctor`. From source, run `uv run --locked --all-extras regimex service doctor` with the worker environment loaded. |
| Request interrupted | **Retry request** resumes the original submission or status check. |
| Analysis failed | Read the reason beside the status. **Details** includes the failure code and analysis ID. After resolving the cause, use **Run again** to create a new analysis. |
| Storage full | Check the reserved space and **Pending deletions**. Remove unused experiments or ask the administrator to raise the quota. |
