# Using Regimex

With a running service, all you need is its address, your access token and
the JSON export of a completed experiment. The [Docker Compose guide](deployment.md)
uses `https://localhost:8443`. The [source setup](#start-locally) below
serves `http://127.0.0.1:8000`.

| What you want to do | Where to start |
| --- | --- |
| Analyze one export or a batch | **Analyses → New analysis** |
| Find a result, rerun a model, or compare two runs | **Analyses → Experiments** |
| Remove a result or a saved experiment | [Delete saved work](#delete-saved-work) |
| Train, evaluate, or publish a model | [Training](#train-a-model) (administrator) |
| Connect a simulator and manage its service keys | [Connections](access.md#manage-connections-in-the-browser) (administrator) |

## Analyze an experiment

1. Paste your token into **Access token** and select **Sign in**. The
   setup guides save the user token in `var/credentials/user.token`.
2. Under **New analysis**, select **Choose JSON files** or drop an export there.
   For a first run from a checkout, [generate a 512-step example](#terminal-tools)
   and choose `var/run.json`.
3. Check the run and instruments, select a published model, then choose
   **Analyze**. The analysis keeps that model version.
4. When the job finishes, read the preview or select **Download report** for
   an offline copy. **Download JSON** saves the predictions and statistics.

To analyze several exports with the same model, select or drop the files together.
Select **Analyze files** (the button includes the count). Keep the batch open while
files are waiting to be submitted. Each row shows its status and opens the accepted
analysis. The browser keeps up to two analyses in progress. If a quota or a service
interruption pauses submissions, use **Resume batch**. Choose **Retry analysis**
on a failed file to run it again using its saved snapshot and original model.
Once the batch finishes, **Retry failed** retries all failed jobs together.
Accepted analyses stay in **Experiments** when you close the batch.

Use the browser’s Back and Forward buttons to return to an analysis or comparison.
Bookmark its URL to reopen it after signing in with the same account. A session
lasts 30 minutes by default. Select an experiment to open its latest matching
analysis, or expand the analysis count to pick a particular run. Search by name, instrument,
run, model or analysis ID. Choose **Manage → Rename** to change the display name.

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
selection on that reservation. The form lists the benchmarks available to your account.
[Data preparation](api.md#1-create-observations-and-references) ·
[Benchmark reservation](api.md#2-reserve-final-data-before-selection)

1. Choose **New run**. You can name the run after its data or the change you
   are testing, for example `EUR/USD quiet sessions`. Select a development
   dataset. The form shows its compatible training labels, validation
   references and selection benchmarks.
   If data is missing, use the setup links beside the fields.
2. Choose **Logistic baseline** or **Temporal convolutional network (TCN)**.
   **Training settings** lets you edit architecture parameters, the training
   policy and CPU runtime limits. Choose **Start training** to queue the run.
   You can bookmark the run or leave and return while the worker trains it.
   If you leave a form before submitting, browser Back restores the draft in the same tab.
3. Select a run from the list, or open **Run history** on a phone. Unnamed
   runs show the architecture and dataset. Choose **Rename** to change the
   name, or clear the field and save to restore the automatic title. You can
   cancel a run while it is queued or running.
   The page checks active jobs about every 2.5 seconds. Charts appear as the
   worker records measurements. You can reopen the run to catch up or expand
   **Values** under a chart to read individual measurements. If the worker
   retries training, the charts start again with the new attempt.

   The baseline scores each regularization setting on the validation data.
   The TCN records training loss at every epoch or optimizer update. With an
   epoch budget, it checks validation interval macro-F1 at epoch 1, every
   `validation_interval` epochs, and at the final scheduled epoch if reached.
   With an update budget, it checks the updates listed in `validation_updates`.

   For a TCN with several repeats, the charts follow the repeat being trained.
   Choose a **Training repeat** to keep its curves on screen. Each seed sets
   the starting weights and training order. After training, the repeat marked
   **saved model** produced the checkpoint with the best validation interval
   macro-F1. Its score appears above the charts.
4. Under **Model evaluation**, choose **Evaluate model** to score the saved
   weights on the validation references. When the evaluation finishes, its
   interval and step scores and results by direction appear on the run page.
   Opening the run again shows the saved result. **All fields** shows the
   full report in the browser, and **Download JSON** saves a copy. If the
   evaluation fails or is cancelled, choose **Retry evaluation**.
   To make this model the default for new analyses, choose **Use as default**
   and confirm the change.

You can train and compare several candidates under one reservation. The
training screen keeps the final test references sealed. Once you have chosen the
model for the final evaluation, freeze it and open the benchmark through the
API. [Final evaluation](api.md#4-freeze-and-evaluate-final-data)

## Delete saved work

Use **Delete analysis** under **Details** to remove one result. In the
experiment list, choose **Manage → Delete experiment** to remove the export
and its analyses. The preview lists the affected resources and any jobs or
datasets that block deletion.
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
