# Using Regimex

With a running service, all you need is its address, your access token and
the JSON export of a completed experiment. The [Docker Compose guide](deployment.md)
uses `https://localhost:8443`. The [source setup](#start-locally) below
serves `http://127.0.0.1:8000`.

The sun or moon in the header switches between light and dark. Regimex follows
your device by default and resumes following it when you switch back to its theme.

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
   You can leave while the worker runs. Returning to the analysis refreshes
   its status. Temporary connection errors are retried automatically.

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
If a saved experiment has no analyses, choose **Set up analysis** beside it.
Select a model under **New analysis** and choose **Analyze** to use that saved snapshot.

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

Select **Pauses within trends** on the price chart to highlight short stretches
where trades continued at roughly the same price before growth or decline
resumed. Hover over the price line to see the duration and the change between
the first and last execution prices in the highlighted stretch. You can also
expand the list of pauses and select a step range to zoom in. The overlay is
off by default. These highlights are derived from execution prices within the
model's intervals. The saved predictions still contain the three directions
and their original boundaries. A very slow trend can also produce a highlight.
See [how well the overlay distinguishes pauses from slowdowns](trend-pause-detection.md).

A direction applies from the start step up to, but excluding, the stop step.
`no_trades` marks a period without executions and `insufficient_data` marks a period with
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

Sign in with an administrator token and open **Training** at `/training`. The
service needs a running `admin` worker. For your first run, choose **New run →
Prepare training data**, set the experiment counts and length, then select
**Generate datasets**. The worker generates two independent datasets and
creates automatic labels. The form shows how many experiments enter training,
validation and the sealed final test partition.

When the labels are ready, review the service-wide inventory and the listed
source namespaces. If known prior data use is missing, register it before
confirming. Choose **Reserve benchmark**, then **Use prepared data** to fill
the training form. Training starts when you choose **Start training**.
To resume after leaving or refreshing the page, reopen **New run** in the
same browser and account and choose **Continue preparation**. Queued jobs
continue on the server while the page is closed.

To use imported snapshots or externally prepared data, use the API.
[Data preparation](api.md#1-create-observations-and-references) ·
[Benchmark reservation](api.md#2-reserve-final-data-before-selection)

1. Choose **New run**, or open `/training/new` directly. You can name the run
   after its data or the change you are testing, for example `EUR/USD quiet sessions`.
   Select a development dataset. The form shows its compatible training labels, validation
   labels and reserved benchmarks.
   Hover over a **?** or tab to it for a field explanation. Click or tap the
   **?** to keep the explanation open. Press Escape or tap outside to dismiss it.
   If data is missing, use the setup links beside the fields.
2. Choose an architecture. The form fills in its defaults. Choose **Start
   training** to use them, or open **Training policy** to adjust the run.
   **Advanced settings** holds the architecture parameters and CPU runtime
   overrides.

   The **Logistic baseline** uses one seed and compares regularization settings.
   Choose **Generate seed** for a fresh value. For the **Temporal convolutional
   network (TCN)**, set **Training repeats** to the number of models to train.
   The form fills in one seed per repeat. Choose **Generate seeds** for a fresh
   set, or paste the seeds from an earlier run to reuse them. You'll find the
   saved list under `policy.seeds` in **Run details and settings**.

   **Model size** shows each TCN preset's channel count and number of trainable
   parameters. For a custom layout, edit **Model configuration (JSON)** under
   **Advanced settings**. The run history also shows the parameter count beside
   the architecture. [TCN architecture and measurements](tcn.md)

   Under **Training budget**, choose **Epochs** to limit passes over the
   training data, or **Optimizer updates** to give every seed a fixed number of
   weight updates. For a fixed budget, enter **Updates per seed**. Leave
   **Validation checkpoints** blank to spread up to ten checks across each repeat,
   or enter the update numbers to check, ending with the final update.

   Choose **Edit JSON** to edit or paste a complete run configuration. You can
   also paste just a training policy to use it with the data and architecture
   you selected. Choose **Start training** in the editor to submit the JSON, or
   **Use form** to continue editing the same configuration in the form.

   You can bookmark the run or leave and return while the worker trains it.
   If you leave a form before submitting, browser Back restores the draft in the same tab.
3. Select a run from the list, or open **Run history** on a phone. Unnamed
   runs show the architecture and dataset. To preview the model size and full
   dataset ID, hover over the run, focus it with Tab or tap its info button.
   Choose **Rename** to change the
   name, or clear the field and save to restore the automatic title. You can
   cancel a run while it is queued or running.
   The page checks active jobs about every 2.5 seconds. Charts appear as the
   worker records measurements. You can reopen the run to catch up or expand
   **Values** under a chart to read individual measurements. If the worker
   retries training, the charts start again with the new attempt.

   **Elapsed** runs from submission until the job finishes. **In queue** includes
   waits between attempts. **With worker** covers the time each attempt was
   assigned, including data loading, fitting and saving. If a worker stops
   responding, its time ends when the service recovers the job. Older runs
   show the breakdown only when their timing was recorded. For newly completed
   models, **Run details and settings** also shows how long the successful fit took and how many
   training and validation rows it received. Each row is one instrument at
   one time step on the original grid.

   Hover or tap inside a chart to read the nearest measurement. With the chart
   cursor focused, use the arrow keys to move between measurements. Each curve
   marks its most recent measurement. The validation curve also marks the
   first measurement to reach its highest score.

   The baseline chart shows a validation score for each regularization setting.
   The TCN records training loss at every epoch or optimizer update. With an
   epoch budget, it scores validation interval macro-F1 at epoch 1, at the
   chosen interval and at the final scheduled epoch if reached. **Checks
   without improvement** sets how many consecutive validation scores can tie
   or fall below the repeat's best score before it stops. With an update
   budget, validation scores appear at the chosen checkpoints. The charts show
   when each repeat ended and why.

   For a TCN with several repeats, the charts follow the repeat being trained.
   Choose a **Training repeat** to keep its curves on screen. Each seed sets
   the starting weights and training order. The TCN keeps the checkpoint with
   the highest validation interval macro-F1. If scores tie, it keeps the first
   checkpoint to reach that score. The repeat that produced it is marked
   **used for this model**. Its score and seed appear above the charts.
4. Under **Model evaluation**, choose **Evaluate model** to score the saved
   weights on the validation references. When the evaluation finishes, its
   interval and step scores and results by direction appear on the run page.
   Opening the run again shows the saved result. **All fields** shows the
   full report in the browser, and **Download JSON** saves a copy. If the
   evaluation fails or is cancelled, choose **Retry evaluation**.
   To make this model the default for new analyses, choose **Use as default**
   and confirm the change.

To put a finished run away, choose **Archive run**. To move several at once,
choose **Select runs** in the history, check the runs and choose **Archive
selected**. Switch from **Current runs** to **Archived runs** to find them.
The history filter keeps your open run or draft in place.
Open an archived run and choose **Restore run** to return it to the current
list. The same link opens its charts in either view. Archiving preserves any
saved weights and keeps published models available for analyses.

Choose **New run from this run** to open an editable copy with the same data,
architecture, seeds and resource settings. Select replacements for any
references or reservations that are no longer available. **Start training**
submits the new run. The JSON under **Run details and settings** can also be
copied into the new-run editor.

To compare two runs, choose **Select runs**, check both and choose
**Compare**. The page shows times, parameter counts and settings that differ.
For two successful runs with the same dataset, validation references and
analysis profile, it also shows the validation interval macro-F1 difference.
The comparison URL opens the same pair again.

To free space, expand **Run details and settings** and choose **Delete model
weights**. Review the file size and optionally check **Also archive this run**
before confirming. Deleting weights permanently retires the model version.
The charts, settings and completed evaluations stay with the run.

You can train and compare several candidates under one reservation. The
training screen keeps the final test references sealed. Once you have chosen the
model for the final evaluation, freeze it and open the benchmark through the
API. [Final evaluation](api.md#4-freeze-and-evaluate-final-data)

### Why these defaults?

The built-in training policy gives each run a repeatable starting point on
CPU. The exact values are configuration choices. Published scores describe the
data, settings and budget used for each evaluated model. [Policy definitions](../src/regimex/model_registry.py) · [Evaluation settings and results](quality.md)

| TCN setting | Default | What the choice does |
| --- | --- | --- |
| Random seeds | 11, 23, 47 | Trains three times with different initial weights and training orders. Reusing these integers makes comparisons repeatable. |
| Maximum epochs | 30 | Caps passes over the training data for each seed. Early stopping can save unused passes. |
| Epochs between validation checks | 2 | Scores at epoch 1, every even epoch and the final epoch if reached. Scoring less often leaves more time for weight updates. |
| Checks without improvement | 5 | Allows four consecutive checks without a new best score. The fifth ends the repeat. For example, if epoch 4 gives the best score, five ties at epochs 6 through 14 stop it at epoch 14. |
| Batch size | 16 | Processes up to 16 instrument sequences together. Decrease the batch size if long sequences exceed available memory. |
| Learning rate and weight decay | 0.001 and 0.0001 | Sets the AdamW step size and applies a small shrinkage to the weights at each update. Compare changes on the same validation set. |

The logistic baseline uses a deterministic solver with one recorded seed, 11.
It compares inverse regularization strengths of 0.1, 1 and 10 on the same
validation data and saves the fit with the highest interval macro-F1.

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
| Submission interrupted | Choose **Retry request** to confirm or resume the original submission. |
| Analysis failed | Read the reason beside the status. **Details** includes the failure code and analysis ID. After resolving the cause, use **Run again** to create a new analysis. |
| Storage full | Check the reserved space and **Pending deletions**. Remove unused experiments or ask the administrator to raise the quota. |
