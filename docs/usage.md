# Using Regimex

With a running service, all you need is your access token and the JSON export
of a completed experiment. The default local address is `http://127.0.0.1:8000/`.
To start a service, use the [Linux installer](installation.md), [Docker
Compose](deployment.md) or the [manual setup](#start-locally) below.

If you have the reference TCN asset pack, open its `examples/index.html` in
your browser to see the saved reports.

## Analyze an experiment

1. Paste your token into **Access token** and select **Sign in**. With the
   default local installation, use the contents of `var/credentials/user.token`.
2. Under **New analysis**, select **Choose JSON file** or drop an export there.
   Try `examples/1-instrument/snapshot.json` from the reference TCN pack.
3. Check the run and instruments, select a published model, then choose
   **Analyze**. The analysis keeps that model version.
4. When the job finishes, read the preview or select **Download HTML** for
   an offline copy. **Download JSON** saves the predictions and statistics.

You can sign out and return to **Experiments** later. A session lasts 30 minutes
by default. Select an experiment to open its latest matching analysis, or
expand the analysis count to pick a particular run. Search by name, instrument,
run, model or analysis ID. **Rename** changes the display name in your workspace.

## Read a report

In the downloaded HTML, select an instrument to see its prices, class
probabilities and predicted intervals on one time axis. Drag to zoom or use
**Reset zoom** to return to the full run. **Download JSON** in this HTML file
saves the original snapshot under `export` and the predictions under `result`.
In a multi-experiment CLI report,
select an experiment, then an instrument. Select an interval in the table to
zoom in on it.

A direction applies from the start step up to, but excluding, the stop step.
`no_trades` marks a trading pause and `insufficient_data` marks a period with
too little information to assign a direction. Return standard deviation
measures the spread of adjacent log price changes within an interval. The
missing-data plot shows fields with gaps. For quotes, an empty book side means
there was no resting order on that side. See [interval definitions](models.md).

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

From a source checkout, these commands validate an export and write the JSON schemas.

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
| Analysis stays queued | Check the worker status beside the job. For a manual installation, run `uv run --locked --all-extras regimex service doctor` with the worker environment loaded. For the local installer, use `var/service.sh doctor`. |
| Request interrupted | **Retry request** resumes the original submission or status check. |
| Analysis failed | Read the reason beside the status. **Details** includes the failure code and analysis ID. After resolving the cause, use **Run again** to create a new analysis. |
| Storage full | Check the reserved space and **Pending deletions**. Remove unused experiments or ask the administrator to raise the quota. |
