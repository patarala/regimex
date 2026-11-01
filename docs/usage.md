# Running and using Regimex

For a quick look, open `release/examples/tcn-1-report.html` or
`release/examples/overview/index.html`. These reports work offline.
The optional release assets contain saved models and examples; they are separate
from the source checkout. Place them under `release/` to use the commands below.

## Start locally

Use Linux, Python 3.14.4, uv 0.13.0 and PostgreSQL 16. PostgreSQL must already be
running with a database and role that you can use. For a local installation, an
administrator can create them with:

```sh
sudo -u postgres createuser --pwprompt regimex
sudo -u postgres createdb --owner=regimex regimex
```

Put the database password in your private PostgreSQL password file (`~/.pgpass`,
mode `0600`), using the line format `host:port:database:user:password`. Commands
below assume `127.0.0.1:5432`, database `regimex` and role `regimex`.

From the project root:

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

HTTP is permitted here only because the address is literal loopback and the
setting explicitly enables it. Use [HTTPS deployment](deployment.md) for a server.

Import the supplied TCN bundle. Compare the files with the release checksums
before importing them; then use the manifest digest:

```sh
MODEL_MANIFEST_SHA256=$(sha256sum release/models/tcn/manifest.json | cut -d ' ' -f 1)
uv run --locked --all-extras regimex service register-model \
  --bundle release/models/tcn --manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --token-file var/credentials/admin.token --activate
```

Without the release models, create and train a candidate through the
[administrative API](api.md#administration). An analysis never trains its own model.

Open four terminals in the project root. Run `. var/environment.sh` in each,
then one command per terminal:

```sh
uv run --locked --all-extras regimex service serve
uv run --locked --all-extras regimex service worker --queue analysis
uv run --locked --all-extras regimex service worker --queue admin
uv run --locked --all-extras regimex service worker --queue maintenance
```

The API serves `http://127.0.0.1:8000/`; API documentation is at `/docs`.
Stop processes with Ctrl+C. Keep `var/objects` and PostgreSQL data: they contain
saved resources. `var/work` is temporary computation and cache space.

## Analyze an experiment

1. Open `http://127.0.0.1:8000/`. Paste the content of
   `var/credentials/user.token` into the sign-in field and select **Sign in**.
   The token is cleared after it is exchanged for a session.
2. Under **New analysis**, select **Choose JSON file**, or drop the file there.
   For a demonstration, use `release/examples/input-1.json` or `input-2.json`.
3. Check the run and instrument summary, then select **Analyze**. The server
   resolves the active published model when it accepts the request.
4. Wait for the result. **Cancel** stops queued work or prevents a running
   attempt from publishing. **Retry request** reuses an interrupted request's
   identity; it does not intentionally create another analysis.
5. Read the preview, or use **Download HTML** and **Download JSON**. Technical
   input/model identity is available under **Details**.
6. Reopen saved work from **History**. **Previous** and **Next** browse pages of
   20 records; **Refresh** returns to the newest page. **Sign out** ends the session.

The default session expires after 30 minutes from sign-in. Refreshing the page
does not extend it. Results survive sign-out and service restarts.

## Read a report

Select an instrument. In a report containing multiple experiments, select the
experiment first. Direction bands show growth, decline or sideways movement;
symbols distinguish them as well as color. `no_trades` and `insufficient_data`
are availability states, not direction classes.

Prices use the instrument's quote units. Return standard deviation describes
the spread of adjacent log returns within an interval. Known zero activity stays
zero; unknown values stay missing. The missing-data panel shows only fields with
actual gaps. Empty bid/ask sides are normal when no order is resting there.

Drag to zoom; the plots stay aligned. **Reset zoom** restores the whole horizon.
Selecting an interval focuses its period. Interval bounds are `[start, stop)`:
the stop step belongs to the following interval. The report contains predictions,
not independently verified labels. See [measured quality](quality.md).

## Terminal tools

Validate an export without a running service:

```sh
uv run --locked regimex validate --input release/examples/input-1.json
uv run --locked regimex schema --output var/schemas
```

Render a multi-experiment offline report from exports and matching saved results:

```sh
uv run --locked --extra viz regimex report \
  --input release/examples/input-1.json --result release/examples/tcn-1-result.json \
  --input release/examples/input-2.json --result release/examples/tcn-2-result.json \
  --output var/report
```

Open `var/report/index.html` and keep the directory together. The output directory
must be absent or empty. Rendering does not run inference or assign labels.
Every result must match its export and use the same model/policy. By default,
at most 12 experiments are displayed, chosen evenly from sorted snapshot IDs;
`--samples` changes that limit up to 100. All inputs are validated first.

## If something fails

| Symptom | Action |
| --- | --- |
| Sign-in fails | Use the user token's full content, the exact configured origin and a current token. A revoked or expired session needs a new sign-in. |
| File is rejected | Check export schema 2.1, a completed run, at most two instruments, 4,096 steps and 16 MiB. Changing only a version field does not convert another format. |
| No active model | An administrator must register/activate a supplied bundle or evaluate/publish a trained candidate. |
| Analysis remains queued | Check the analysis worker is running with the same database and storage configuration. |
| Analysis fails | Read its error, check worker logs and capacity settings. A failure does not produce a partial successful report. |
| History is empty | Check the signed-in owner; other users' analyses are not visible. |

For API automation use [API workflows](api.md). For backups and resource cleanup
use [storage operations](storage.md); deleting files from the object root is not
an application cleanup operation.
