# Measuring the analysis service under load

To measure a running Compose or source deployment, [point k6 at a dedicated test instance](#running-against-an-existing-test-installation). To compare worker counts at a fixed resource budget, run the disposable stand from a source checkout. It starts its own PostgreSQL cluster, API and workers, registers the model, and records completed analyses and queue timings.

## Prepare a resource-limited stand

Use a Linux account with a delegated cgroup v2 subtree. Set
`REGIMEX_LOAD_CGROUP_PARENT` to its absolute path under `/sys/fs/cgroup`,
or pass `--cgroup-parent`. The account must be able to create children and
move a child process from the launching shell's cgroup into them. The parent
must already have `cpu` and `memory` enabled in `cgroup.subtree_control`.
Each child needs `memory.swap.max`, `memory.peak` and `cgroup.kill`. The
driver configures only groups it creates. [Linux cgroup delegation](https://docs.kernel.org/admin-guide/cgroup-v2.html#delegation)

Install the project's pinned Python and uv versions, Grafana k6 and PostgreSQL
16. The stand finds `initdb` and `postgres` on `PATH` or through
`pg_config --bindir`. Set `REGIMEX_PG_BINDIR` to select a different binary
directory. Supply a saved model bundle, or use `release/models/tcn` from an
unpacked asset pack.

```sh
sh scripts/load_test.sh --stand --cgroup-parent "$REGIMEX_LOAD_CGROUP_PARENT" --bundle release/models/tcn
```

## Workloads

The default stand runs the matrix and sustained profiles. Select `--profile matrix` to compare worker counts, `--profile sustained` for fixed arrivals, or `--profile journeys` for [two-owner workloads](#two-owner-workloads). Use `--profile all` to run all three.

| Profile | Workload | Workers |
|---|---|---:|
| Matrix | Four simultaneous submissions, one per market. Repeat with 512 and 4,096 steps | 1 and 2 at each size |
| Sustained | New 512-step submissions at 6/min for two minutes, then allow outstanding jobs to finish | 2 |

Matrix and sustained runs reuse four two-instrument exports (seeds 42–45) and an
immutable model version. The timer starts after snapshot upload and one warm-up.
With two workers, only one worker is guaranteed to have seen the model. Supply
`--runtime-file PATH` to compare different thread counts or batch sizes.

k6 checks the launch source, model digest, inference policy, snapshot, source
revision, instruments, units and time grid. For every step it checks normalized
class probabilities. Decoded intervals must cover the input and represent each
unavailable interval with a null direction.

The whole stand, including PostgreSQL and k6, shares two CPU cores, 3 GiB of
memory and zero swap by default. Use `--cpus`, `--memory-mib` and
`--http-concurrency-limit` to change the budget. The API admits eight concurrent
requests by default and allows 24 additional connections for idle sockets.

The sustained executor schedules arrivals independently of completions. The default requests 12 analyses. To vary the arrival rate, window and client pool, run e.g.

```sh
sh scripts/load_test.sh --stand --profile sustained --rate-per-minute 12 \
    --duration-seconds 60 --max-vus 16 --bundle release/models/tcn
```

Choose a whole number of arrivals (`rate × seconds / 60`), up to 100. A documented quota refusal counts separately from a verified result. An exhausted k6 client pool, unexpected HTTP response, incomplete drain, result mismatch or missed threshold fails the run. Compare queue samples from the two halves of the arrival window. Finishing all admitted work during the drain can coexist with a growing queue during arrivals.

k6 counts a failed status check or download on its first attempt. The user-facing CLI retries temporary read failures, so it can recover from an interruption that fails the load test.

## Two-owner workloads

For a disposable installation, run the following command. Each scenario gets a fresh
database, two idle user accounts, two analysis workers and the chosen resource budget.

```sh
sh scripts/load_test.sh --stand --profile journeys \
  --cgroup-parent "$REGIMEX_LOAD_CGROUP_PARENT" --bundle release/models/tcn
```

To repeat a single scenario on a fresh stand, use its name as the profile,
for example `--profile uploads`.

| Scenario | Traffic and completion condition |
| --- | --- |
| `contention` | One owner uploads a 4,096-step, two-instrument export and submits four analyses. The second uploads a 256-step, single-instrument export and submits a short analysis. Poll all five, verify downloads and repeat with the owners swapped. |
| `burst` | Both owners submit a simultaneous burst using fresh keys. Count explicit HTTP or queue capacity refusals, verify every accepted result, then submit a fresh job for each owner after the queue drains. |
| `replay` | Repeat an upload, send two concurrent submissions with the same key, replay after completion and verify a single analysis in history. Reuse that key for different input and check the conflict. The second owner's key remains independent. |
| `browse` | Prepare one complete analysis per owner. Two virtual users make 12 visits in total, loading experiment history, analysis details and the HTML report. |
| `cancel` | Cancel a long analysis, check its terminal state and the absence of published output, and complete a short job for the other owner. A passing run must observe a cancellation. |
| `uploads` | Concurrently upload a short and a long export for both owners. Repeat the four uploads, check ownership and stable snapshot IDs, then analyse two of the snapshots. |

The runner generates new seeds for each scenario and records them in `inputs.json`.
A burst defaults to four submissions per owner. The stand accepts
`--burst-per-owner 2` through `12`. The two-owner scenarios use bounded
iterations. Use the sustained profile to find the arrival rate a deployment can
maintain over time.

Against an existing **test** installation, use tokens for two dedicated idle
owners and a published model version. The script checks that the same upload
receives different snapshot IDs for the two accounts. The snapshots and
analyses persist on the installation.

```sh
export REGIMEX_LOAD_TOKEN_FILE=var/load-instance/first-user.token
export REGIMEX_LOAD_SECOND_TOKEN_FILE=var/load-instance/second-user.token
export REGIMEX_LOAD_MODEL=tcn-0.1.0
export REGIMEX_LOAD_PORT=8000
sh scripts/load_test.sh --journeys contention
```

Use `--journeys all` to run the suite, or substitute a scenario name from the
table. Set `REGIMEX_LOAD_BURST_PER_OWNER` to change burst size on an existing
installation. For a multi-node test, direct traffic to the load balancer using
`REGIMEX_LOAD_SCHEME=https`, `REGIMEX_LOAD_HOST`, and `REGIMEX_LOAD_PORT`.
Use its certificate hostname and set `SSL_CERT_FILE` when it uses a private CA.

Each scenario directory contains `journey.json` (outcome counts), `summary.json`
(k6 thresholds and timings), and `points.jsonl` (individual samples).
`journey_upload_ms` times the upload HTTP exchange. `journey_submit_to_result_ms`
ends at the return of the result download. `journey_upload_to_result_ms` includes
the initial upload in a contention round. Check the `owner` and `kind` tags to
compare the delays for each participant. `long_jobs_active_at_short_intake`
counts the submitted long jobs still active when the client queries the queue
right after the short submission. The disposable stand also writes exact median
queue wait by scenario, owner and input kind to `timings.json`. The client-only
runner works without database access.

### Observed results

An isolated two-worker installation running `tcn-0.1.0` accepted 26
analyses across contention, burst, replay, browse and cancel. k6 verified
25 downloads, one cancellation and 12 history visits, with zero unexpected
HTTP responses. The short analysis waited 12.84 s and 14.28 s in the queue
in the two contention rounds. These five scenarios shared one test installation
without a CPU or memory cap. A separate capped contention run during other
CPU-intensive work timed out on a status request.

The uploads scenario ran on a fresh stand limited to two CPU cores and 3 GiB
of memory. All four simultaneous uploads and their retries retained the
correct snapshot identities. Verified downloads took 5.35 s from submission
for the short export and 7.78 s for the long one. Peak memory for the whole
stand was 900 MiB. Use the sustained profile and repeated measurements to
estimate capacity for a deployment.

## Results and cleanup

Open `comparison.md` in the printed output directory. It reports completed
jobs, latency, peak memory and HTTP failures for each cell. The runner exits
nonzero on a failed cell and keeps partial results. Use a new `--output DIR`
to choose where the stand saves its artifacts.

| File | What to inspect |
| --- | --- |
| `run.json`, cell `envelope.json` | Model and runtime digests, workload parameters, versions and actual CPU/memory limits. Matrix and sustained input hashes are in `run.json`. Journey seeds and hashes are in the cell's `inputs.json`. |
| `comparison.json` | Machine-readable outcomes, queue samples and available stage timings. A missing measurement is reported as unavailable. |
| Cell k6 summary and points | Per-request timings and HTTP statuses, including the failing route. Matrix and sustained files use the phase name, e.g. `sustained.json` and `sustained-points.jsonl`. Journey files are `summary.json` and `points.jsonl`. |
| Cell `timings.json` | Queue wait, worker time, model loading, prediction and report generation. Use `jobs.jsonl` for matrix or sustained submissions and `verified.jsonl` for completed journey jobs. [Timing definitions](analysis-timings.md) |

A four-job p95 describes that batch. Use longer fixed-arrival runs to characterize
tail latency. Peak memory includes page cache and setup. The supervisor stops
each cell and waits for the cgroup to empty before removing credentials, database
files and objects. If cleanup cannot verify shutdown, it reports the retained
private directory and cgroup.

## Measure the cost of a fresh inference process

Run the offline comparison on a CPU with the release bundle installed.

```sh
uv run --locked --extra ml python scripts/benchmark_inference_process.py
```

The script generates four 512-step, two-instrument exports with seeds 42–45
and verifies the pinned `tcn-0.1.0` bundle. Each export runs first in its own
process, then all four run sequentially in a new process that loads the model
once. Both modes use one inference thread. The command prints JSON only after
the class probabilities match byte for byte on all four inputs.

Compare `fresh_total_wall_ms` with `reused_process.wall_ms` for the whole
batch, including interpreter startup. Each process also records bundle
restoration, per-export parsing and backend prediction times. Linux also
reports peak resident memory for each process. File reads
can benefit from the operating system cache. To measure queueing, report
rendering, network traffic and publication, use the service workloads above.

In one run, four fresh processes took 10.88 s in total (2.68 s median per
export). One process handled the same four exports in 2.59 s, including
1.34 s for model restoration. Its first prediction took 55 ms and the next
three took 20–22 ms. Peak memory was 271 MiB for the largest fresh process
and 273 MiB for the reused one. All 4,096 rows had identical probability
vectors. This measures four sequential CPU jobs. Per-job cancellation,
memory growth over long runs and serving concurrent users require separate
service-level tests.

## Running against an existing test installation

For a single owner, use a dedicated idle test account with a published model
and an analysis worker. The default command runs one warm-up, four serial
analyses and four parallel analyses. These jobs persist in the installation.
Supply the concrete version from **Training** or the imported bundle manifest.

```sh
export REGIMEX_LOAD_TOKEN_FILE=var/credentials/user.token
REGIMEX_LOAD_MODEL=$(python3 -c 'import json; print(json.load(open("release/models/tcn/manifest.json"))["model_version_id"])')
export REGIMEX_LOAD_MODEL
```

For a test deployment following the [Compose guide](deployment.md#start-the-service), use the public CA certificate that signed the server certificate. This command connects to `https://localhost:8443` and verifies the certificate for `localhost`.

```sh
REGIMEX_LOAD_SCHEME=https \
  SSL_CERT_FILE="$PWD/var/secrets/api-ca.crt" sh scripts/load_test.sh
```

For a deployment started with the [source setup](development.md#start-locally), use its HTTP port 8000.

```sh
export REGIMEX_LOAD_PORT=8000
sh scripts/load_test.sh
REGIMEX_LOAD_RATE_PER_MINUTE=12 REGIMEX_LOAD_DURATION_SECONDS=60 \
    sh scripts/load_test.sh --sustained
```

With `--sustained`, `REGIMEX_LOAD_MAX_VUS` sets the preallocated client pool. Use `REGIMEX_LOAD_MAX_RESULT_MS` to impose a submit-to-download latency threshold.
