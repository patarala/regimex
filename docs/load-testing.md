# Measuring the analysis service under load

To measure a running Compose or source deployment, [point k6 at a dedicated test instance](#running-against-an-existing-test-installation). To compare worker counts at a fixed resource budget, run the disposable stand from a source checkout. It starts its own PostgreSQL cluster, API and workers, registers the model and verifies a warm-up prediction before measuring the workload.

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

The default command runs both profiles. Use `--profile matrix` for the four capacity comparisons or `--profile sustained` for a single fixed-arrival run.

| Profile | Workload | Workers |
|---|---|---:|
| Matrix | Four simultaneous submissions, one per market. Repeat with 512 and 4,096 steps | 1 and 2 at each size |
| Sustained | New 512-step submissions at 6/min for two minutes, then allow outstanding jobs to finish | 2 |

The generator writes four two-instrument markets per size, using seeds 42 through 45. Every worker configuration at that size reuses the same files byte for byte. All clients request the same pinned version. Snapshot upload and the single warm-up job precede the timed submissions. A single warm-up can populate only one worker’s local model cache in the two-worker cells. The other worker may pay the first-use download and validation cost in the measured batch. Pass `--runtime-file PATH` to compare an alternate inference policy. For example, a JSON file containing `{"threads":1,"max_batch_size":8}` uses a single inference thread with a maximum batch size of eight. The driver validates the file, expands defaults and freezes its contents in `run.json`. It supplies that snapshot to every registration and checks the accepted bundle and policy digests. The registered backends currently support CPU inference.

The runner checks each downloaded result against the submitted export, including the source run and revision, time grid, instrument order, currencies and units. It also checks the analysis ID and the model and policy digests recorded at submission. For each instrument, it verifies normalized probabilities on every step and full interval coverage. Observed intervals must carry a direction class. Intervals marked `no_trades` or `insufficient_data` must carry a null direction.

Each comparison cell runs in its own cgroup. By default it has two logical CPUs, a CPU quota of 200%, a total memory limit of 3 GiB and zero swap. The API, database, workers, computation subprocesses and k6 share those limits. All cells use the same CPU IDs. The driver verifies the effective affinity and kernel settings before starting the workload. Use `--cpus` and `--memory-mib` to select other limits. Host contention and page-cache state can vary between runs.

Use `--http-concurrency-limit` to vary how many active requests the private API can handle per process. The default is 8 and the minimum is 2. The stand records the value in `run.json` and leaves room for 24 additional connections, including idle keep-alive sockets.

The sustained executor schedules arrivals independently of completions. The default requests 12 analyses. To vary the arrival rate, window and client pool, run e.g.

```sh
sh scripts/load_test.sh --stand --profile sustained --rate-per-minute 12 \
    --duration-seconds 60 --max-vus 16 --bundle release/models/tcn
```

Choose a whole number of arrivals (`rate × seconds / 60`), up to 100. A documented quota refusal counts separately from a verified result. An exhausted k6 client pool, unexpected HTTP response, incomplete drain, result mismatch or missed threshold fails the run. Compare queue samples from the two halves of the arrival window. Finishing all admitted work during the drain can coexist with a growing queue during arrivals.

k6 counts a failed status check or download on its first attempt. The user-facing CLI retries temporary read failures, so it can recover from an interruption that fails the load test.

## Results and cleanup

The runner prints the artifact directory under `var/load-test/run-*`. Select a new directory with `--output`. A supervisor outside each cell enforces its deadline and watches for driver exit. During cleanup it stops the cell's descendants and waits for the kernel to confirm that the cgroup is empty before removing private credentials, database files and objects. If shutdown cannot be confirmed, it reports the cgroup and preserves the private directory for inspection.

The driver exits nonzero on failure. A partial comparison shows how many cells reported, including k6 counters, latency samples, HTTP failures and available stage timings for accepted jobs. Missing measurements appear as unavailable.

- `run.json` records the input file hashes, model manifest digest, tool versions, source revision when available, common resource budget and arrival configuration. Each cell's `envelope.json` records its applied kernel settings.
- `comparison.md` and `comparison.json` compare completed jobs per minute, median and p95 submit-to-result latency, and peak memory. Matrix throughput counts verified downloads over the time from first measured submission to last verification. Sustained throughput divides the verified count by the full configured arrival window or the elapsed time up to the last verification or quota refusal, whichever is longer. The JSON records this denominator and the observed span to last verification separately. Both exclude upload and warm-up. The Markdown report also compares median stage durations.
- Peak memory is the kernel's `memory.peak` in the cell's own cgroup from setup through its measured phase. It includes resident memory and charged cache for the whole stand. It is reported in MiB. Compare runs with the same included processes and workload.
- In each cell, `warmup.json` and the measured `parallel.json` or `sustained.json` contain the k6 summaries. The corresponding `*-points.jsonl` holds timestamped samples. `*-console.log` includes the accepted analysis IDs and sustained queue samples. The sustained summary reports scheduled, started, accepted, verified, quota-refused, failed and dropped counts plus active jobs after drain.
- When interruption leaves a truncated summary or timing file, the comparison identifies the unreadable artifact and retains evidence from intact files. HTTP failure samples can appear even when the summary count is unavailable.
- `jobs.jsonl` maps phase names to accepted IDs. `timings.json` joins these IDs to database delivery events and worker stage timings. Inspect per-analysis records alongside phase medians for queue wait, worker occupancy, supervised child time, model loading, prediction and report creation. The backend predict call is part of overall prediction and includes backend input preparation. These intervals overlap. A four-job p95 summarizes this batch and has little resolution for estimating tail latency.

## Running against an existing test installation

Use a dedicated loopback instance with an analysis worker, a published model and an idle test owner. This mode needs k6 and uv and uses the deployment's own resource limits. It submits one warm-up, four serial and four parallel 512-step analyses. The sustained option runs the warm-up and fixed-arrival workload. These analyses and their outputs persist in that installation.

Set the token file and a concrete published model version. The runner requires `REGIMEX_LOAD_MODEL` for this mode and rejects the moving alias `active`. Both setup guides write the user token to `var/credentials/user.token`. If you registered the TCN from a local asset pack, read its version from the same manifest you imported. For a model published through **Training**, use its **Model version** instead.

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

For a deployment started with the [source setup](usage.md#start-locally), use its HTTP port 8000.

```sh
export REGIMEX_LOAD_PORT=8000
sh scripts/load_test.sh
REGIMEX_LOAD_RATE_PER_MINUTE=12 REGIMEX_LOAD_DURATION_SECONDS=60 \
    sh scripts/load_test.sh --sustained
```

With `--sustained`, `REGIMEX_LOAD_MAX_VUS` sets the preallocated client pool. Use `REGIMEX_LOAD_MAX_RESULT_MS` to impose a submit-to-download latency threshold.
