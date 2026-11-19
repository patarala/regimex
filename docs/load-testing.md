# Measuring the analysis service under load

Run the disposable stand from a source checkout. It creates its own loopback PostgreSQL cluster and API, local object store, user and admin tokens, registered model version and analysis workers. It waits for model readiness and the requested worker registrations and verifies a warm-up prediction before each measurement.

```sh
sh scripts/load_test.sh --stand --bundle release/models/tcn
```

Use Linux with unified cgroup v2, a running systemd user manager with the CPU and memory controllers delegated, and a kernel exposing `memory.peak`. Run as an ordinary user. Install the Python and uv versions in `.python-version` and `pyproject.toml`, Grafana k6 on PATH, and the PostgreSQL 16 server tools. The stand discovers `initdb` and `postgres` on PATH or via `pg_config --bindir`. Set `REGIMEX_PG_BINDIR` to override this discovery. Unpack a released model bundle or supply the directory of a saved model with `--bundle`. The default path is `release/models/tcn` relative to the checkout. The stand records the manifest SHA-256 once and passes it to every registration. uv installs the locked service, ML and visualization dependencies.

## Workloads

The default command runs both profiles. Use `--profile matrix` for the four capacity comparisons or `--profile sustained` for a single fixed-arrival run.

| Profile | Workload | Workers |
|---|---|---:|
| Matrix | Four simultaneous submissions, one per market. Repeat with 512 and 4,096 steps | 1 and 2 at each size |
| Sustained | New 512-step submissions at 6/min for two minutes, then allow outstanding jobs to finish | 2 |

The generator writes four two-instrument markets per size, using seeds 42 through 45. Every worker configuration at that size reuses the same files byte for byte. All clients request the same pinned version. Snapshot upload and the single warm-up job precede the timed submissions. A single warm-up can populate only one worker’s local model cache in the two-worker cells. The other worker may pay the first-use download and validation cost in the measured batch. Pass `--runtime-file PATH` to compare an alternate inference policy. For example, a JSON file containing `{"threads":1,"max_batch_size":8}` uses a single inference thread with a maximum batch size of eight. The driver validates the file, expands defaults and freezes its contents in `run.json`. It supplies that snapshot to every registration and checks the accepted bundle and policy digests. The registered backends currently support CPU inference.

The runner checks each downloaded result's analysis, snapshot and model identities and each instrument's grid length.

Each cell gets a fresh systemd user service with two logical CPUs shared by all descendants, a CPU quota of 200%, a total memory limit of 3 GiB and zero swap. This includes the API, database, workers, computation subprocesses and k6. Each worker executes one analysis at a time. All cells use the same CPU IDs. The driver verifies the effective affinity and the unit's kernel quota, memory and swap settings. Use `--cpus` and `--memory-mib` to select another common envelope. Host contention and page-cache state can vary between runs.

Use `--http-concurrency-limit` to vary the private API’s Uvicorn connection/task limit independently of worker count. The default is 8 and the minimum is 2. The selected value is recorded in `run.json`.

The sustained executor schedules arrivals independently of completions. The default requests 12 analyses. To vary the arrival rate, window and client pool, run e.g.

```sh
sh scripts/load_test.sh --stand --profile sustained --rate-per-minute 12 \
    --duration-seconds 60 --max-vus 16 --bundle release/models/tcn
```

Choose a whole number of arrivals (`rate × seconds / 60`), up to 100. A documented quota refusal counts separately from a verified result. An exhausted k6 client pool, unexpected HTTP response, incomplete drain, result mismatch or missed threshold fails the run. Compare queue samples from the two halves of the arrival window. Finishing all admitted work during the drain can coexist with a growing queue during arrivals.

## Results and cleanup

The runner prints the artifact directory under `var/load-test/run-*`. Select a new directory with `--output`. The driver returns nonzero on failure. It stops the transient service and its computation descendants on completion or interruption. Each unit has a hard runtime deadline and a ten-second stop timeout. Temporary passwords, tokens, PostgreSQL data and stored objects live in a private temporary directory that the driver removes after confirming the unit stopped. If the user manager cannot confirm shutdown, the driver preserves the directory and reports its path alongside the unit name. Verify that the named unit has stopped before cleaning up its data. The runner also attempts to collect stage timings for accepted jobs after a failed phase. A partial comparison states how many of the planned cells completed. Logs and available evidence remain in the artifact directory.

- `run.json` records the input file hashes, model manifest digest, tool versions, source revision when available, common resource budget and arrival configuration. Each cell's `envelope.json` records its applied kernel settings.
- `comparison.md` and `comparison.json` compare completed jobs per minute, median and p95 submit-to-result latency, and peak memory. Matrix throughput counts verified downloads over the time from first measured submission to last verification. Sustained throughput divides the verified count by the full configured arrival window or the elapsed time up to the last verification or quota refusal, whichever is longer. The JSON records this denominator and the observed span to last verification separately. Both exclude upload and warm-up. The Markdown report also compares median stage durations.
- Peak memory is the kernel's `memory.peak` in the cell's own cgroup from setup through its measured phase. It includes resident memory and charged cache for the whole stand. It is reported in MiB. Compare runs with the same included processes and workload.
- In each cell, `warmup.json` and the measured `parallel.json` or `sustained.json` contain the k6 summaries. The corresponding `*-points.jsonl` holds timestamped samples. `*-console.log` includes the accepted analysis IDs and sustained queue samples. The sustained summary reports scheduled, started, accepted, verified, quota-refused, failed and dropped counts plus active jobs after drain.
- `jobs.jsonl` maps phase names to accepted IDs. `timings.json` joins these IDs to database delivery events and worker stage timings. Inspect per-analysis records alongside phase medians for queue wait, worker occupancy, supervised child time, model loading, prediction and report creation. The backend predict call is part of overall prediction and includes backend input preparation. These intervals overlap. A four-job p95 summarizes this batch and has little resolution for estimating tail latency.

## Running against an existing test installation

Use a dedicated loopback instance with an active model, analysis worker and idle test owner. The default script submits one warm-up, four serial and four parallel 512-step analyses. The sustained option runs the warm-up and fixed-arrival workload. These analyses and their outputs persist in that installation.

```sh
sh scripts/load_test.sh
REGIMEX_LOAD_RATE_PER_MINUTE=12 REGIMEX_LOAD_DURATION_SECONDS=60 \
    sh scripts/load_test.sh --sustained
```

The defaults are port 8001, `var/load-instance/credentials/user.token` and model `tcn-0.2.0`. Set `REGIMEX_LOAD_PORT`, `REGIMEX_LOAD_TOKEN_FILE` or `REGIMEX_LOAD_MODEL` to override them. With `--sustained`, `REGIMEX_LOAD_MAX_VUS` sets the preallocated client pool. Use `REGIMEX_LOAD_MAX_RESULT_MS` to impose a submit-to-download latency threshold on the external run. The disposable stand manages its own configuration, credentials and resource envelope.
