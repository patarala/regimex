# Analysis timing report

The [disposable load stand](load-testing.md) collects queue and worker timings
before it shuts down each temporary database. Open `timings.json` in the cell
directory under the run's output directory, for example `parallel-512-2w/timings.json`.

To collect the same report on a test installation you manage, keep its
PostgreSQL database available and load its service environment, including the
DSN and schema. Set `TIMING_DIR` to a directory containing your accepted job
IDs and worker logs. This example uses two analysis workers.

```sh
uv run --locked --extra service python scripts/load_timings.py \
  --jobs "$TIMING_DIR/jobs.jsonl" \
  --worker-log "$TIMING_DIR/worker-1.log" \
  --worker-log "$TIMING_DIR/worker-2.log" \
  --output "$TIMING_DIR/timings.json"
```

Supply one raw JSON service log per analysis worker. The jobs file contains
one acceptance per line, for example
`{"phase":"serial","analysis_id":"7acb77f4-6144-4ab1-bd3f-0aab23535fa3"}`.
Every accepted ID must be unique across phases. The k6 summaries give the
client's intake and result latencies and verified job count. The timing
collector only sees IDs listed in the jobs file.

| Metric (ms) | Measurement boundaries |
| --- | --- |
| `queue_wait_ms` | PostgreSQL transaction timestamps of the Procrastinate `deferred` and `started` events, from the intake transaction to the worker fetching the delivery. |
| `worker_occupancy_ms` | PostgreSQL transaction timestamps from `started` through the queue's `succeeded` acknowledgement, including result/report publication and task cleanup. |
| `child_process_ms` | Parent monotonic clock around child supervision, including launch, imports, scratch I/O, checks, reap and manifest validation. |
| `model_load_ms` | Child monotonic clock around bundle verification, backend resolution and restoration of the predictor from the local cache. |
| `prediction_ms` | Child monotonic clock from observation input preparation through prediction, interval decoding, result serialization and digest calculation. |
| `backend_predict_ms` | Child monotonic clock around the loaded backend’s `predict` call. This includes feature preparation performed by the backend and returning the probability array to the caller. |
| `report_ms` | Child monotonic clock around validation and rendering of the self-contained HTML report in memory. |

Model loading, prediction and rendering are components of `child_process_ms`,
which is itself part of worker occupancy. `backend_predict_ms` is a component
of `prediction_ms`. Child timings travel in the private subprocess
manifest and operator logs. PostgreSQL uses its own clock for both queue
boundaries. Its transaction timestamps can include time spent in the intake
transaction before the queued job becomes visible.

`analyses` lists state, application attempt count, delivery count, ordered
queue event types, measurements and any issues for each ID. `phases` gives
the submitted count, complete count and medians for complete records. A
complete record requires one successful application attempt, one successful
queue delivery and one matching child timing event. Retries, unfinished or
missing events and inconsistent durations are flagged. Unavailable values
are JSON `null`. The collector writes the report and exits with status 1
when any submitted analysis has issues.

## Reference workload

These three measurements used one owner, two analysis workers and previously
uploaded 512-step exports. The API, database and workers shared two CPUs and
3 GiB of memory. Each request used `tcn-0.2.0`. The client timed submission
through result download and verified the returned result. The measurements
used different revisions, so repeat the workload before sizing a deployment.

| Scheduled per minute | Arrival window | Verified / scheduled | Refused at owner quota | Median queue wait | p95 submission to download |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 90 s | 18 / 18 | 0 | 0.03 s | 6.56 s |
| 24 | 90 s | 36 / 36 | 0 | 2.16 s | 12.33 s |
| 24 | 240 s | 75 / 96 | 21 | 16.15 s | 34.58 s |

In the four-minute run, the median child spent 2.63 s restoring the model
and 0.20 s predicting. The [process benchmark](load-testing.md) measures the
effect of restoring the same model once for a sequence of inputs.
