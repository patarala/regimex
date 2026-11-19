# Analysis timing report

The load stand can join the analyses accepted in each k6 phase to PostgreSQL
queue events and structured analysis-worker logs. To collect these measurements
independently, use the disposable instance's service environment (including
its database DSN and schema) and run the collector before deleting the instance.

```sh
uv run --locked --extra service python scripts/load_timings.py \
  --jobs var/load-test/jobs.jsonl \
  --worker-log var/load-test/analysis-1.log \
  --worker-log var/load-test/analysis-2.log \
  --output var/load-test/timings.json
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
| `report_ms` | Child monotonic clock around validation and rendering of the self-contained HTML report in memory. |

The three child stages are components of `child_process_ms`, which is itself
part of worker occupancy. Child timings travel in the private subprocess
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
