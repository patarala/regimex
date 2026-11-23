# Architecture

Regimex accepts a snapshot of a completed market experiment. The API serves the
browser and accepts the same exports from a simulator backend. PostgreSQL stores
ownership, job state and analysis results. Immutable object storage holds the
exports, model files and HTML reports.

## From upload to report

```mermaid
sequenceDiagram
    actor User as Browser or CLI
    participant API as Regimex API
    participant DB as PostgreSQL
    participant Store as Local files or S3
    participant Worker as Analysis worker
    participant Child as Computation process
    User->>API: Upload completed export
    API->>Store: Write verified snapshot
    API->>DB: Register snapshot for this owner
    API-->>User: Snapshot ID
    User->>API: Submit snapshot ID and model
    API->>DB: Pin model version and enqueue analysis in one transaction
    API-->>User: Analysis ID
    Worker->>DB: Claim attempt
    Worker->>Store: Load snapshot and model bundle
    Worker->>Child: Predict, decode intervals and render HTML
    Child-->>Worker: Result JSON and report files
    Worker->>Store: Write verified HTML
    Worker->>DB: Commit result JSON and report reference under the attempt token
    User->>API: Check status and download the result or report
```

The browser and CLI upload the snapshot and request an analysis in two calls.
A simulator backend can do both in one request to
`POST /api/v1/integration/analysis-intake`. The service saves the snapshot
first, so it can be reused if analysis submission fails. Both paths enqueue
the same worker task. [Simulator request sequence](integration.md#request-sequence)

The worker renews its claim while downloading inputs, computing and saving the
report. A retry gets a new attempt token. The database checks the token and
cancellation state in the transaction that makes a result visible. If
cancellation wins, the job ends as cancelled and the report stays unpublished.

## Where to find the code

Paths below are relative to `src/regimex/`.

| Area | Code | Responsibility |
| --- | --- | --- |
| Browser | `web/` | Analysis, model training and integration connection management |
| HTTP | `api.py`, `auth_api.py`, `admin_api.py`, `benchmark_api.py`, `integration_api.py`, `deletion_api.py` | Authenticate requests, validate inputs and call application workflows |
| Workflows | `application/` | Check access, manage snapshots and jobs, train and publish model versions |
| Persistence | `infrastructure/` | Database transactions, migrations, queue and object storage |
| Workers | `service_worker.py`, `computation_worker.py` | Claim tasks, supervise child processes and recover unfinished work |
| Scientific code | `model_registry.py`, `backends/`, `directions.py`, `segmentation.py`, `evaluation.py` | Restore models, prepare features, decode predictions and measure quality |
| Inputs and artifacts | `exports.py`, `data_catalog.py`, `model_bundles.py` | Validate exports, dataset membership and saved model bundles |
| Generator | `market.py`, `market_sampling.py`, `market_export.py` | Simulate a sampled market and build the same export format for the CLI and generated datasets |
| Service reports | `application/report.py`, `application/analysis_comparison.py`, `templates/` | Render a single analysis or compare two completed analyses of the same saved snapshot |
| Report directories | `visualization.py`, `comparison.py` | Render saved predictions across experiments or compare model evaluations |

## Why the components are separate

**Queues and storage.** Procrastinate uses PostgreSQL, so accepting a job and
enqueuing it succeeds or rolls back in one transaction. Analysis, administrative
work and maintenance have separate queues. An administrator can train a model
without occupying an analysis worker. Files can be written before a transaction
commits. Downloads therefore check the database record for ownership and
publication. On deletion, the database removes access first. Storage usage
drops after cleanup succeeds.

**Models and workers.** Each backend receives the original observations, timing
and units. It restores its own weights and preprocessing. The analysis worker's
child process turns the backend's per-step probabilities into intervals using
the decoder settings from that model's bundle. The analysis result records the
settings and the pinned model version. Changing the active model affects future
submissions that select it. To publish a trained candidate, the administrator
must supply a completed validation evaluation for the same weights and
inference policy.

During training, the child writes measured scores and losses to a file in its
attempt directory. The worker sends new rows to PostgreSQL in batches, checking
the attempt token before each write. The browser fetches rows after its last
cursor. Retrying a fit starts a new curve. On success, the worker commits the
metrics from the verified bundle in the same transaction as the model and the
completed run. The browser reads this result from PostgreSQL, so charts remain
available if the owner later deletes the candidate's weights.

**Simulator integration.** The simulator backend checks access to the run,
assembles its observations and holds the integration credential. Regimex checks
the delegated identity and owns the analysis, model and report. A client that
displays the chart alongside a saved result must match both to the same
snapshot. This lets each service keep its own database and release cycle.
See the [integration contract](integration.md).

**Offline reports.** The browser workspace reads jobs and history from the API.
A downloaded HTML report carries the observations, predictions and plotting
code it needs. The in-app preview can reuse a cached copy of Plotly when its
bytes match the version embedded in the report. The CLI can also build a report
directory from matching exports and saved results. See [using reports](usage.md#read-a-report).

## Scaling Regimex

The API accepts and serves work. Analysis workers claim jobs from the same
PostgreSQL queue, and each runs one bounded computation at a time. Training and
maintenance have their own workers. These process boundaries let an operator
add analysis capacity independently and keep model imports out of the web
process. PostgreSQL also commits the intake record and queue delivery together.
A separate message broker would require another mechanism for that guarantee.

### Add capacity on one host

For an existing installation, set the desired worker count and resources in the
project's `.env`. Run Compose from the deployment directory with the same project
name and files used at startup. For [S3 storage](deployment.md#s3-profile),
keep `COMPOSE_FILE=compose.yaml:compose.s3.yaml` in the environment.

```sh
docker compose --profile service up -d --wait --no-deps analysis-worker
```

Each replica has a separate work directory, model cache and health identity.
All replicas share the job queue and the configured object store.

| Setting | Default | Effect |
| --- | --- | --- |
| `REGIMEX_ANALYSIS_REPLICAS` | 1 | Number of analysis worker containers |
| `REGIMEX_ANALYSIS_WORKER_CPUS` | 2 | CPU quota per replica, fractional values allowed |
| `REGIMEX_ANALYSIS_CPUSET` | All CPUs available to the container | Optional CPU IDs, for example `2-3`, for the workers and their computation children |
| `REGIMEX_ANALYSIS_WORKER_MEMORY` | 8g | Container memory limit per replica, including its cache, scratch and child |

When running from source, additional worker processes need distinct
`REGIMEX_WORK_ROOT` values. Use the same database, object store and storage
limits. Stop all replicas before [offline maintenance](storage.md#stop-writers-before-maintenance).

Compare completed analyses per minute, queue delay, worker occupancy and peak
memory while varying replica count. For a comparison at constant resources,
keep the combined worker CPU and memory budgets fixed. For an expansion, leave
headroom for PostgreSQL, the API and administrative work. The model's thread
count is set on registration. More threads and more workers can compete for the
same CPUs.

If upload or polling latency rises while workers have spare capacity, measure
the API and database separately. Each API process admits up to
`REGIMEX_HTTP_CONCURRENCY_LIMIT` active requests. A request holds its slot
until the response finishes. Static assets and the liveness probe bypass
this limit. A `503 capacity_limit` with `Retry-After` means all slots are busy.
Before raising the limit, check memory use and database connections under load.
If Uvicorn logs `Exceeded concurrency limit`, it has reached the separate
`REGIMEX_HTTP_CONNECTION_LIMIT`, which also counts idle keep-alive sockets.
When a submission returns `429 capacity_limit` because the queue
quota is full, wait for an outstanding job to finish. Both analysis and admin
intake count the owner's queued, running and cancellation-requested operations.
By default, analysis intake refuses new work at eight outstanding operations
and admin intake at four. An integration also has a limit of 32 outstanding
analyses across its users. See [request and queue limits](access.md#request-and-queue-limits).

Analysis jobs share a priority. A free worker claims the ready, unlocked
delivery with the lowest job ID. An owner with room in their quota can still
wait behind another owner's backlog.
Two workers can serve the same owner when that owner is the only one waiting.

The load stand uses one owner. To compare scheduling policies, fill one
owner's queue with long analyses and submit a short analysis from a second
owner. Count the waiting jobs that start before the short analysis, then swap
the owners. Also test an older queued job while other owners submit one job
at a time. A priority fixed from the owner's backlog at submission lets each
new arrival overtake that older job. Measure both owners' queue delays and
total completed analyses before changing the scheduling policy.

The supplied Compose API publishes one host port. Multiple API replicas would
need a shared HTTPS entry point that preserves the configured origin and
authentication checks.

### Accelerators and multiple hosts

All three packaged backends run on the CPU. The adaptive forest uses NumPy
for inference, the baseline uses CPU numerical libraries and the TCN uses the
CPU build of PyTorch. Device selection forms part of the immutable inference policy.
Operators can pin the device, thread count and batch limit when
[registering a model](models.md#saved-models). The packaged workers validate
CPU execution and reject a CUDA request.
Before adding a GPU worker, measure the part of worker time spent in model
prediction. Model restoration, report generation and storage also contribute
to latency.

A GPU deployment would need a separately tested worker image, a device-specific
queue chosen with the model at intake, and capability checks on the workers
that consume it. Keep the API machine independent of accelerator availability.
Bind the selected runtime to the accepted job, enforce device memory limits
and verify probabilities and decoded intervals against the CPU reference on
real hardware. A dedicated inference service becomes worth evaluating when
keeping models resident or batching requests across users saves enough time
to cover the extra transfer and scheduling cost.

The current storage contract limits writers to a single host. The local
backend uses filesystem locks, and S3 writers also share a local coordination
directory for capacity checks and deletion. To put writers on multiple hosts,
first replace this coordination with shared, fenced reservations and test
worker loss during writes, cleanup and restore. See [storage guarantees](storage.md).
