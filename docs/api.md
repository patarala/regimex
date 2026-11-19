# API workflows

Start the service using the [usage guide](usage.md). Open `/docs`, select
**Authorize**, enter your token and use **Try it out** to send a request.
Resource paths in this guide are relative to `/api/v1`. Health checks use
`/health` at the server root.

## Authentication and request rules

Send a bearer token for API calls. The browser exchanges a user or administrator
token at `POST /session` for an HttpOnly session cookie. Use `GET /session` to
restore a session and `DELETE /session` to sign out. Cookie-authenticated
mutations require the configured Origin, an allowed fetch-site context and the
session's `X-CSRF-Token`. Use one authentication method per request.

Sessions expire 30 minutes after sign-in by default. Revoking the underlying
token also ends access on the next request. Simulator backends use a separate
[integration credential](integration.md).

Send an `Idempotency-Key` for queued submissions and synchronous administrative
imports. Retry with the same body and key to receive the original resource,
including the profile and model selected when the request was accepted. Reusing
that key for different content under the same owner and operation returns `409`.

## Manage platform connections

Administrator routes under `/admin/integrations` manage platform connections:

| Request | Result |
| --- | --- |
| `POST /admin/integrations` | Register an integration account/source |
| `GET /admin/integrations` | Paginated connection list |
| `GET /admin/integrations/{id}` | Connection state and revision |
| `PATCH /admin/integrations/{id}` | Rename, pause or resume with `expected_revision` |
| `POST /admin/integrations/{id}/keys` | Issue a named, scoped key with an expiry; secret returned once |
| `GET /admin/integrations/{id}/keys` | Paginated key metadata, never secrets |
| `POST /admin/integrations/{id}/keys/{token_id}/revoke` | Permanently revoke a key |
| `GET /admin/integrations/{id}/events` | Paginated access-change history |
| `GET /admin/integrations/{id}/storage` | Connection-wide storage usage and limit |

A simulator backend can check its own key with `GET /connection`. Send the
integration bearer token alone. The response contains `source_namespace` and
`scopes`. For key rotation, capabilities and shared limits, see
[platform access](access.md).

## Analyze an export

Use `Content-Type: application/json` for the schema 2.1 export. The service
accepts one or two instruments, up to 4,096 original time bins and up to 16 MiB
of uncompressed JSON. Save the export with `POST /snapshots` and copy its
`snapshot_id` into a `POST /analyses` request.

```json
{"snapshot_id":"SNAPSHOT_UUID","model_version_id":"active"}
```

Poll the returned analysis link. When the job succeeds, download the saved JSON
result or HTML report. To analyze the same experiment again, submit the same
snapshot ID with a new idempotency key. Choose a published model version to
pin it explicitly or use `"active"` for the version active at acceptance.

| Request | Result |
| --- | --- |
| `POST /validate` with export JSON | Validates without storing or running a model |
| `POST /snapshots` with export JSON | Saves an owned immutable snapshot |
| `GET /snapshots/{id}` | Snapshot metadata |
| `GET /snapshots/{id}/export` | Original export |
| `POST /analyses` | Queues an analysis and returns `202` with polling links |
| `GET /analyses/{id}` | Analysis status and available result/report links |
| `GET /jobs/{id}` | Job state and error information |
| `POST /jobs/{id}/cancel` | Cancels queued work or prevents a running commit |
| `GET /analyses/{id}/result` | Saved JSON after success |
| `GET /analyses/{id}/report` | Self-contained HTML attachment |
| `GET /analyses/{id}/comparison?other=ANALYSIS_UUID` | Offline HTML comparison of two successful analyses |
| `GET /analyses` | Owned history, newest first, with `limit`/`offset` |
| `GET /experiments` | Owned history grouped by saved snapshot, with `limit`/`offset` |
| `PATCH /experiments/{snapshot_id}/name` | Rename an experiment with a revision check |
| `GET /service-status` | Current analysis-queue availability for signed-in users |
| `GET /storage` | The authenticated owner's storage usage and limit |

### Storage and history

`GET /storage` returns `used_bytes`, `reserved_bytes`, `limit_bytes` and
`available_bytes`. The last value is the owner's remaining object quota.
Reservations include uploads and files awaiting deletion. Delegated clients use
`GET /integration/storage` with the `read` capability. See [storage
quotas](storage.md#owner-and-connection-quotas) for which files count.

Use `GET /analyses` for individual attempts and `GET /experiments` for
history grouped by saved snapshot. Both accept these filters before pagination.

| Parameter | Meaning |
| --- | --- |
| `q` | A literal, case-insensitive search (up to 128 characters) across experiment names and source, run, snapshot, instrument, model and analysis identifiers |
| `state` | One of `queued`, `running`, `cancel_requested`, `cancelled`, `succeeded`, `failed`, or `active` for the first three |
| `limit`, `offset` | Page size and starting position |

```text
GET /api/v1/analyses?q=BTC%2FUSD&state=succeeded&limit=20&offset=0
```

Each experiment shows the total and successful analysis counts, the count
matching the filters, and the latest matching analysis. To list its attempts,
call `GET /analyses?snapshot_id=...` with the same filters. Snapshots that have
never been analyzed appear with zero counts and `latest_analysis: null` unless
a state filter is set. A failed job reports `failure.code`, `failure.message`
and attempt counts. Once it reaches `failed`, automatic retries have ended.

### Choose a model or compare results

`GET /models?published_only=true` lists published versions and the active
model ID. Administrators can omit the filter to see unpublished candidates.
`GET /models/{id}` and `GET /metrics` expose model metadata and evaluation
information. Use `GET /architectures` as an administrator to inspect installed
backend schemas.

Compare two successful analyses of the same saved snapshot with
`GET /analyses/{id}/comparison?other=ANALYSIS_UUID`. The HTML attachment
aligns their predictions on the original price grid. A comparison of different
snapshots returns `409`, even if the observations match. Prediction disagreements
show where the two models classify the same data differently.

## Delete saved resources

Only the owner can delete a resource. An integration key also needs the
`delete` capability and access to the source and run.

| Request | Result |
| --- | --- |
| `POST /analyses/{id}/deletion-preview` | Preview removing one terminal analysis and its saved report |
| `POST /experiments/{snapshot_id}/deletion-preview` | Preview removing the snapshot and all its analyses |
| `POST /deletions/{plan_id}` with `{"sha256":"PREVIEW_SHA256"}` | Confirm the reviewed selection, or resume its cleanup |
| `GET /deletions/{plan_id}` | Read the receipt of an accepted deletion |
| `GET /deletions?limit=20&offset=0` | List accepted deletions with pending cleanup |

Previews include counts, object bytes, blockers, a plan ID and confirmation digest.
They expire after 15 minutes. Confirmation checks current references and the
reviewed resource set again; changed or blocked selections return `409`. Running,
queued and cancelling analyses must finish first. Experiments registered in the
model data catalog cannot be deleted through this workflow.

A `200` receipt with `state: completed` confirms cleanup. A `202` receipt with
`state: pending` means the resources have been removed from history but some
physical files still need cleanup. Their bytes remain charged. Receipts distinguish
`reclaimed_bytes`, `pending_bytes` and `pending_objects` and include a safe failure
message when available. Repeat the same plan ID and digest to resume; the pending
list also supplies these values after a client restart. If a confirmation response
is lost, read its receipt before retrying. A `404` receipt means confirmation has
not been recorded; retry the same reviewed plan if it remains valid.

Deleting an analysis keeps its input snapshot. Deleting the experiment retires
its source/run/snapshot identity, so uploading that identity again returns
`409`.

## Experiment names

Send `PATCH /experiments/{snapshot_id}/name` with the current revision.

```json
{"display_name":"Experiment A","expected_revision":0}
```

Names are limited to 120 Unicode characters after normalization. Empty strings
and control characters are rejected. Use `null` to restore the default name.
The response includes `snapshot_id`, `display_name` and the new `revision`. On
`409`, refresh the experiment before editing again. Renaming affects display
metadata only.

## Administration

Sign in to `/docs` with an administrator token and run the `admin` worker.
Poll `/admin-runs/{id}` or the returned job link for each queued operation.
Use the resource IDs from the completed runs in the following requests.

### 1. Create observations and references

`POST /datasets` either generates observations or combines owned snapshots with
explicit partitions. Generation returns `202`; importing membership returns `201`.
For example:

```json
{
  "mode": "generate",
  "plan": {"seed": 48291, "experiment_count": 60, "step_count": 512},
  "splits": {"seed": 1729, "train_fraction": 0.7, "validation_fraction": 0.15}
}
```

The split keeps whole experiments and seed groups together. Download the
ordered membership with `GET /datasets/{id}/manifest`.

Create separate training and validation references with `POST /reference-sets`:

```json
{"mode":"automatic","dataset_id":"DATASET_ID","partition":"train"}
```

Repeat with `partition: "validation"`. The default profile is
`direction-32/2.1`; automatic labels use `retrospective-directions/3.0`.
To import human references, provide complete annotations and identify the
reviewer. The reviewer attests to independence. The ordinary annotation
download covers the training and validation partitions.

### 2. Reserve final data before selection

Prepare a disjoint final dataset and its sealed test reference set before fitting.
Test references cannot enter ordinary training or validation evaluation.
Automatic test references use `POST /reference-sets` with that dataset ID and
`partition: "test"`; their annotation payloads remain sealed.

| Route | Purpose |
| --- | --- |
| `POST /benchmarks/history/datasets` | Register development exposure for every partition of a dataset |
| `POST /benchmarks/history/import` | Register other known data exposures |
| `POST /benchmarks/attest` | Declare the inventory scope and source namespaces |
| `POST /benchmarks` | Reserve the complete test partition and its references |
| `POST /benchmarks/{id}/selection-start` | Begin model selection under that reservation |
| `GET /benchmarks/registry` | Inspect persisted inventory and reservation metadata |

Regimex checks the proposed final dataset against the content, seed groups,
runs and reservations in the registry. Include known external exposures and
renew the attestation when history changes. If you use human final references,
import them through `POST /benchmarks/reference-sets` before reservation.

### 3. Train and publish

1. Read `/architectures` for `baseline` or `tcn` parameters. Submit
   `POST /training-runs` with the development dataset, training-label and
   validation-reference IDs, selection benchmark, analysis profile, parameters,
   training policy and separate training/inference runtimes.
2. Training requires at least one validation interval with a known direction.
   The worker saves the candidate, reloads it and verifies the predictions.
   Copy its model version ID from the completed run.
3. Submit `POST /evaluations` for that concrete model and its development
   validation references. This performs fixed-model evaluation without fitting.
   Download `GET /evaluations/{id}/result` to inspect detailed metrics.
4. Submit `POST /models/{id}/publish` with the successful matching evaluation and
   `expected_previous`. Publication checks the bundle again and changes the
   active pointer atomically. An outdated expectation conflicts.

Architecture descriptors list `supported_training_budgets`. The TCN accepts
`epochs` and `updates`; baseline accepts `epochs`. Both default to an epoch policy.
For a fixed budget, send the following `policy` object; epoch-only fields such as
`epochs`, `patience` and `validation_interval` are rejected:

```json
{
  "budget": "updates",
  "optimizer": "adamw",
  "seeds": [11],
  "updates": 1200,
  "validation_updates": [300, 600, 900, 1200],
  "batch_size": 16,
  "learning_rate": 0.001,
  "weight_decay": 0.0001
}
```

Publication makes the candidate available for analysis. A candidate
evaluated on the development validation set has quality status
`validation_only`. For held-out metrics, use the final evaluation below.

### 4. Freeze and evaluate final data

After selection, call `POST /benchmarks/{id}/freeze` with one concrete model
version. Pending training blocks this action; further training under the frozen
reservation is rejected. Then call `/benchmarks/{id}/open` and
`/benchmarks/{id}/evaluations`. The worker evaluates only the exact frozen bundle,
runtime, decoder and bound test references. Access and identity are checked before
reading payloads and again at commit. Further selection requires fresh final data.

### Compare evaluations

Download `GET /evaluations/{id}/result` for a baseline and a TCN evaluated on
the same dataset and references. Save them as `baseline-evaluation.json` and
`tcn-evaluation.json` in your working directory, then render the comparison.

```python
import json
from pathlib import Path
from regimex.comparison import render_model_comparison

evaluations = tuple(
    json.loads(Path(p).read_text()) for p in ("baseline-evaluation.json", "tcn-evaluation.json")
)
render_model_comparison(evaluations, Path("var/comparison"))
```

The output directory must be absent or empty. Open `var/comparison/index.html`.
The first model is the reference. Paired differences use the same whole-experiment
bootstrap samples for both models. One experiment can supply a score or paired
difference. Sampling intervals require at least two independent experiments.

Boundary F1 is undefined when the reference and the prediction both have zero
transitions. Evaluation version 2.4 records how many bootstrap draws have
undefined boundary F1. It reports a percentile interval only when every draw
has a defined score. Paired intervals require a defined score for both models
in every draw. The comparison lists experiments with an undefined difference
separately from ties.

Incompatible reference grids, populations, datasets, policies or bootstrap
settings are rejected. See [metrics](quality.md).

## Health and error handling

`/health/live` checks the API process. `/health/ready` checks database schema and
restore readiness. `/health/models` reports active deployment configuration; it
does not execute a backend. Workers have a separate heartbeat health command.

Authenticated `GET /service-status` reports analysis queue availability. The `state`
field is `available` for a free worker slot, `busy` when all slots are occupied,
`unavailable` when there is no fresh worker, and `unknown` when the check
fails. A fresh heartbeat alone cannot tell you whether the worker can load a
particular model. Run `service doctor` to inspect queues, storage and local
model restoration. See the [usage guide](usage.md).

A `409` can mean changed content under an existing key, an outdated revision or
a conflict with the current job state. Read the error code and message before
resubmitting. For an integration rate limit (`429`, `rate_limit`), wait for
`Retry-After`. A storage quota refusal (`429`, `storage_quota_exceeded`) requires
freeing storage or raising the quota. For other capacity limits, read the
reported reason. If intake returns `409 model_unavailable`, an administrator
needs to activate a published model. After a lost response, retry a queued submission with
its original idempotency key.
