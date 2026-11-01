# API workflows

Start the service as described in [usage](usage.md). Open `/docs`, select
**Authorize**, enter your provisioned token, then use **Try it out** on a route.
The generated OpenAPI schema is the complete field reference. All resource paths
below have the `/api/v1` prefix unless shown otherwise.

## Authentication and request rules

Resource routes accept a bearer token or an HttpOnly browser session. The browser
exchanges a token at `POST /session`; `GET /session` restores session state and
`DELETE /session` signs out. Cookie-authenticated mutations require
`X-CSRF-Token`, the configured Origin and an allowed fetch-site context. Do not
mix bearer, cookie and body-login credentials.

Only token/session hashes are persisted. Sessions expire absolutely after 30
minutes by default, and token revocation invalidates their next request. Integration
tokens cannot create browser sessions. Credentials are never saved in browser
storage or URLs. Roles are `user`, `admin` and source-bound `integration`.

Queued submissions and synchronous administrative imports require an
`Idempotency-Key`. Its scope is the authenticated owner and operation. A retry
with identical canonical content returns the original resource; changed content
returns `409`. Every resource and download is authorized on the server.

## Analyze an export

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
| `GET /analyses` | Owned history, newest first, with `limit`/`offset` |

Example analysis body, using the returned snapshot ID:

```json
{"snapshot_id":"SNAPSHOT_UUID","model_version_id":"active"}
```

Exports use schema 2.1 and `Content-Type: application/json`. The limits are
16 MiB, 4,096 original time bins and two instruments. Compressed request bodies,
unknown versions, invalid grids and inconsistent values are rejected; inputs are
never truncated or resampled. `active` is pinned to an immutable version/policy
at intake. Retrying the same request preserves that choice after a model change.

A successful job has both a complete JSON result and HTML report. Partial
uploads and cancelled/stale attempts cannot publish either. Invalid stored
contracts return `409` after authorization; reads do not silently regenerate them.
A missing model or failed analysis never triggers training or publication.

`GET /models`, `/models/{id}` and `/metrics` expose allowed model metadata and
the evaluation attached to the active publication. `/architectures` is an
administrator catalog of installed backend schemas, not a worker capability test.

## Administration

Use an administrator token in Swagger. Keep the `admin` worker running separately
from the `analysis` worker. Poll `/admin-runs/{id}` or the returned job link for
queued operations. Generation, reference creation, training, evaluation and
publication are separate actions.

For each POST, select **Try it out**, fill its body and required headers, then
**Execute**. Give each new logical operation a new descriptive idempotency key;
reuse it only when retrying that operation. Copy resource IDs from completed
responses into the next request rather than inventing them.

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

This is a usage example, not a quality benchmark. Fix generation settings before
observing labels or scores. The split keeps whole experiments and seed groups
together. `GET /datasets/{id}/manifest` downloads ordered membership and identity.

Create separate training and validation references with `POST /reference-sets`:

```json
{"mode":"automatic","dataset_id":"DATASET_ID","partition":"train"}
```

Repeat with `partition: "validation"`. The default profile is
`direction-32/2.0`; automatic labels use `retrospective-directions/3.0`.
The same profile name cannot describe changed settings. Human-reference import
is also supported with complete annotations and reviewer/independence attestations;
the server checks the contract, not whether a reviewer was genuinely independent.
Ordinary annotation downloads expose training/validation references only.

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

The candidate final dataset is checked against known content, seed groups, runs
and reservations. Renew the inventory attestation when history changes. Human
final annotations, if used, enter through `POST /benchmarks/reference-sets` before
reservation. The service enforces reference access, but cannot prove that an
operator never inspected external files or discover unknown data exposures.

### 3. Train and publish

1. Read `/architectures` for `baseline` or `tcn` parameters. Submit
   `POST /training-runs` with the development dataset, training-label and
   validation-reference IDs, selection benchmark, analysis profile, parameters,
   training policy and separate training/inference runtimes.
2. Training creates an immutable candidate and verifies exact prediction reload.
   It does not change the active deployment. Obtain its version from the completed run.
3. Submit `POST /evaluations` for that concrete model and its development
   validation references. This performs fixed-model evaluation without fitting.
   Download `GET /evaluations/{id}/result` to inspect detailed metrics.
4. Submit `POST /models/{id}/publish` with the successful matching evaluation and
   `expected_previous`. Publication checks the bundle again and changes the
   active pointer atomically. An outdated expectation conflicts.

The resulting quality status remains `validation_only`. Publication does not
establish independent predictive accuracy. Managed training candidates require
this evaluation/publication workflow; CLI bootstrap activation cannot bypass it.

### 4. Freeze and evaluate final data

After selection, call `POST /benchmarks/{id}/freeze` with one concrete model
version. Pending training blocks this action; further training under the frozen
reservation is rejected. Then call `/benchmarks/{id}/open` and
`/benchmarks/{id}/evaluations`. The worker evaluates only the exact frozen bundle,
runtime, decoder and bound test references. Access and identity are checked before
reading payloads and again at commit. Further selection requires fresh final data.

### Compare evaluations

Save two compatible evaluation JSON files, then render without running a model:

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
bootstrap samples for both models. Incompatible reference grids, populations,
datasets, policies or bootstrap settings are rejected. See [metrics](quality.md).

## Health and error handling

`/health/live` checks the API process. `/health/ready` checks database schema and
restore readiness. `/health/models` reports active deployment configuration; it
does not execute a backend. Workers have a separate heartbeat health command.

Treat `4xx` responses as input, authentication, ownership or state failures.
Conflict responses include changed idempotent content and outdated publication
expectations. Capacity failures are explicit; unsupported settings do not fall
back to another model or device. Retry infrastructure errors using the same
operation key. Integration uses the separate [delegation contract](integration.md).
