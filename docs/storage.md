# Storage and recovery

PostgreSQL records who owns each resource and which files belong to it. The
object store holds immutable observation files, model bundles and HTML reports.
An upload becomes visible after Regimex checks its contents and registers it in
the database. A usable backup includes both the database and those files.

## Owner and connection quotas

Snapshot files and analysis HTML reports count toward the owner's quota and,
for simulator data, the connection's shared quota. Defaults are 1 GiB per owner
and 2 GiB per connection. Configure `REGIMEX_MAX_OWNER_STORAGE_BYTES` and
`REGIMEX_MAX_INTEGRATION_STORAGE_BYTES` consistently on the API and workers.
The owner limit must not exceed the connection limit. The overall object-store
capacity remains a separate hard limit shared with administrator-managed models,
reference sets and other artifacts.

PostgreSQL reserves an immutable object's bytes before an upload. Datasets owned
by the same user share identical generated snapshots. If a worker retries an
analysis, identical HTML for that analysis also reuses its reservation.
Usage includes both registered objects and reservations, so concurrent requests,
key rotation and interrupted workers do not reset the budget. Failed uploads
remain reserved because storage may have accepted their bytes before the
failure became visible. Offline maintenance can release a reservation only
after proving that its unregistered object is absent.

Open the workspace to see the current owner's usage. Administrators can see a
connection's total in **Connections**. A quota refusal returns `storage_quota_exceeded`.
If a quota is lowered below current usage, existing files can still be read.
Delete saved resources to release their bytes. For a failed upload, use the
offline [orphan cleanup procedure](#orphan-cleanup).

Allow additional disk space for database rows, logs, temporary files and backup
archives.

## Delete saved resources

Delete an analysis to remove its result and report while keeping its input for
another run. Delete the experiment to remove its saved input and analyses.
Only the owner can delete these resources. Simulator backends also need the
`delete` capability for the same subject and run.

1. Preview the deletion and review the resource counts, stored bytes and
   blockers. Queued, running and cancelling analyses must finish first. A
   snapshot in the model data catalog is protected.
2. Confirm within 15 minutes. If analyses or references have changed, request a
   new preview. An owner may have up to 100 open previews.
3. Check the receipt. On confirmation, the resources disappear from history.
   Their bytes remain charged until Regimex verifies removal of the exact files.
   If cleanup was interrupted, open **Pending deletions** and select
   **Retry cleanup**. A replacement integration key can resume cleanup on the
   same connection if it has the `delete` capability.

Deletion is permanent. Replaying the old request cannot recreate an analysis,
and uploading the original identity of a deleted experiment returns a conflict.
Previously downloaded copies stay with their recipients. Regimex retains files
still referenced elsewhere and flags unexpected replacement files for operator
inspection. Complete pending deletions before backing up the installation.

For a training run, an administrator can review and delete the weights of an
unused candidate. The review counts the weight file in object storage. Other
bundle files and completed evaluations stay registered. PostgreSQL stores
the selected seed, score and chart data with each successful run. For an older
run, opening its metrics or previewing weight deletion saves that result from
the verified bundle. [Training workspace](training.md)

## Local storage

Use a dedicated local directory on a filesystem with advisory locks, hard links,
file and directory fsync, and user extended attributes. Keep temporary work files
and the model cache elsewhere. Uploading the same bytes under the same key
reuses the file. A different payload under that key is rejected.

Readers can verify and copy files concurrently. Writers and maintenance take
an exclusive lock. The local store admits up to 100,000 committed files.

Each analysis worker caches model bundles in `$REGIMEX_WORK_ROOT/model-cache`,
keyed by bundle and inference-policy hashes. The Compose workers each have
their own work directory. If a cache is full, stop the workers using that
directory before clearing the cache. They fetch the bundles again on demand.

## S3 storage

Regimex tests against SeaweedFS 4.48 with versioning enabled, conditional
creation and HTTPS. Use a dedicated bucket prefix. Other providers must
support strongly consistent reads and version listings, atomic
`If-None-Match: *` creation and user metadata on immutable versions. Configure
intermediaries to forward each PUT once.

When starting Regimex from a checkout, load the service's database settings
first, then configure the provider. Give the service account write access to
the work and coordination directories.

```sh
export REGIMEX_OBJECT_BACKEND=s3
export REGIMEX_S3_ENDPOINT_URL=https://objects.example.org
export REGIMEX_S3_REGION=us-east-1
export REGIMEX_S3_BUCKET=private-regimex
export REGIMEX_S3_PREFIX=installation-one
export REGIMEX_WORK_ROOT=/var/lib/regimex/work
export REGIMEX_S3_LOCK_ROOT=/var/lib/regimex/s3-coordination
```

Configure credentials through the SDK's environment, profile or role chain.
`REGIMEX_S3_CREDENTIAL_PROFILE` selects a profile and `REGIMEX_S3_CA_BUNDLE`
specifies a private CA. Check the resulting connection with
`uv run --locked --extra service --extra s3 regimex service storage-check`.
The endpoint cannot contain credentials, a path, query or
fragment. HTTP is allowed only for literal loopback with
`REGIMEX_S3_ALLOW_LOOPBACK_HTTP=true`. The production Compose profile requires HTTPS.

Choose how processes coordinate access to this prefix.

| `REGIMEX_S3_COORDINATION` | Deployment |
| --- | --- |
| `local` (default) | Run all API processes, workers and maintenance commands on one host. Mount the same `REGIMEX_S3_LOCK_ROOT` into each process. |
| `postgres` | Use one PostgreSQL primary and Regimex schema across hosts. Each host can use its own private staging directory at `REGIMEX_S3_LOCK_ROOT`. See [multi-host deployment](multi-host.md). |

Use the same endpoint URL, bucket, prefix and limits everywhere. Reserve that
prefix exclusively for this installation. Provider lifecycle rules or tools
that change its objects bypass service coordination. To switch an installation
to PostgreSQL coordination, drain and stop all writers, apply the database
migration, then restart every process with `REGIMEX_S3_COORDINATION=postgres`
and a fresh staging directory. Existing provider objects stay in place.
The first operation that writes to the provider records the namespace and
limits. Subsequent changes to those settings require an
[offline backup and restore](#backup-and-restore) to a fresh target with the
new settings. Choose limits that cover every object in the archive.

In PostgreSQL mode, each active storage operation uses a dedicated database
session. Connect directly to PostgreSQL or configure a pooler in session mode.
Transaction and statement pooling release the backend holding the advisory locks.
Concurrent downloads pin and verify a specific object version and its
SHA-256. Downloads and uploads to different keys can overlap. Deletion waits for
those operations and removes only the verified generation. Before sending a new
object, a writer reserves bytes and one inventory entry in a committed database
transaction. If PostgreSQL fails over, the new primary must retain all committed
reservations. Automatic failover requires synchronous replication and fencing
the former primary so that only one server can admit writes.

| Limit | Default |
| --- | --- |
| `REGIMEX_MAX_OBJECT_BYTES` | 512 MiB, bounded single-part uploads |
| `REGIMEX_MAX_STORE_BYTES` | 4 GiB, including retained S3 payload versions |
| `REGIMEX_S3_MAX_INVENTORY_ENTRIES` | 100,000 retained versions and delete markers |
| `REGIMEX_S3_CONNECT_TIMEOUT_SECONDS` | 5 |
| `REGIMEX_S3_READ_TIMEOUT_SECONDS` | 10 per blocking read |
| `REGIMEX_S3_MAX_ATTEMPTS` | 3 (PostgreSQL-coordinated PUTs send one request per reservation) |

At the entry limit an upload to a new key fails. Existing versions remain
readable and retrying an identical upload succeeds. Staging an upload requires
additional local disk space.

Run `storage-check` to check connectivity, versioning, retained versions and
payload bytes. In PostgreSQL mode it also reports `pending_writes` and
`reserved_payload_bytes`. A write awaiting confirmation can appear in both
provider usage and reservations. Once confirmed, only the provider charge
remains. See the [deployment guide](deployment.md#s3-profile) for container
settings.

Workers renew their database leases while loading inputs, running the model
and uploading results. The attempt's time budget covers all three stages.
During these transfers, workers check for cancellation and timeouts between
reads. A blocked storage call can delay shutdown until it returns. A cancelled
attempt or one that lost its lease cannot publish a result. A completed upload
left behind by such an attempt can appear in an orphan cleanup plan.

## Stop writers before maintenance

For backup or orphan cleanup, stop new intake and drain or cancel pending work.
Then stop every writer, including API instances, workers, schedulers and external
upload or cleanup processes. For restore, stop processes that use the target.
Ensure exclusive access to the object namespace. Record the actual stop time
in a private JSON attestation.

```json
{
  "stopped_at": "2026-11-01T12:00:00Z",
  "statement": "All writers are stopped; this maintenance process owns the store.",
  "api_stopped": true,
  "analysis_workers_stopped": true,
  "administrative_workers_stopped": true,
  "maintenance_stopped": true,
  "garbage_collection_stopped": true,
  "exclusive_object_access": true
}
```

Replace the example timestamp. The tools check database jobs and sessions. The
operator must also stop anything that writes to the filesystem or S3 directly.

### Resolve unconfirmed S3 writes

If a node disconnects during an upload, PostgreSQL keeps the reservation.
Retrying the same key and content confirms a completed upload by its write
token, exact provider version and digest. Until confirmation, that key is
protected from deletion. Other keys can still use the remaining capacity.

Before backup, resolve outstanding reservations. Stop writers as described
above, then run:

```sh
uv run --locked --extra service --extra s3 regimex service storage-reconcile-writes \
  --attestation-file var/stopped.json
```

The command reports verified writes, released missing writes and remaining
reservations. A missing object can still be in flight at the provider. Only
after all former clients have stopped and the provider has finished processing
their outstanding requests, including timed-out requests, repeat the command
with `--provider-drained` to release reservations for absent objects. Each run
checks up to 1,000 keys. If the result includes a non-null `next_after`, pass
that value as `--after-key` to continue. Leave reservations in place if provider
completion cannot be established.

## Backup and restore

Use PostgreSQL 16 `pg_dump` and `pg_restore`, or the Compose `ops` image.
Set `PGDUMP_BIN` and `PGRESTORE_BIN` or pass `--pg-dump` and `--pg-restore`
to select the binaries. Backups require the schema revision supported by the
running release (currently `0016_s3_coordination`).

With all writers stopped:

```sh
uv run --locked --all-extras regimex service backup \
  --attestation-file var/stopped.json --output /private/backup-001
```

For Compose, put the attestation in a private host directory at
`$RECOVERY_DIR/stopped.json`. Give the operator container's UID 10001 read
access to that file and write access to the directory. The command creates a
new `backup` subdirectory. Use a fresh directory for each backup.

```sh
docker compose --profile ops run --rm --no-deps \
  --volume "$RECOVERY_DIR:/recovery" ops service backup \
  --attestation-file /recovery/stopped.json --output /recovery/backup
```

Keep the printed manifest SHA-256 separately from the archive, along with the
matching application release. Protect access to the archive. It includes
authentication hashes, experiment metadata and all registered files.

This release restores archives from schemas `0007_resource_deletion`,
`0008_analysis_metadata`, `0009_training_run_names`, `0010_evaluation_reuse`,
`0011_training_progress`, `0012_training_metrics`, `0013_training_archive`,
`0014_external_training`, `0015_terminal_activity` and `0016_s3_coordination`.
Point its configuration at a **new, empty database** and a **fresh local object
root or S3 prefix**. Use the original schema name. Restore checks the target's
schema inventory and large objects before copying files. Database tools use the
host, port, database and user of the verified connection, including when libpq
selects them from a service file or environment. Leave the target offline until
restoration succeeds and any required migrations have run.

```sh
uv run --locked --all-extras regimex service restore \
  --attestation-file var/target-stopped.json --input /private/backup-001 \
  --manifest-sha256 EXTERNALLY_RECORDED_SHA256
```

After checking both targets are empty, restore loads the database with intake
closed, verifies its contents, then copies the objects. Readiness opens only
after their new physical versions are committed. Running migrations or
restarting the API after an interrupted copy leaves intake closed. Existing
dataset and model identities are preserved. The target gets
a new maintenance identity, so cleanup plans from the original installation
cannot be applied there.

New backups use UTC when hashing database timestamps and can be restored across
server time zones. For an archive without `evidence_timezone`, use the time
zone of its source connection during restoration.

For offline backup and restore, the overall time limit also covers S3 inventory
requests, downloads, uploads and verification. On timeout the command stops the
transfer and releases its locks. A backup that times out copying an S3 object
stops before publishing its manifest. During restore, S3 may already have
accepted a write when the command times out waiting for confirmation.

If restore fails, discard the partial target and start again with an empty one.
Its readiness guard blocks intake with `503`. After success, apply any required
upgrade and migrations, restart the services, and check a saved report and a
model prediction before admitting users. Take a new backup after a schema upgrade.
For the container restore commands, see
[moving an installation to Compose](deployment.md#move-an-existing-installation-to-compose).

## Orphan cleanup

Orphan cleanup targets unregistered generations and incomplete uploads older
than 24 hours. Registered reports, experiments and models are retained
regardless of age. To free space by removing a saved result, use
[resource deletion](#delete-saved-resources).

After establishing the stopped-services conditions above:

```sh
uv run --locked --all-extras regimex service storage-gc-plan \
  --token-file var/credentials/admin.token --attestation-file var/stopped.json \
  --output var/orphan-plan.json
uv run --locked --all-extras regimex service storage-gc-apply \
  --token-file var/credentials/admin.token --attestation-file var/stopped.json \
  --plan-id PLAN_ID --plan-sha256 REVIEWED_SHA256
uv run --locked --all-extras regimex service storage-gc-receipt \
  --token-file var/credentials/admin.token --plan-id PLAN_ID
```

Inspect the plan before applying its recorded digest. Plans bind installation,
namespace, cutoff and exact generations; durable deletion fences prevent a stale
uploader from later registering a removed generation. Changed/young files and
registered hard-link aliases are preserved. A staging name left by an older writer
after publication shares the public file's generation. It can be cleaned up after
the public object is deleted and does not hold up quota reconciliation for other
keys. Unknown private entries fail planning.
Each plan includes up to 1,000 candidates. Create another plan after applying a
full batch. A pending resource deletion keeps its objects until its own retry.
Receipts report completion for the entire plan and return up to 1,000 audit
events per page. To read the next page, run `storage-gc-receipt` with
`--after-event-id` set to the returned `next_after_event_id`. Incomplete cleanup
exits nonzero. Manual deletion in the object root bypasses ownership accounting.

## Integration test setup

Use a disposable PostgreSQL cluster and set `REGIMEX_TEST_DATABASE_DSN` to its
test account. Queue tests need schema-creation permission; recovery tests need
`CREATEDB`, and worker-health process tests also need `CREATEROLE`. Set
`PGDUMP_BIN` and `PGRESTORE_BIN` to compatible PostgreSQL 16 clients, or provide
them on `PATH`.

The fixture requires OpenSSL and the pinned SeaweedFS 4.48 Linux amd64 binary.
It verifies the binary before starting the provider.

```sh
uv run --locked --all-extras --group service-test --group queue-test \
  python tests/support/run_storage_tests.py --binary /path/to/weed --tls -- \
  python -m pytest
```

`REGIMEX_TEST_SEAWEEDFS_BIN` can replace `--binary`. The helper starts a temporary
authenticated loopback provider, supplies credentials and its private test CA only
to the test subprocess, and stops its own processes afterward. Without the command
after `--`, it runs the object-store/S3 tests only. Missing database permissions or
recovery tools produce skipped tests; those skips do not verify the affected paths.
