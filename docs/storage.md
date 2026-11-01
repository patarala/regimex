# Storage and recovery

PostgreSQL controls resource ownership, state and publication. Immutable local or
S3 objects hold observations, models and reports. Uploading bytes does not publish
a resource. Only an authorized application commit can register a checked object;
a digest or provider version alone does not authorize deletion.

Hash-bound JSON uses PostgreSQL `json`, preserving canonical numeric representations
through reads and backups, including signed zero and exponent-form floats. Other
metadata can use `jsonb`. A content/hash mismatch fails explicitly; it is never
repaired by replacing the expected digest.

## Local storage

The default adapter requires a dedicated Linux/POSIX directory with advisory
locks, hard links, file/directory fsync and user extended attributes. Unsupported
filesystem guarantees fail explicitly. Keep the persistent object root separate
from temporary work/cache space. Identical uploads are idempotent; different bytes
cannot overwrite an existing key. Reads verify committed bytes and generation.

Model cache entries bind bundle and inference-policy hashes and are verified
before use. Capacity is bounded across workers. A full cache fails explicitly;
an operator can clear it with workers stopped. It is disposable and is not a backup.

## S3 storage

The verified provider is SeaweedFS 4.48, with conditional creation, enabled object
versioning and verified HTTPS. Its Linux amd64 archive SHA-256 is
`4a7d108384d044d95212d1342cdda9533fa55842c1c9b41f606ca3c8a9561124`.
Provider support does not imply equivalent behavior on another S3 service.

For native processes:

```sh
export REGIMEX_OBJECT_BACKEND=s3
export REGIMEX_S3_ENDPOINT_URL=https://objects.example.org
export REGIMEX_S3_REGION=us-east-1
export REGIMEX_S3_BUCKET=private-regimex
export REGIMEX_S3_PREFIX=installation-one
export REGIMEX_WORK_ROOT=/var/lib/regimex/work
export REGIMEX_S3_LOCK_ROOT=/var/lib/regimex/s3-coordination
uv run --locked --extra service --extra s3 regimex service storage-check
```

Credentials use the SDK's environment/profile/role chain;
`REGIMEX_S3_CREDENTIAL_PROFILE` selects a profile and `REGIMEX_S3_CA_BUNDLE`
specifies a private CA. The endpoint cannot contain credentials, a path, query or
fragment. HTTP is allowed only for literal loopback with
`REGIMEX_S3_ALLOW_LOOPBACK_HTTP=true`. The production Compose profile requires HTTPS.

Every writer to the same bucket/prefix must use the same dedicated coordination
directory and byte limits on one host. Other hosts and out-of-band writes are
unsupported. Operations check versioning, pin exact provider versions and hash
actual bytes; ETags and user metadata are not integrity evidence.

| Limit | Default |
| --- | --- |
| `REGIMEX_MAX_OBJECT_BYTES` | 512 MiB, bounded single-part uploads |
| `REGIMEX_MAX_STORE_BYTES` | 4 GiB, including retained S3 payload versions |
| `REGIMEX_S3_MAX_INVENTORY_ENTRIES` | 100,000; excess fails |
| `REGIMEX_S3_CONNECT_TIMEOUT_SECONDS` | 5 |
| `REGIMEX_S3_READ_TIMEOUT_SECONDS` | 10 per blocking read |
| `REGIMEX_S3_MAX_ATTEMPTS` | 3 |

`storage-check` verifies connectivity, enabled versioning and bounded inventory.
It does not upload/delete probes or establish model readiness. Store lock waits
are bounded to five seconds. Temporary staging requires additional disk space.
The [deployment guide](deployment.md#s3-profile) covers container settings.

## Stop writers before maintenance

Backup, restore and orphan cleanup are offline operations. Stop API, all workers,
schedulers, uploaders and cleanup processes, including external instances. Drain
or cancel pending work and establish exclusive access to the store namespace.
Create a private JSON attestation with the actual current stop time:

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

Replace the example timestamp. This is an operator assertion, not automatic proof
of filesystem exclusivity. Database checks reject pending/active jobs and other
application sessions. Their locks cannot stop independent filesystem writers.

## Backup and restore

Install PostgreSQL 16 `pg_dump` and `pg_restore`, or use the Compose `ops` image.
Set `PGDUMP_BIN`/`PGRESTORE_BIN` or pass `--pg-dump`/`--pg-restore` if needed.
Archives must use the current database revision, `0008_exact_json`, including its
restore-readiness guard. Use an explicit schema migration before backing up a
database at another revision; restore does not reinterpret incompatible archives.

With all writers stopped:

```sh
uv run --locked --all-extras regimex service backup \
  --attestation-file var/stopped.json --output /private/backup-001
```

Record the printed manifest SHA-256 separately. The backup contains the database
dump, every registered object, checksums and table evidence. Protect it like
credentials because it includes authentication hashes and private metadata.
A failed backup is not usable. Database metadata alone is not a complete backup.

Restore into a **new empty database** and a **fresh empty local root or S3 prefix**.
Keep the original schema name, configure the target environment, and stop every
target process. Do not run migrations before restore:

```sh
uv run --locked --all-extras regimex service restore \
  --attestation-file var/target-stopped.json --input /private/backup-001 \
  --manifest-sha256 EXTERNALLY_RECORDED_SHA256
```

Restoration verifies the dump and inventory, restores database/object bytes and
rebinds physical generations. Copying objects alone is insufficient: local inode
identities and S3 version IDs change. Content/dataset/model identities remain
unchanged. The installation receives a new maintenance identity, so old cleanup
plans cannot apply to the restored target.

A failed restore may leave a partial offline target. Discard that target and use
a new empty one; never serve or overwrite it. Its readiness guard rejects intake
with `503`, and rerunning migration cannot mark it ready. After success, start
services and verify authorized saved reports and actual model inference before
opening intake. Rebuild disposable caches as needed.

## Orphan cleanup

Cleanup retains **every registered object**, including referenced candidates,
models, datasets, reports and evidence. It only removes unregistered generations
and incomplete uploads older than 24 hours. It is not a policy for deleting old
application resources to save space.

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
registered hard-link aliases are preserved. Unknown private entries fail planning.
Receipts retain partial failures and support safe replay; incomplete cleanup exits
nonzero. Never replace this workflow with manual deletion in the object root.

## Integration test setup

Use a disposable PostgreSQL cluster and set `REGIMEX_TEST_DATABASE_DSN` to its
test account. Queue tests need schema-creation permission; recovery tests need
`CREATEDB`, and worker-health process tests also need `CREATEROLE`. Set
`PGDUMP_BIN` and `PGRESTORE_BIN` to compatible PostgreSQL 16 clients, or provide
them on `PATH`.

With OpenSSL and the verified SeaweedFS 4.48 Linux binary available:

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
