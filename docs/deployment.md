# Deployment

Use a Linux host with Docker Compose. The service profile runs PostgreSQL, an
HTTPS API, analysis/admin workers and a maintenance worker. API and maintenance
images contain service/S3 dependencies; CPU workers add ML/report dependencies.
The operator image adds PostgreSQL recovery clients. Images build the same wheel
from the same locked dependencies.

The images pin Python, uv and PostgreSQL by digest. This configuration uses
PostgreSQL 16. To verify a host installation, run the
[container smoke test](#verify-a-container-installation).

## Secrets and TLS

Provision private files outside Git. Set `REGIMEX_SECRETS_DIR` to their directory,
for example `$PWD/var/secrets`. The application role owns its dedicated database
and schema but has no superuser, database-creation, role-creation or replication
privileges. Its password differs from the PostgreSQL bootstrap password.

| File | Content | Container owner / permissions |
| --- | --- | --- |
| `database-password` | PostgreSQL bootstrap password | `999:999`, `0600` |
| `application-password` | Application-role password | `999:999`, `0600` |
| `database-dsn` | `postgresql://regimex:<URL-encoded-password>@postgres:5432/regimex` | `10001:10001`, `0600` |
| `api.crt` | Server certificate and intermediate chain | Readable by `10001` |
| `api.key` | Unencrypted server private key | `10001:10001`, `0600` |
| `api-ca.crt` | Trusted CA chain for the TLS health probe | Readable by `10001` |

The certificate SAN must match `REGIMEX_PUBLIC_ORIGIN`. Keep the parent directory
private. Local Compose file secrets preserve host ownership/mode; their Compose
`uid`/`gid`/`mode` fields cannot remap bind-mounted files. Account for user-namespace
mapping on rootless installations. Do not place secrets in images, source, task
payloads, URLs or command history.

The API terminates TLS directly and supports a TCP pass-through proxy. It
ignores forwarded headers. Keep TLS intact up to the API. For a remote
PostgreSQL server, configure the DSN to verify its certificate against a trusted
CA.

Start the API with `regimex service serve`. By default, clients have 10 seconds
to finish sending headers and 30 seconds to finish sending the body. Both clocks
measure elapsed time even while bytes are arriving. An expired request closes
with HTTP `408` if a response has not started. Model and application processing
after the upload use separate budgets. An alternative ASGI host must enforce
its own header deadline because the application only sees completed headers.

## Start the service

For a new installation, provision the files above and run the commands below.
To keep an existing registry, models and analyses, start with
[moving the installation](#move-an-existing-installation-to-compose).

```sh
export REGIMEX_SECRETS_DIR="$PWD/var/secrets"
export REGIMEX_PUBLIC_ORIGIN=https://localhost:8443
docker compose --profile service config
docker compose --profile service build api analysis-worker
docker compose --profile ops build ops
docker compose --profile service pull postgres
docker compose --profile service up --detach
docker compose --profile service ps
```

Use the real public origin and a matching certificate. The default host bind is
`127.0.0.1:8443`; configure `REGIMEX_BIND_ADDRESS` and `REGIMEX_HTTPS_PORT` with an
appropriate firewall when exposing it. PostgreSQL is not published to the host.

The migration service waits for PostgreSQL to become healthy. The API and
workers start once migration exits successfully.

Check storage, then create an administrator token and a user token.

```sh
docker compose --profile ops run --rm ops service storage-check
docker compose --profile ops run --rm ops service token-create \
  --owner admin --role admin --output /operator-output/admin.token
docker compose --profile ops run --rm ops service token-create \
  --owner researcher --role user --output /operator-output/user.token
```

Only the operator container mounts the token volume. To sign in as the ordinary
user, copy the token to a new private file on the host and paste its contents
into the workspace sign-in form.

```sh
(
  umask 077
  set -C
  mkdir -p var/credentials
  docker compose --profile ops run --rm --no-deps -T --entrypoint cat ops \
    /operator-output/user.token > var/credentials/user.token
)
```

To activate a release model, use the bundle path and manifest digest from the
verified archive's `SHA256SUMS`. For the TCN in an extracted archive containing
`release/models/tcn/manifest.json`, run this from the archive root.

```sh
manifest_sha256=$(awk '$2 == "release/models/tcn/manifest.json" {print $1}' SHA256SUMS)
docker compose --profile ops run --rm -v "$PWD/release/models/tcn:/bundle:ro" ops \
  service register-model --bundle /bundle --manifest-sha256 "$manifest_sha256" \
  --token-file /operator-output/admin.token --activate
```

Use the [administrative API](api.md#administration) to train and publish a
candidate in this installation.

## Move an existing installation to Compose

1. Block client and simulator traffic to the source. Drain or cancel pending jobs
   and finish pending deletions. Make a [coordinated backup](storage.md#backup-and-restore)
   of the database and its registered objects. If the installation has its own
   `service.sh`, you can run `./service.sh backup --output PATH` in its directory.
   The helper resumes managed services when it finishes. Keep traffic blocked
   and stop the old instance and its automatic startup before cutover.
2. Record the printed manifest SHA-256 separately. Keep the original data,
   configuration and token files private. The backup preserves token hashes and
   model registrations. The token files are still needed to sign in.
3. Configure [Compose secrets and TLS](#secrets-and-tls) for the destination.
   The database DSN uses the new database role credentials. Choose an unused
   Compose project name so PostgreSQL and the local object store get fresh
   volumes. For S3, use a fresh prefix in a versioned bucket and the [S3 override](#s3-profile)
   on every command. Use the source schema name for all target processes.
   If it differs from the default `public`, set `REGIMEX_DATABASE_SCHEMA` in
   the destination's Compose `.env` before restoring.
4. Set `RECOVERY_DIR` to an absolute path containing a private copy of the
   backup directory named `backup` and a [target attestation](storage.md#stop-writers-before-maintenance)
   named `target-stopped.json`. Make both readable by the container's UID 10001.
   Set `BACKUP_SHA256` to the digest recorded at backup time. Start only
   PostgreSQL, then restore before running any migrations.

```sh
export COMPOSE_PROJECT_NAME=regimex-target
docker compose --profile service build api analysis-worker
docker compose --profile ops build ops
docker compose --profile ops up --detach --wait postgres
docker compose --profile ops run --rm --no-deps \
  --volume "$RECOVERY_DIR:/recovery:ro" ops service restore \
  --input /recovery/backup --manifest-sha256 "$BACKUP_SHA256" \
  --attestation-file /recovery/target-stopped.json
```

After restore succeeds, keep the same project name and run the following commands.
The service profile applies migrations before starting the API and workers.

```sh
docker compose --profile service up --detach --wait
docker compose --profile ops run --rm --no-deps ops service doctor
```

Sign in with an existing user token, open a saved report and run an analysis
with the restored active model before directing clients to the new origin.
Keep the old instance offline. If restore fails, keep the source data and use
a fresh destination for the next attempt. A database that has already been
migrated or used for another install is ineligible for restore.

## Configuration and capacity

Processes started from source accept `REGIMEX_DATABASE_DSN` or a private, owned UTF-8 file at
`REGIMEX_DATABASE_DSN_FILE`, never both. Object/work roots must be distinct;
use absolute paths when processes have different working directories. Invalid
or ineffective backend settings fail explicitly.

| Environment variable | Default |
| --- | --- |
| `REGIMEX_OBJECT_BACKEND` | `local` |
| `REGIMEX_OBJECT_ROOT` / `REGIMEX_WORK_ROOT` | `var/objects` / `var/work` when started from source |
| `REGIMEX_DATABASE_SCHEMA` | `public` |
| `REGIMEX_PUBLIC_ORIGIN` | `https://localhost` when started from source |
| `REGIMEX_SESSION_TTL_SECONDS` | 1,800 seconds, absolute expiry |
| `REGIMEX_HTTP_HEADER_TIMEOUT_SECONDS` | 10 seconds; range 1–60 |
| `REGIMEX_HTTP_BODY_TIMEOUT_SECONDS` | 30 seconds; range 1–300 |
| `REGIMEX_HTTP_CONCURRENCY_LIMIT` | 8 connections or tasks per API process, minimum 2 |
| `REGIMEX_MAX_SESSIONS_PER_TOKEN` | 8 |
| `REGIMEX_MAX_PENDING_PER_OWNER` / `REGIMEX_MAX_ADMIN_PENDING_PER_OWNER` | 8 / 4 |
| `REGIMEX_MAX_GENERATION_EXPERIMENTS` | 128 |
| `REGIMEX_MAX_TRAINING_SEEDS` / `REGIMEX_MAX_TRAINING_EPOCHS` | 8 / 1,000 |
| `REGIMEX_MAX_TRAINING_UPDATES` | 32,000 successful updates per seed |
| `REGIMEX_ANALYSIS_TIME_BUDGET_SECONDS` | 120 per attempt |
| `REGIMEX_ANALYSIS_MEMORY_BUDGET_BYTES` / `REGIMEX_ANALYSIS_SCRATCH_BUDGET_BYTES` | 4 GiB / 128 MiB |
| `REGIMEX_ADMIN_TIME_BUDGET_SECONDS` | 3,600 |
| `REGIMEX_ADMIN_MEMORY_BUDGET_BYTES` / `REGIMEX_ADMIN_SCRATCH_BUDGET_BYTES` | 4 GiB / 4 GiB |
| `REGIMEX_HEARTBEAT_SECONDS` / `REGIMEX_STALE_SECONDS` | 10 / 120; stale must allow at least three heartbeats |
| `REGIMEX_MAX_OBJECT_BYTES` / `REGIMEX_MAX_STORE_BYTES` | 512 MiB / 4 GiB |
| `REGIMEX_MAX_OWNER_STORAGE_BYTES` / `REGIMEX_MAX_INTEGRATION_STORAGE_BYTES` | 1 GiB per owner / 2 GiB per connection, including reservations |
| `REGIMEX_CACHE_BUDGET_BYTES` | 512 MiB |

Compose fixes some infrastructure values in its shared environment mapping and
forwards analysis budgets, integration limits and storage quotas from the shell. Inspect `docker compose
config` after changing limits. Child memory limits are virtual address space,
not measured resident memory. Requests must fit both application and container limits.

Containers use UID 10001, a read-only root, dropped capabilities, PID limits and
bounded tmpfs. Each CPU worker handles one task at a time, with default 8 GiB RAM,
2 CPUs and 5 GiB scratch. Size the host for both workers running together,
PostgreSQL, the API and the maintenance worker. Operator jobs need additional headroom. Reserve disk space for the
database, retained objects, container images and backups.

For replica counts, worker CPU placement and multi-host constraints, see
[scaling Regimex](architecture.md#scaling-regimex).

## Measure analysis under load

For a test Compose deployment, run k6 against its HTTPS API as described in
the [load testing guide](load-testing.md#running-against-an-existing-test-installation).
For repeatable capacity comparisons, use the disposable stand with a writable
delegated cgroup v2 directory for the database, workers, API and load generator.

```sh
sh scripts/load_test.sh --stand --cgroup-parent "$REGIMEX_LOAD_CGROUP_PARENT" --bundle release/models/tcn
```

The [load testing guide](load-testing.md) covers prerequisites, individual
profiles, resource limits, reports and running against an existing test instance.
The [analysis timing report](analysis-timings.md) explains the queue, worker
and computation measurements.

## S3 profile

The supported provider is SeaweedFS 4.48 over verified HTTPS. Provision a private
bucket with versioning **Enabled**, a dedicated nonempty prefix and provider-issued
credentials. Another S3 implementation requires its own guarantee/integration tests.

Add `s3-credentials` to the secrets directory, owned by `10001:10001`, mode `0600`:

```ini
[regimex]
aws_access_key_id = PROVIDER_ISSUED_ACCESS_KEY
aws_secret_access_key = PROVIDER_ISSUED_SECRET_KEY
```

Add the trusted provider CA as `s3-ca.crt`. Do not include a CA private key or
provider administrator credentials. Configure only the nonsecret location:

```sh
export REGIMEX_S3_ENDPOINT_URL=https://objects.example.org
export REGIMEX_S3_REGION=us-east-1
export REGIMEX_S3_BUCKET=private-regimex
export REGIMEX_S3_PREFIX=installation-one
docker compose -f compose.yaml -f compose.s3.yaml --profile service config
docker compose -f compose.yaml -f compose.s3.yaml --profile service up --build -d
docker compose -f compose.yaml -f compose.s3.yaml --profile ops run --rm ops service storage-check
```

Use both Compose files for every service/operator command in this installation.
The endpoint has no credential, path or query component; the prefix has no leading
or trailing slash. All writers must share `/coordination` on one host and use
the same namespace, byte limits and inventory limit. The extra storage network permits outbound HTTPS without publishing provider ports.
Rotate the credential file and restart services to replace cached SDK clients.
See [storage](storage.md) for S3 settings, quotas and recovery.

## Health, shutdown and maintenance

`/health/live` checks the API; `/health/ready` checks schema and restore readiness.
`/health/models` checks active deployment metadata without testing inference.
`service health --component worker --queue analysis` checks the registered local
worker and fresh database heartbeat. A completed saved-model analysis is the
functional check for backend loading, prediction, report rendering and persistence.

For a combined diagnostic report, run from the operator image with the same
configuration and storage mounts:

```sh
docker compose --profile ops run --rm ops service doctor
```

From a source checkout, use `uv run --locked --extra service --extra ml regimex
service doctor`; add `--extra s3` when configured. Checks cover connectivity,
schema/restore readiness, readable storage, the active model, fresh workers for
each queue and restoration of the model's verified weights/preprocessing in an
isolated local child. Add `--json` for machine-readable output. A required failed
or skipped check exits with status 1. The command does not change jobs or
publication and does not test remote worker environments or prediction quality.

Run `service migrate` when upgrading, then restart the API and workers together.
A worker's heartbeat can remain fresh for up to `REGIMEX_STALE_SECONDS` after
an abrupt shutdown. A busy worker can pass diagnostics while every slot is in use.

Logs are bounded JSON with route templates, status and elapsed time; credentials,
request bodies and arbitrary query strings are excluded. Docker rotates its local
logs. Protect logs as operational data.

Before maintenance, take the API out of service and drain or cancel queued work.
Stop the remaining writers and confirm their status.

```sh
docker compose stop api analysis-worker admin-worker maintenance-worker
docker compose --profile service ps --all
```

Graceful shutdown waits 30 seconds for jobs; Compose allows 45 seconds. Fencing
prevents abandoned attempts from overwriting later results. Maintenance retries
an abandoned job within its operation's attempt limit, including when the worker
dies after taking a delivery but before starting the operation. The next recovery
pass also closes abandoned maintenance jobs. Each pass clears up to 1,000
successful recovery jobs and their queue events. Run `regimex service recover`
to process another batch immediately. Failed jobs remain available for
diagnosis. Permanent data or configuration errors end the operation.
Recovery leaves operations with an unsupported contract version for a compatible
release to process. Their job endpoints return `unsupported_contract`. This query
lists affected operations in the service schema for operation contract 1.2.

```sql
SELECT operation_id, state, document->'version' AS version
FROM regimex_operations
WHERE state IN ('queued', 'running', 'cancel_requested')
  AND document->'version' IS DISTINCT FROM '"1.2"'::jsonb;
```

Backups, restoration and orphan cleanup require every writer stopped, including
processes outside Compose. Follow the [storage procedure](storage.md).

## Verify a container installation

```sh
python tests/support/compose_service_smoke.py
```

The test creates a disposable project and private credentials, builds images,
runs saved-model analysis, checks reports/access/restart/idempotency and removes
only its own resources. Docker must be available, with permission to assign
secret-file ownership.
