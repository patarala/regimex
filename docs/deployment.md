# Deployment

Run these commands from the checkout root on Linux with Docker Engine,
BuildKit and Docker Compose. The S3 configuration requires Compose 2.24.4 or
later. The service profile starts PostgreSQL 16, the HTTPS API and three
workers. The ops profile provides administrative commands.

The Dockerfile builds one wheel for all three application images. The API
image serves requests and runs maintenance, the worker image adds the CPU model
and report dependencies, and the ops image adds PostgreSQL backup and restore
tools. Python, uv and PostgreSQL are pinned by digest. BuildKit caches locked
dependencies across builds, so source or README edits only rebuild the wheel
and its installation.

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

The server certificate must cover the hostname in `REGIMEX_PUBLIC_ORIGIN`.
Local Compose file secrets retain their host ownership and mode. The following
example is for a fresh installation on a rootful Linux engine. Set `TLS_DIR`
to the absolute directory containing your issued `api.crt`, `api.key` and
`api-ca.crt`. On a rootless engine, translate the container IDs in the table
to their mapped host IDs before assigning ownership.

```sh
export REGIMEX_SECRETS_DIR="$PWD/var/secrets"
(
  set -eu
  umask 077
  : "${TLS_DIR:?Set TLS_DIR to the directory containing the issued certificate and key}"
  for file in api.crt api.key api-ca.crt; do test -r "$TLS_DIR/$file"; done
  mkdir -p "$PWD/var"
  mkdir "$REGIMEX_SECRETS_DIR"
  cd "$REGIMEX_SECRETS_DIR"
  openssl rand -hex 32 > database-password
  openssl rand -hex 32 > application-password
  sed 's|.*|postgresql://regimex:&@postgres:5432/regimex|' application-password > database-dsn
  install -m 0644 "$TLS_DIR/api.crt" "$TLS_DIR/api-ca.crt" .
  install -m 0600 "$TLS_DIR/api.key" .
  sudo chown 999:999 database-password application-password
  sudo chown 10001:10001 database-dsn api.key
)
```

The command stops if the secrets directory already exists. For an existing
database, keep its role credentials. The PostgreSQL image runs
[`init-db.sh`](../deploy/init-db.sh) only on first initialization.

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

For a new installation, provision the files above. For object storage on S3,
configure the [S3 profile](#s3-profile) before starting PostgreSQL and the service.
To keep an existing registry, models and analyses, start with
[moving the installation](#move-an-existing-installation-to-compose).

```sh
export REGIMEX_SECRETS_DIR="$PWD/var/secrets"
docker compose --profile service --profile ops config --quiet
docker compose --profile service --profile ops build api analysis-worker ops
docker compose --profile service pull postgres
docker compose --profile service up --detach --wait
docker compose --profile service ps
```

The default URL is `https://localhost:8443`, bound to loopback.
`REGIMEX_HTTPS_PORT` selects a different local port. For a public deployment
set `REGIMEX_PUBLIC_ORIGIN` to the external HTTPS URL and configure
`REGIMEX_BIND_ADDRESS` and the host firewall. PostgreSQL stays on the
internal network. The database health check logs in with the application role
over TCP. Migrations run once that query succeeds, followed by the API and workers.

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
docker compose --profile service --profile ops build api analysis-worker ops
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
| `REGIMEX_HTTP_CONCURRENCY_LIMIT` | 8 active requests per API process, minimum 2. Reads of static assets and the liveness probe are exempt. |
| `REGIMEX_HTTP_CONNECTION_LIMIT` | Active request limit + 24 (32 by default), minimum 2. Uvicorn returns 503 when the number of open connections or tasks reaches this threshold. |
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
export COMPOSE_FILE=compose.yaml:compose.s3.yaml
```

With these variables set, follow [Start the service](#start-the-service).
Keep `COMPOSE_FILE` set for every service and operator command in this installation.
The endpoint has no credential, path or query component; the prefix has no leading
or trailing slash. All writers share the `s3-coordination` volume at
`/coordination` on one host and use the same namespace, byte limits and
inventory limit. The extra storage network permits outbound HTTPS. The
database migration needs only its DSN and schema. After rotating the credential
file, recreate the storage clients' containers to remount the file and renew
their SDK clients.
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

### Update individual components

For a change compatible with the running schema, queue payloads and saved data,
open the new checkout with the deployment's Compose configuration. Keep the same
project name (`COMPOSE_PROJECT_NAME` when set), secrets, image tag and environment.
Give each installation on a shared engine its own `REGIMEX_IMAGE_TAG`.
For S3, keep `COMPOSE_FILE=compose.yaml:compose.s3.yaml` and the provider settings.
Run one update at a time.

```sh
# Web UI or HTTP handlers
python3 deploy/update_components.py api

# Training code
python3 deploy/update_components.py admin-worker

# Report format that requires a new reader
# Backup and restore validate saved report metadata with the operator image.
docker compose --profile ops build ops && python3 deploy/update_components.py api analysis-worker
```

Select any combination of `api`, `analysis-worker`, `admin-worker` and
`maintenance-worker`. The helper builds each required image once and checks the
installed schema, queue schema and restore state from the new image before
replacing a container. It preserves the current replica count of each selected
role, including workers scaled with `--scale`. PostgreSQL keeps serving traffic.
Compose recreates roles whose image or configuration changed, and waits up to
three minutes for health. Before each replacement the helper checks the replica
counts again and verifies that roles updated earlier are still healthy. Repeating
the command with the same image and configuration keeps healthy containers
running. The singleton API can briefly drop requests, and queued work can wait
while its workers restart.

The analysis worker publishes saved HTML reports. For a new report format, update
the API and the backup/restore image before the writer. Training and maintenance
workers can keep running through a renderer change. For changes shared by multiple
queues, include each affected worker. A rebuilt image tag can be shared by several
roles. Their existing containers keep their original image. A later unscoped
`docker compose up` can bring those roles onto the rebuilt tag as well.
For secret rotation that needs a new file mount, explicitly recreate each
affected role as described in the [S3 profile](#s3-profile).

On failure, inspect `docker compose --profile service ps --all` and
`docker compose logs --tail 100 SERVICE` for the affected role. A failed build or
schema probe leaves the current containers running. If a health check fails after
replacement, earlier roles can already be on the new image. Correct the failure
and repeat the command. Docker may be restarting a failed container while the
helper builds its replacement. For a role you explicitly stopped or paused,
restore its intended running state first. Rolling back a reader requires that it
can open every record the updated writer has already saved.

### Upgrade an existing deployment

Use this coordinated procedure for schema migrations or incompatible changes to
operation payloads and stored data.

Block intake, drain pending jobs, stop all writers and take a
[coordinated backup](storage.md#backup-and-restore) with the current release
before replacing its images. Keep PostgreSQL running. Open the new checkout
with the same Compose project name, secrets directory and storage configuration.
If multiple installations share an engine, assign each its own `REGIMEX_IMAGE_TAG`
to keep their locally built images separate.

For an application upgrade on the existing PostgreSQL deployment, build the
new images, run their migrations, then recreate the API and workers. If a step
fails, the shell stops before starting the next one.

```sh
(
  set -e
  docker compose --profile service --profile ops build api analysis-worker ops
  docker compose --profile service run --rm --no-deps migrate
  docker compose --profile service up --detach --wait --force-recreate --no-deps \
    api analysis-worker admin-worker maintenance-worker
  docker compose --profile ops run --rm --no-deps ops service doctor
)
```

Verify a saved-model analysis before reopening intake.

### Worker shutdown and recovery

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
python3.14 tests/support/compose_service_smoke.py
```

The test runs an analysis with a small baseline, recreates the API and worker,
then backs up the database and objects. It restores the backup into a second
Compose project with fresh volumes. The original user token must open the
saved report, and a new analysis using the restored active model must reproduce
the predictions. The test removes both projects and its image tags. Use
`--no-build` to reuse existing images. Docker and permission to assign
secret-file ownership are required.

To check the TCN you plan to deploy, place the verified
[release asset pack](releasing.md#include-saved-models-examples-and-editable-documents)
in `release/` or pass its directory after `--release-tcn`. The test reads the
manifest digest from the pack's `SHA256SUMS`.

```sh
python3.14 tests/support/compose_service_smoke.py --release-tcn
```

For the optional S3 test, set the endpoint, bucket and test prefix in the
shell, then pass `--s3-credentials-file` and `--s3-ca-file`. The source and
restored installations each use a fresh prefix. Configure the test bucket's
lifecycle to expire their object versions.
