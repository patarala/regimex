# Deployment

Use a Linux host with Docker Compose. The service profile runs PostgreSQL, an
HTTPS API, analysis/admin workers and a maintenance worker. API and maintenance
images contain service/S3 dependencies; CPU workers add ML/report dependencies.
The operator image adds PostgreSQL recovery clients. Images build the same wheel
from the same locked dependencies.

The configured Python, uv and PostgreSQL images are pinned by digest. The current
PostgreSQL container uses major version 16. Native service/storage execution has
been verified; Docker image execution remains unverified. Run the container smoke
below before treating a host installation as validated.

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

The API terminates TLS directly; forwarded headers are disabled. A TCP TLS
pass-through proxy is supported. HTTP reverse-proxy termination requires a trusted
proxy feature that is not implemented. Cross-host PostgreSQL must use a DSN with
verified database TLS and its trusted CA. Regimex never needs simulator DB credentials.

## Start the service

After provisioning the files above:

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

The one-shot migration service waits for PostgreSQL health, then API/workers wait
for migration success. A successful migration container exiting is expected.
Pulling PostgreSQL explicitly fetches its pinned runtime image even when the
engine has only cached the Dockerfile build stage.

```sh
docker compose --profile ops run --rm ops service storage-check
docker compose --profile ops run --rm ops service token-create \
  --owner admin --role admin --output /operator-output/admin.token
docker compose --profile ops run --rm ops service token-create \
  --owner researcher --role user --output /operator-output/user.token
```

Token files are private inside the operator volume; their values are not printed.
Retrieve them using authorized host tooling. API/worker containers do not mount
this volume. Use the administrator token in `/docs` for the [model lifecycle](api.md#administration).

To import a release bundle, mount it read-only into an operator container and run
`service register-model --bundle /bundle --manifest-sha256 RECORDED_SHA256
--token-file /operator-output/admin.token --activate`. Ordinary analyses restore
saved models; startup never trains or activates one automatically.

## Configuration and capacity

Native processes accept `REGIMEX_DATABASE_DSN` or a private, owned UTF-8 file at
`REGIMEX_DATABASE_DSN_FILE`, never both. Object/work roots must be distinct;
use absolute paths when processes have different working directories. Invalid
or ineffective backend settings fail explicitly.

| Environment variable | Default |
| --- | --- |
| `REGIMEX_OBJECT_BACKEND` | `local` |
| `REGIMEX_OBJECT_ROOT` / `REGIMEX_WORK_ROOT` | `var/objects` / `var/work` in native processes |
| `REGIMEX_DATABASE_SCHEMA` | `public` |
| `REGIMEX_PUBLIC_ORIGIN` | `https://localhost` in native processes |
| `REGIMEX_SESSION_TTL_SECONDS` | 1,800 seconds, absolute expiry |
| `REGIMEX_MAX_SESSIONS_PER_TOKEN` | 8 |
| `REGIMEX_MAX_PENDING_PER_OWNER` / `REGIMEX_MAX_ADMIN_PENDING_PER_OWNER` | 8 / 4 |
| `REGIMEX_MAX_GENERATION_EXPERIMENTS` | 128 |
| `REGIMEX_MAX_TRAINING_SEEDS` / `REGIMEX_MAX_TRAINING_EPOCHS` | 8 / 1,000 |
| `REGIMEX_ANALYSIS_TIME_BUDGET_SECONDS` | 120 per attempt |
| `REGIMEX_ANALYSIS_MEMORY_BUDGET_BYTES` / `REGIMEX_ANALYSIS_SCRATCH_BUDGET_BYTES` | 4 GiB / 128 MiB |
| `REGIMEX_ADMIN_TIME_BUDGET_SECONDS` | 3,600 |
| `REGIMEX_ADMIN_MEMORY_BUDGET_BYTES` / `REGIMEX_ADMIN_SCRATCH_BUDGET_BYTES` | 4 GiB / 4 GiB |
| `REGIMEX_HEARTBEAT_SECONDS` / `REGIMEX_STALE_SECONDS` | 10 / 120; stale must allow at least three heartbeats |
| `REGIMEX_MAX_OBJECT_BYTES` / `REGIMEX_MAX_STORE_BYTES` | 512 MiB / 4 GiB |
| `REGIMEX_CACHE_BUDGET_BYTES` | 512 MiB |

Compose fixes some infrastructure values in its shared environment mapping and
forwards the three analysis budgets from the shell. Inspect `docker compose
config` after changing limits. Child memory limits are virtual address space,
not measured resident memory. Requests must fit both application and container limits.

Containers use UID 10001, a read-only root, dropped capabilities, PID limits and
bounded tmpfs. Each CPU worker handles one task at a time, with default 8 GiB RAM,
2 CPUs and 5 GiB scratch. The five persistent services have about 19.25 GiB of
combined memory limits. A 4-vCPU, 24 GiB RAM host is a starting capacity allocation;
concurrent operator jobs need additional headroom. This is not a measured minimum
or throughput guarantee. Account separately for retained objects, database data,
images and backups on disk.

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
or trailing slash. TLS verification stays enabled. All six storage-capable
services share `/coordination` on one host, with identical namespace and byte
limits. This is a required process-shared lock, not distributed coordination.
The extra storage network permits outbound HTTPS without publishing provider ports.
Rotate the credential file and restart services to replace cached SDK clients.
See [storage](storage.md) for native S3 settings, quotas and recovery.

## Health, shutdown and maintenance

`/health/live` checks the API; `/health/ready` checks schema and restore readiness.
`/health/models` checks active deployment metadata without testing inference.
`service health --component worker --queue analysis` checks the registered local
worker and fresh database heartbeat. A completed saved-model analysis is the
functional check for backend loading, prediction, report rendering and persistence.

Logs are bounded JSON with route templates, status and elapsed time; credentials,
request bodies and arbitrary query strings are excluded. Docker rotates its local
logs. Protect logs as operational data.

Before maintenance, stop intake and drain/cancel queued work, then stop all writers:

```sh
docker compose stop api analysis-worker admin-worker maintenance-worker
docker compose --profile service ps --all
```

Graceful shutdown waits 30 seconds for jobs; Compose allows 45 seconds. Fencing
prevents abandoned attempts from overwriting later results. Maintenance recovery
requeues eligible stale attempts; permanent data/configuration errors are not
retried. Backups, restoration and orphan cleanup require every writer stopped,
including processes outside Compose. Follow the exact [storage procedure](storage.md).

## Verify a container installation

```sh
python tests/support/compose_service_smoke.py
```

The test creates a disposable project and private credentials, builds images,
runs saved-model analysis, checks reports/access/restart/idempotency and removes
only its own resources. Docker must be available, with permission to assign
secret-file ownership. The test does not publish images or measure model quality.
