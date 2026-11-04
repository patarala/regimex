# Local installation and updates

The installer creates a persistent local service for Linux with a systemd user
manager. It uses a private PostgreSQL 16 cluster, local immutable storage and a
loopback HTTP listener. Run it as your ordinary user. Existing system PostgreSQL
clusters and other Regimex installations are not modified.

Prerequisites are uv 0.13.0, installed PostgreSQL 16 binaries, a working
`systemctl --user`, and at least 3 GiB free for staging a new runtime. uv installs
the exact Python version and dependencies recorded in the release. The bootstrap
command below explicitly uses Python 3.14.4; it needs no installed Regimex package.
The installer does not download PostgreSQL or run commands with sudo.

Use a release ZIP produced by `scripts/build_release.py --assets release`, with
both baseline and TCN model bundles. Obtain the archive and its SHA-256 digest
from a trusted source. Checksums detect changes; they do not establish who
published the archive. The installer executes the verified release's code.
Uncommitted preview archives and source-only archives are rejected.

## Install

From a checkout or unpacked release containing `scripts/install_service.py`:

```sh
uv run --no-project --python 3.14.4 scripts/install_service.py \
  --archive release/build/regimex-0.1.0.zip \
  --sha256 ARCHIVE_SHA256 \
  --root var \
  --postgres-bin /usr/lib/postgresql/16/bin
```

Replace `ARCHIVE_SHA256` with the 64-character archive digest. If `pg_ctl`,
`postgres` and `initdb` are already on PATH, `--postgres-bin` can be omitted.
Use an installation directory on a persistent filesystem, not a temporary RAM
filesystem. Paths currently accept letters, digits, `/`, `.`, `_` and `-`.

The command verifies every archive member, stages an isolated locked environment,
creates a nonsuperuser application database role and private access tokens,
applies migrations, imports both supplied models and activates the TCN. It then
starts the API and analysis, administration and maintenance workers. Readiness
and `service doctor` must pass before the installation is marked ready.

Open **http://127.0.0.1:8000/** and sign in using `var/credentials/user.token`.
The administrator token is in `var/credentials/admin.token`. Token contents
and database passwords are never printed by the installer.

```sh
var/service.sh status
var/service.sh doctor
var/service.sh stop
var/service.sh start
```

Services start with the systemd user manager. Starting that manager at boot or
keeping it after logout is a host administrator's login/linger policy. The
installer does not change that policy. The local HTTP service is not a public
HTTPS deployment; use the separate [deployment guide](deployment.md) for that.

Use `--name regimex-other --port 8001 --root var/other` for a separate installation.
The name is the stable systemd unit prefix. PostgreSQL listens only on a private
Unix socket under the chosen root; `--postgres-port` defaults to 55439. Keep the
chosen name and ports when updating.

## Update or repeat an installation

Run the same command with the new trusted archive and digest, keeping `--root`,
`--name` and ports. The saved PostgreSQL binary path is reused unless explicitly
provided. Repeating an already completed release performs health checks without
reinstalling dependencies into its running environment.

A new runtime is fully staged before intake stops. The updater then stops the
API, checks for queued/running/cancelling operations, stops the workers and
checks again. If the first check finds work, the original API resumes and the
update refuses to continue. Let the work finish and repeat the command. The
installer never cancels, discards or silently drains pending work.

Migrations run before the new application starts. Stored experiments, model
versions, reports, database credentials and access tokens are preserved. Updates
preserve the active model; they do not publish a new model merely because a
release includes another bundle. Old staged releases remain on disk. Database
major-version upgrades and arbitrary external database/S3 deployments are outside
this installer's scope.

Keep a coordinated database/object backup before upgrading an installation with
valuable data; see [storage and recovery](storage.md). Old code on disk is not a
rollback of a migrated database.

## Adopt an existing native installation

An installation previously started using the documented native `var/` layout
can be adopted explicitly:

```sh
uv run --no-project --python 3.14.4 scripts/install_service.py \
  --archive release/build/regimex-0.1.0.zip --sha256 ARCHIVE_SHA256 \
  --root /absolute/project/var --adopt
```

Adoption requires the recorded `runtime/installation.json`, private `service.env`
and database DSN, local `objects/` and `work/`, and matching recorded systemd unit
files. It checks their ownership/layout and uses the existing PostgreSQL binaries.
Specify the existing custom name or ports when applicable. It preserves existing
resources and credentials and rewrites the operator helper commands to use the
new runtime. Unrelated or locally modified unit files are refused for inspection.

## Interrupted or failed updates

`local-installation.json` records the current phase. Private command output is
in `logs/installation.log`; component logs remain available through
`journalctl --user -u regimex-api.service` and the corresponding worker unit.

A staging failure leaves the original application running. After workers have
stopped, a migration/startup/verification failure leaves the API and workers
stopped with phase `recovery_required`. PostgreSQL stays available for inspection.
There is no automatic database downgrade or restoration of old code against a
possibly newer schema. Correct the reported problem and rerun the same trusted
archive; completed staged environments are reused without modifying a runtime
that may have started before an interruption. Resume an incomplete update before
switching to a different release.

If work appeared during the second quiescence check, inspect the installation and
recover the existing delivery explicitly before retrying. Never edit committed
result digests or delete queued tasks to make an update pass.
