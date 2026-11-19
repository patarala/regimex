# Local installation and updates

The native installer runs Regimex on Linux under your systemd user manager.
It creates a private PostgreSQL 16 cluster and serves the browser on loopback.
For a public HTTPS endpoint, see [deployment](deployment.md).

## Install

You need uv 0.13.0, PostgreSQL 16 binaries, a working `systemctl --user` and
at least 3 GiB free to stage the runtime. Obtain a committed release ZIP with
both saved models and its trusted SHA-256 digest. The installer uses uv to
provision Python 3.14.4 and the locked dependencies.

Run this as your ordinary user from a checkout or unpacked release containing
`scripts/install_service.py`. Substitute the archive path and its digest:

```sh
uv run --no-project --python 3.14.4 scripts/install_service.py \
  --archive release/build/regimex-0.1.0.zip \
  --sha256 ARCHIVE_SHA256 \
  --root var \
  --postgres-bin /usr/lib/postgresql/16/bin
```

Use a persistent installation path containing only ASCII letters, digits,
`/`, `.`, `_` and `-`. You can omit `--postgres-bin` if the three server
binaries (`pg_ctl`, `postgres`, `initdb`) are together on your PATH or in
`/usr/lib/postgresql/16/bin`.

The installer verifies the archive, migrates the database, imports the two
models and activates the TCN. Once the API and three workers pass verification,
open `http://127.0.0.1:8000/` and sign in with the contents of
`var/credentials/user.token`. Use `var/credentials/admin.token` for
administration. See [using Regimex](usage.md) for the first analysis.

```sh
var/service.sh status
var/service.sh doctor
```

Use `var/service.sh stop` and `var/service.sh start` to manage the installation.
The host's systemd user login or linger policy determines whether it runs after
logout. For a second instance choose a separate root, unit name and HTTP port,
for example `--root var/other --name regimex-other --port 8001`. PostgreSQL
uses a socket inside that root (`--postgres-port` defaults to 55439).

## Update or repeat an installation

Take a coordinated backup, then run the install command with the new archive
and digest. Keep the original root, name and ports. The installer remembers
the PostgreSQL binary path. Repeating a completed installation checks the
running services. If a check fails, inspect `var/service.sh status` and rerun
the same command after the services recover. An update preserves credentials,
saved experiments and the active model.

The updater stages dependencies while the service is available, then stops
intake and checks for unfinished jobs. If this first check finds work, the
API resumes. Let the jobs finish and repeat the command. The updater checks
again after stopping the workers and before migrating the database.

## Back up and check restoration

The installer's PostgreSQL binary directory must also contain `pg_dump` and
`pg_restore` from PostgreSQL 16. With PostgreSQL running, use a fresh backup
directory and finish any pending deletion cleanup. Run the commands from the
directory containing `var/`.

```sh
var/service.sh backup --output var/backups/backup-001
```

The helper pauses intake, checks for pending jobs and stops the managed writers.
It backs up the database and registered objects, verifies the copies and restores
the previous service states. If jobs are still active, let them finish and retry.
Keep backups private. Retain your release archive, environment and credential
files separately.

Record the returned `manifest_sha256` separately. To check that the backup
can actually be restored, use that value in:

```sh
var/service.sh verify-backup --input var/backups/backup-001 \
  --manifest-sha256 RECORDED_MANIFEST_SHA256
```

Verification starts a disposable PostgreSQL cluster on a private Unix socket.
It restores the database and objects, applies any required migration to the
disposable copy, checks readiness and loads the active model. Allow space for
the restored data. See [supported archive revisions](storage.md#backup-and-restore).
Success returns `restore_verified` and removes the temporary cluster and files.

Run maintenance through `service.sh` so that updates, backups and restarts
share the installation lock. If a backup or restore check is interrupted, run:

```sh
var/service.sh resume-maintenance
```

This reads the maintenance journal and resumes the recorded services or
cleans up the verification cluster. If recovery fails, check `service.sh status`
and `var/logs/maintenance.log`. Keep the journal for recovery and use a new
destination for a subsequent backup. For externally managed writers, follow
the [offline backup procedure](storage.md#stop-writers-before-maintenance).

## Adopt an existing native installation

Use `--adopt` only for a native installation with `runtime/installation.json`,
a private `service.env`, local storage and the matching systemd units:

```sh
uv run --no-project --python 3.14.4 scripts/install_service.py \
  --archive release/build/regimex-0.1.0.zip --sha256 ARCHIVE_SHA256 \
  --root /absolute/project/var --adopt
```

Supply the installation's custom name and ports if applicable. Adoption checks
ownership and unit contents before attaching the new runtime to the existing
data and credentials.

## Interrupted or failed updates

Check the phase in `var/local-installation.json` and the private log at
`var/logs/installation.log`. For the default API unit, use
`journalctl --user -u regimex-api.service`.

While an update stages dependencies, the installation record and `service.sh`
continue to describe the active release. If the first queue check finds work,
the installer restores the previous API state. You can use the usual maintenance
commands after either a staging failure or a refused update.
Both the updater and the backup command check the queue in the schema set by
`REGIMEX_DATABASE_SCHEMA` in `service.env`.

An earlier installer may have left `staging`, `staged`, or `intake_resumed` in
the record while the previous release is still serving. To recover maintenance
access, run the current installer with the original archive whose SHA-256
matches the directory name in `active_release`. Use its independently
recorded digest when invoking the installer. Supply the original installation root,
service name and ports. The installer checks the release, systemd units, running
processes, database readiness and model health before restoring the active
record. This check runs while the services are online. If it fails, inspect
the installation log and complete the interrupted candidate update once the
queue is idle.

The installer records the candidate before stopping workers. If shutdown,
migration or startup fails after that point, inspect the units and the
installation log, correct the problem and rerun the same archive and digest.
PostgreSQL stays available for inspection. Restoring an earlier database
requires its coordinated backup.
