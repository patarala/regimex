# Platform access

A connection identifies a simulator backend by its Regimex account and source
namespace. Its key stays on the backend. The simulator signs users in and
checks their access to the run before each Regimex request. Regimex uses the
connection and verified subject to identify the owner. Replacing a key on the
same connection preserves access to that owner's saved analyses. See the
[delegation contract](integration.md) for request headers.

## Manage connections in the browser

Sign in with a Regimex administrator credential and select **Connections**.

1. Create a connection with a display name, integration account and source namespace.
   Use an account dedicated to the backend.
2. Select the platform by its account and export source. Open **Backend setup**
   to check its identifiers and verification URL. Issue a named key with the
   capabilities and lifetime the backend needs.
3. Copy the secret while it is displayed and store it in the simulator backend's
   private credential configuration. If clipboard access is unavailable, **Show
   key** selects the text for manual copying. **Done** clears it. You can
   retrieve the secret only during this handoff.
4. Inspect the key's expiry, last authentication and revocation in the key list.
   The access history records connection and key changes.

Keys expire after 30 days by default. Choose a lifetime of 1–365 days.
A connection can have up to 32 unexpired, unrevoked keys. If a key has no
expiry, issue a replacement and rotate it when the backend can switch over.

Capabilities narrow the existing owner/source/run restrictions:

| Capability | Permitted operations |
| --- | --- |
| `read` | Read authorized metadata, results and reports, including comparisons |
| `upload` | Validate/store an export and rename an authorized experiment |
| `analyze` | Submit an analysis or request cancellation of an authorized job |
| `delete` | Preview and confirm deletion of owned resources, inspect receipts and resume cleanup |

Give the backend `read`, `upload` and `analyze` to offer analysis, or only
`read` for a result viewer. Grant `delete` only if the backend offers resource
deletion. Integration credentials cannot train models, publish a model, manage
connections or sign in to the browser.

To rotate a key, issue its replacement on the same connection, update the backend
and send a GET request to the verification URL with the new key. Check the
returned source and scopes before revoking the old key by ID. If the issuance response was
lost, identify the new key in the list and revoke it before issuing another.

Pause the connection to block all its keys. Resuming restores access for valid
keys. Revocation and pausing leave saved results and already accepted jobs in
place. Requests authorized before the change may finish.

If another administrator edits the same connection, the page shows the saved
name and status and keeps the name you typed. Choose **Save name** to apply
your edit or **Use saved name** to discard it.

## Operator commands

The administrator token file must be owned by the current user and inaccessible
to other users. With the service environment configured:

```sh
uv run --locked --extra service regimex service connections \
  --token-file var/credentials/admin.token create \
  --owner simulator-backend --source simulator --name "Trading simulator"

uv run --locked --extra service regimex service connections \
  --token-file var/credentials/admin.token list
```

Use the returned `integration_id` in place of `CONNECTION_ID`:

```sh
uv run --locked --extra service regimex service connections \
  --token-file var/credentials/admin.token key-create CONNECTION_ID \
  --name "Simulator backend" --days 30 \
  --scope read --scope upload --scope analyze \
  --output var/credentials/integration.token

uv run --locked --extra service regimex service connections \
  --token-file var/credentials/admin.token keys CONNECTION_ID

uv run --locked --extra service regimex service connections \
  --token-file var/credentials/admin.token events CONNECTION_ID
```

Use a new output path. The CLI saves the secret with mode `0600` before the
database commits. If the CLI reports that it saved the file but could not confirm
issuance, keep the file and check `keys CONNECTION_ID` before retrying.
Use `key-revoke CONNECTION_ID TOKEN_ID` to revoke a key. Use
`update CONNECTION_ID --revision REVISION --disable` to pause a connection, or
`--enable` to resume it. Read its current revision from `list` before updating.
A stale revision returns a conflict instead of overwriting another operator's change.

Local `service token-create --role integration` provisions the same connection/key
records with `read`, `upload` and `analyze` and a 30-day lifetime. To choose
capabilities in the CLI, use `service connections key-create`. Omitting
`--scope` gives those same three defaults. Supplying any scopes replaces
the defaults, so repeat `--scope` for each capability the backend needs.

## Request and queue limits

Limits apply across processes and across every key of a connection:

| Environment setting | Default | Meaning |
| --- | --- | --- |
| `REGIMEX_INTEGRATION_REQUESTS_PER_MINUTE` | `600` | Requests admitted per connection in a 60-second window |
| `REGIMEX_INTEGRATION_SUBJECT_REQUESTS_PER_MINUTE` | `120` | Requests admitted per authenticated owner in a 60-second window |
| `REGIMEX_INTEGRATION_MAX_PENDING` | `32` | Queued, running or cancelling analyses across the connection |
| `REGIMEX_MAX_PENDING_PER_OWNER` | `8` | Pending operations allowed per owner before a new analysis is refused |

The first admitted request starts a 60-second window. A request can count even
if the subsequent resource lookup or job submission fails. All keys on the
connection share its quota. The subject's limit must fit within the connection limit.

If a request returns `429` with `rate_limit`, wait the number of seconds in
`Retry-After`. If the pending-job quota is full, wait for a job to finish or
cancel one. Retrying an accepted analysis with the same idempotency key still
works at the pending-job limit and counts toward the request limit.

For object-storage usage, the workspace shows the signed-in owner's usage and
**Connections** shows the shared connection total. Saved snapshots, reports
and reserved uploads count toward those limits. See [storage and
reservations](storage.md#owner-and-connection-quotas).

## History and diagnostics

Call `GET /api/v1/connection` with the simulator's integration bearer key and
no delegated headers. Compare the returned `analysis_protocol`, `source_namespace`
and `scopes` with the backend configuration. Any valid integration key can do this, even
without `read`. It counts against the connection's request budget and the
integration account's own minute window. A `401` can mean a wrong,
expired or revoked key, or a paused connection. Check the key and connection
in **Connections**. For worker and model readiness use the [deployment health
checks](deployment.md#health-shutdown-and-maintenance).

Connection creation, name/state changes and key issuance/revocation are recorded
in the same transaction as the change. History retains the latest 10,000 events
per connection, including the actor and the connection's name/state at that event.
It contains public identifiers and metadata, never key secrets or uploaded files.

Responses from the configured HTTP service include a server-generated `X-Request-ID`.
Operational logs record it with the matched route, status and duration. Authenticated
integration requests also carry public connection/key identifiers; analysis intake
records the operation identifier. Request bodies, raw URLs, query strings and
authentication headers are excluded. Management history records access changes;
HTTP logs diagnose requests and failures.

Pending resource deletion retains its original owner, source and run restrictions.
If a deletion key expires or is revoked, issue a replacement with `delete` on the
same connection to resume cleanup for the same delegated subject and run.
