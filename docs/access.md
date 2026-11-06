# Platform access

Users sign in to the simulator. Its backend checks their access to the selected
completed run, then calls Regimex with a service key and verified subject/run
identifiers. The key stays on that backend. Regimex does not receive user passwords
or the simulator's JWT signing key. See the [integration contract](integration.md).

An integration has a stable account and source namespace. Its keys can be replaced
without changing the owners of existing snapshots, analyses or reports. A simulator
administrator does not automatically become a Regimex model administrator.
Integration accounts, local user accounts and delegated owners use disjoint
identities. An administrator cannot claim an existing user's account as a connection.

## Manage connections in the browser

Sign in with a Regimex administrator credential and select **Connections**.
Ordinary users do not see this control; the server also rejects their requests.

1. Create a connection with a display name, integration account and source namespace.
   The account identifies the backend, not a person using the simulator.
2. Open the connection and issue a named key. Choose its capabilities and lifetime.
3. Copy the secret while it is displayed and store it in the simulator backend's
   private credential configuration. Regimex stores its hash; the secret cannot be
   retrieved later. Changing the selected connection or signing out clears it.
4. Inspect the key's expiry, last authentication and revocation in the key list.
   The access history records connection and key changes.

New keys expire after 30 days by default; supported lifetimes are 1–365 days.
A connection may have at most 32 unexpired, unrevoked keys. Existing keys retain
their previous expiry during migration. Rotate keys shown without an expiry when
the corresponding backend can be updated.

Migrations reject an existing database that assigns the same account identity to
local credentials and integrations or delegated users. The upgrade rolls back
without changing ownership. Resolve that ambiguity before upgrading; granting a
new integration access to an existing local user's data is not supported.

Capabilities narrow the existing owner/source/run restrictions:

| Capability | Permitted operations |
| --- | --- |
| `read` | Read authorized metadata, results and reports, including comparisons |
| `upload` | Validate/store an export and rename an authorized experiment |
| `analyze` | Submit an analysis or request cancellation of an authorized job |

The usual simulator gateway needs all three. A result reader needs only `read`.
No integration key permits browser login, training, model publication or connection
management.

To rotate a key, create its replacement on the same connection, update the backend,
verify an authorized request and revoke the old key. The overlap avoids an outage.
Revocation is permanent. Pausing a connection blocks authentication for all its keys;
resuming it restores only keys that have not expired or been revoked. These actions
do not delete results or cancel already accepted analyses. Requests already
authorized may finish; subsequent authentication and queued intake check the current
credential and connection state.

If a key-creation response is lost, inspect the key list and revoke the uncertain
key before issuing another. Creation does not return a previously issued secret.

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

The output file must not exist. It is created with mode `0600`; the secret is never
printed. Use `key-revoke CONNECTION_ID TOKEN_ID` to revoke a key. Use
`update CONNECTION_ID --revision REVISION --disable` to pause a connection, or
`--enable` to resume it. Read its current revision from `list` before updating.
A stale revision returns a conflict instead of overwriting another operator's change.

Local `service token-create --role integration` provisions the same connection/key
records with all three capabilities and a 30-day lifetime. Browser and API
management provide explicit names and narrower capabilities.

## Request and queue limits

Limits apply across processes and across every key of a connection:

| Environment setting | Default | Meaning |
| --- | --- | --- |
| `REGIMEX_INTEGRATION_REQUESTS_PER_MINUTE` | `600` | Accepted requests per connection in a 60-second window |
| `REGIMEX_INTEGRATION_SUBJECT_REQUESTS_PER_MINUTE` | `120` | Accepted requests per delegated owner in a 60-second window |
| `REGIMEX_INTEGRATION_MAX_PENDING` | `32` | Queued, running or cancelling analyses across the connection |
| `REGIMEX_MAX_PENDING_PER_OWNER` | `8` | Existing per-owner pending-operation bound |

Each window starts on its first accepted request. These are fixed windows, not a
rolling throughput guarantee. The subject request limit cannot exceed the connection
request limit. Rotating keys does not reset budgets. A delegated request rejected
by its scope or request limit does not register a new external subject.

Request-rate exhaustion returns `429` with code `rate_limit` and `Retry-After` in
seconds. Pending-operation exhaustion returns `capacity_limit`; retry after
existing work finishes or is cancelled. An idempotent replay reuses its analysis
even when the pending quota is full, but remains subject to request limits.
Snapshots and saved analysis HTML also have separate storage budgets: 1 GiB per
owner and 2 GiB per connection by default. Operators set
`REGIMEX_MAX_OWNER_STORAGE_BYTES` and `REGIMEX_MAX_INTEGRATION_STORAGE_BYTES` for
the API and workers together. All keys and delegated users of a connection share
its budget; replacing a key does not reset it. The global store capacity still
applies, including administrator-managed model and dataset assets.

The workspace shows the signed-in owner's usage. **Connections** shows the
selected platform's total. Both distinguish saved bytes from reservations for
uploads or cleanup after failed writes. Reservations continue to consume quota
until registration or verified offline cleanup; retries cannot silently discard
their cost. A quota refusal preserves existing results. See
[storage](storage.md) for cleanup and the scope of these limits.

## History and diagnostics

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

Revoking access on the simulator must take effect in its backend before forwarding
subsequent status or download requests. Regimex cannot independently discover changes
to the simulator's users or permissions.
