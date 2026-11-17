# Integrating a simulator

To show classifications on a simulator's result page, have its backend send the
observations for a completed run to Regimex. Users can also upload the same JSON
file directly in the Regimex workspace.

## Export contract

Send a schema 2.1 export as `application/json`. A file can contain up to 16 MiB,
4,096 time bins and one or two instruments sharing a time grid and quote asset.

- Give each completed run and snapshot a stable identity. Include the source
  version, instrument units and precision.
- Export every original bin for every instrument, including leading and
  trailing inactivity. Declare how the grid treats a partial final bin.
- Use zero for measured inactivity and `null` for unknown values. Record
  whether the last price came from an execution, was carried forward or is
  unavailable. Count each partial fill once. Accepted submissions and rejections
  have separate fields.
- Export historical quotes from saved book observations or complete event
  history. For reconstructed quotes, include the reconstruction policy and
  event-history digest. If only the current book is available, mark historical
  quotes unavailable.

## Delegated access

Create a connection for the simulator in **Connections** and issue a key with
`read`, `upload` and `analyze`. Store it on the simulator backend. For local
provisioning, this command creates a key with those capabilities for 30 days.

```sh
uv run --locked --all-extras regimex service token-create \
  --owner simulator-backend --role integration --source simulator \
  --output var/credentials/integration.token
```

At setup, call `GET /api/v1/connection` with the bearer key alone. Check that
`source_namespace` matches the export's source and `scopes` includes the
capabilities the integration uses. See [platform access](access.md) for rotation
and rate limits.

Before forwarding each request, authenticate the user in the simulator and
check their access to the selected run. Set the following headers on the
server-to-server call.

| Header | Value |
| --- | --- |
| `Authorization: Bearer …` | The connection key |
| `X-Regimex-Subject` | The authenticated simulator user's stable identifier |
| `X-Regimex-Run-ID` | The run ID in the immutable export |

Subject and run IDs must match `[A-Za-z0-9._:@-]{1,128}`. Map other identifiers
to a stable representation. Regimex takes the source from the key and scopes
access to the connection, subject and run. Rotating keys on the same connection
preserves that access. Browser session cookies and Origin/Fetch headers are
rejected on delegated routes. Keep the simulator's own login credentials and
backend keys out of exports and browser links.

## Request sequence

1. Build the export from one consistent view of the completed run.
2. Send it to `POST /api/v1/integration/snapshots`. Check the source, run,
   source snapshot ID and content hash in the response against the export.
3. Send the returned snapshot ID and a model version to
   `POST /api/v1/integration/analyses`. The value `"active"` selects the
   deployment at acceptance. Set an `Idempotency-Key` for this analysis request
   and keep it when retrying after a timeout. A deliberate second analysis of
   the same snapshot uses a new key. A new simulation run gets a new run ID.
4. Poll `GET /api/v1/integration/analyses/{id}` using the returned analysis ID.
   Recheck simulator ownership and the snapshot binding before serving status or
   downloads. Use the job's state and failure information to update the page.
5. On success, proxy the saved `/result` JSON or `/report` HTML. Construct the
   endpoint from the configured service origin and the verified analysis ID.
   Keep the simulator results available if classification fails.

Report download returns `409` before success, `404` if no report was committed,
and `200` for the HTML attachment. Use verified HTTPS, reject redirects and
bound response size and duration.

## Contract tests

The example gateway in `tests/support/simulator_gateway/` shows the delegation
headers, response checks and download limits. Set `REGIMEX_TEST_DATABASE_DSN`
to a disposable PostgreSQL database, then run the gateway tests.

```sh
uv run --locked --all-extras --group queue-test --group service-test \
  pytest tests/test_simulator_gateway.py tests/test_simulator_gateway_integration.py
```

These tests use two local HTTP services and cover analysis, retries, restarts,
delegated ownership and key rotation. Test aggregation and access control in
the simulator adapter against that simulator's own runs.
