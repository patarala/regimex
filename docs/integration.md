# Integrating a simulator

To show classifications on a simulator's result page, have its backend send the
observations for a completed run to Regimex. Users can also upload the same JSON
file directly in the Regimex workspace.

## Export contract

Send a schema 2.1 export as `application/json`. A file can contain up to 16 MiB,
4,096 time bins and one or two instruments sharing a time grid and quote asset.

- Give each completed run and snapshot a stable identity. Include the source
  version, instrument units and precision.
- Export one observation per instrument per bin, in increasing step order
  within each instrument. Include leading and trailing inactivity and declare
  how the grid treats a partial final bin.
- Use zero for measured inactivity and `null` for unknown values. Record
  whether the last price came from an execution, was carried forward or is
  unavailable. Count each partial fill once in the execution totals. The buy
  and sell order totals count accepted submissions, including immediate fills,
  and exclude rejected orders.
- Export historical quotes from saved book observations or complete event
  history. For reconstructed quotes, include the reconstruction policy and
  event-history digest. If only the current book is available, mark historical
  quotes unavailable. An observed empty side has a null quote and state
  `empty`. A side whose history is unavailable has a null quote and state
  `unknown`.

The required `content_sha256` covers the validated payload with this field
omitted. Python adapters can use `seal_export` on an `ExportPayloadV2` from
`regimex.exports`. For other languages, follow the encoding rules in the
`content_sha256` field of the [export schema](../schemas/experiment-v2.schema.json).
Whitespace in the uploaded file has no effect on the digest.

## Delegated access

Create a connection for the simulator in **Connections** and issue a key with
`read`, `upload` and `analyze`. Store it on the simulator backend. For local
provisioning, this command creates a key with those capabilities for 30 days.

```sh
uv run --locked --all-extras regimex service token-create \
  --owner simulator-backend --role integration --source simulator \
  --output var/credentials/integration.token
```

At setup, call `GET /api/v1/connection` with the bearer key alone. The
`analysis_protocol` value `regimex-analysis/1` identifies the analysis intake
and verified delivery endpoints. Check this value before enabling those actions.
Check that `source_namespace` matches the export's source and `scopes` includes
the required permissions. See [platform access](access.md) for rotation
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
preserves that access. Browser session credentials, `Origin` and `Sec-Fetch-Site` are
rejected on delegated routes. Keep the simulator's own login credentials and
backend keys out of exports and browser links.

## Request sequence

1. Build the export from one consistent view of the completed run.
2. Send the export to `POST /api/v1/integration/analysis-intake` with an
   `Idempotency-Key`. This takes one request from the connection and subject
   budgets, and requires both `upload` and `analyze` scopes. A successful
   response contains the accepted `snapshot` and its `analysis`. Check their IDs,
   source, run and content hashes against the submitted export. The active model
   is resolved when the analysis is accepted.
3. Retry a lost response with the same key. To retry a failed or cancelled
   classification, send the same export with `?retry_of=<analysis-id>` and a
   new key for that attempt. A submission failure may leave its valid snapshot
   available for reuse. For an upload without analysis, use
   `POST /api/v1/integration/snapshots`. To choose a published model explicitly,
   submit the returned `snapshot_id` and the `model_version_id` to
   `POST /api/v1/integration/analyses` with an `Idempotency-Key`.
4. Poll `GET /api/v1/integration/analyses/{id}`. List and detail responses
   include `snapshot_sha256` so the simulator can check the binding in that
   request. Recheck simulator ownership before serving status or downloads.
5. After success, fetch the committed classification and its binding in one
   `GET /api/v1/integration/analyses/{id}/delivery`. Add `?include_report=true`
   to include the saved HTML. This consumes one request and requires the `read`
   scope. Construct the URL from the configured origin and the verified ID.

The delivery response is an `application/zip` attachment with uncompressed
members. `analysis.json` contains the scoped analysis detail (up to 128 KiB).
`result.json` contains the exact canonical bytes of the saved result (up to
16 MiB). `report.html` is present only when requested and is limited to 16 MiB.
Verify the analysis ID, snapshot ID, source, run and snapshot digest against the
selected run before serving the requested content. A pending analysis returns
`409 result_pending`, and requesting an absent report on a successful analysis
returns `404 report_unavailable`. An unknown analysis returns `404 not_found`.
The individual `/result` and `/report` routes are also available.

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
