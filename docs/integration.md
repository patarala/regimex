# Integrating a simulator

Regimex accepts an immutable export from a completed run. File upload works
without a live simulator connection. A backend integration can add classification
to an existing result page while keeping the simulator's own results available.
Regimex does not read its database or change trading behavior.

## Export contract

Use observation schema 2.1. The adapter must preserve:

- One completed, stable configuration/run/snapshot revision and source version.
- Every original half-open time bin, including initial/trailing inactivity and
  the declared partial-terminal-bin policy.
- One or two instruments on a shared grid with units and precision, at most
  4,096 steps and 16 MiB; no silent truncation or resampling.
- Known zero activity versus unavailable values, initial/carried/execution prices,
  partial fills, accepted submissions and rejected orders as distinct concepts.
- Historical bid/ask only if retained or reconstructed from documented complete
  events. A current order book does not establish historical quote availability.

The repository's export/gateway fixtures verify this interface, not compatibility
or model accuracy on a particular simulator. Actual aggregation, ownership and
market behavior require tests using that simulator's real exports.

## Delegated access

The simulator backend authenticates its user and checks ownership of the selected
completed run. It alone holds a source-bound Regimex integration token:

```sh
uv run --locked --all-extras regimex service token-create \
  --owner simulator-backend --role integration --source simulator \
  --output var/credentials/integration.token
```

Provision the source namespace deliberately. Never send the token to a browser,
put it in an export, or embed it in a URL. Each backend request supplies:

| Header | Meaning |
| --- | --- |
| `Authorization: Bearer …` | Restricted integration credential |
| `X-Regimex-Subject` | Subject verified by the simulator authentication layer |
| `X-Regimex-Run-ID` | Selected immutable export run ID |

Subject and run IDs use `[A-Za-z0-9._:@-]{1,128}`. Other identifier formats need a
stable agreed encoding. The source comes from the credential, not a browser header.
The effective owner binds integration account, source and verified subject. Another
subject, run, account or source cannot read the resource. Same-account/source token
rotation preserves that binding.

Integration routes require backend bearer credentials and reject browser sessions
and Origin/Fetch headers. Ordinary routes reject integration subject/run assertions.
Integration credentials grant no training or publication rights. Regimex never
receives the simulator's JWT signing key. Arbitrary URL import is unsupported;
remote pull is disabled until a source has an explicitly verified delegation contract.

## Request sequence

1. Construct the complete export atomically from an owned completed run.
2. Send it to `POST /api/v1/integration/snapshots` and verify the returned
   source/run/snapshot/content-hash binding.
3. Send the stored snapshot ID and `model_version_id: "active"` to
   `POST /api/v1/integration/analyses`, with a deterministic `Idempotency-Key`
   bound to the authenticated subject, source, immutable run and export digest.
4. Poll `/api/v1/integration/analyses/{id}`. On every status/download request,
   recheck user/run ownership and the returned snapshot binding; do not rely on an
   in-memory analysis-to-run mapping.
5. Proxy the authorized `/result` JSON or `/report` HTML after success. Construct
   browser links locally from validated IDs; never fetch URLs returned by an upstream.

Queued/running work has no completed result. Report download returns `409` before
success, `404` if no report was committed, and `200` for a complete HTML attachment.
Retried submissions preserve the original resolved model. A rerun of the same
configuration requires a new run/snapshot identity.

Use verified HTTPS, bounded downloads and timeouts, and reject redirects.
Classification failures should leave simulator results visible and offer retry
when appropriate. Keep model predictions distinguishable from automatic references
or independent decisions. The supported reference gateway bounds reports to
16 MiB, uses 3/10-second connection/read timeouts and a 30-second total deadline.

## Contract tests

`tests/support/simulator_gateway/` implements a deliberately small owned-export
fixture. It is excluded from the runtime wheel and is not a production simulator
authentication implementation. Its source demonstrates the proxy sequence and
strict response checks.

Run it through the integration tests with a disposable PostgreSQL database:

```sh
uv run --locked --all-extras --group queue-test --group service-test \
  pytest tests/test_simulator_gateway.py tests/test_simulator_gateway_integration.py
```

Set `REGIMEX_TEST_DATABASE_DSN` to that database. The tests exercise two real TCP
services, saved-model inference, restart/idempotency, source/run/owner isolation,
token rotation and upstream failure. A real integration additionally needs its
own authentication, atomic export and result-page acceptance tests.
