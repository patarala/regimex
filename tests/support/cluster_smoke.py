"""Exercise two Regimex API nodes with a test owner and an already published model."""

import argparse
import hashlib
import json
import socket
import ssl
import time
from http.client import HTTPSConnection
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from regimex.api import AnalysisView, OperationView
from regimex.application.service import AnalysisResult
from regimex.exports import parse_export_v2


class Client:
    """Keep the public origin and TLS name when dialing an individual replica."""

    def __init__(self, origin: str, ca: Path | None) -> None:
        self.origin = origin.rstrip("/")
        parsed = urlsplit(self.origin)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Specify the public HTTPS origin, including its port if needed")
        self.host = parsed.hostname
        self.port = parsed.port or 443
        self.context = ssl.create_default_context(cafile=ca)
        self.cookie = ""
        self.csrf = ""

    def request(
        self,
        node: tuple[str, int],
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        expected: int = 200,
        idempotency_key: str | None = None,
        csrf: bool = True,
    ) -> bytes:
        headers = {"Origin": self.origin}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.cookie:
            headers["Cookie"] = self.cookie
        if csrf and self.csrf and method != "GET":
            headers["X-CSRF-Token"] = self.csrf
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        connection = HTTPSConnection(self.host, self.port, timeout=30, context=self.context)
        with socket.create_connection(node, timeout=30) as raw:
            try:
                connection.sock = self.context.wrap_socket(raw, server_hostname=self.host)
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                if response.status != expected:
                    raise RuntimeError(
                        f"{method} {path} returned {response.status}, expected {expected}"
                    )
                if path == "/api/v1/session" and method == "POST":
                    cookies = SimpleCookie()
                    cookies.load(response.getheader("Set-Cookie", ""))
                    if len(cookies) != 1:
                        raise RuntimeError("Sign-in did not issue one session cookie")
                    self.cookie = cookies.output(header="", attrs=[]).strip()
                return response.read()
            finally:
                connection.close()


def address(value: str) -> tuple[str, int]:
    addr = urlsplit("//" + value)
    if (
        addr.hostname is None
        or addr.username is not None
        or addr.port is None
        or addr.path
        or addr.query
        or addr.fragment
    ):
        raise argparse.ArgumentTypeError("Use a replica address with its HTTPS port")
    return addr.hostname, addr.port


def verify(
    client: Client,
    first: tuple[str, int],
    second: tuple[str, int],
    token: str,
    export: bytes,
    model: str,
    timeout: int,
) -> str:
    if first == second:
        raise ValueError("Supply two distinct replica addresses")
    source = parse_export_v2(export)
    session_url = "/api/v1/session"
    login = client.request(
        first, session_url, method="POST", body=json.dumps({"token": token}).encode()
    )
    client.csrf = json.loads(login)["csrf_token"]
    try:
        # A rejected mutation must leave this session usable on both replicas.
        client.request(second, session_url, method="DELETE", expected=403, csrf=False)
        restored = json.loads(client.request(second, session_url))
        if restored["csrf_token"] != client.csrf:
            raise RuntimeError("The second replica did not restore the session")
        snapshot = json.loads(
            client.request(first, "/api/v1/snapshots", method="POST", body=export, expected=201)
        )
        body = json.dumps(
            {"snapshot_id": snapshot["snapshot_id"], "model_version_id": model}
        ).encode()
        key = f"cluster-smoke-{uuid4()}"

        def submit(node: tuple[str, int], key: str) -> AnalysisView:
            raw = client.request(
                node,
                "/api/v1/analyses",
                method="POST",
                body=body,
                idempotency_key=key,
                expected=202,
            )
            return AnalysisView.model_validate_json(raw)

        accepted = submit(second, key)
        analysis_id = str(accepted.analysis_id)
        if submit(first, key).analysis_id != accepted.analysis_id:
            raise RuntimeError("The two nodes accepted different jobs for the same key")
        url = f"/api/v1/analyses/{analysis_id}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = json.loads(client.request(first, url))["job"]["state"]
            if state == "succeeded":
                break
            if state not in {"queued", "running"}:
                raise RuntimeError(f"Analysis {analysis_id} ended in {state}")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"Analysis {analysis_id} did not complete in {timeout}s")
        result = client.request(second, url + "/result")
        parsed = AnalysisResult.model_validate_json(result)
        if (
            str(parsed.analysis_id) != analysis_id
            or str(parsed.snapshot_id) != snapshot["snapshot_id"]
            or parsed.snapshot_sha256 != source.content_sha256
            or parsed.time_grid.step_count != source.time_grid.step_count
            or parsed.model != accepted.job.model
        ):
            raise RuntimeError("The result does not describe the submitted export")
        if result != client.request(first, url + "/result"):
            raise RuntimeError("The nodes returned different result bytes")
        reports = [client.request(node, url + "/report") for node in (first, second)]
        if not reports[0].startswith(b"<!doctype html>") or reports[0] != reports[1]:
            raise RuntimeError("The nodes did not return the same complete HTML report")
        extra = submit(first, f"cluster-cancel-{uuid4()}")
        cancel_url = f"/api/v1/jobs/{extra.job.operation_id}/cancel"
        response = OperationView.model_validate_json(
            client.request(second, cancel_url, method="POST")
        )
        cancelled_url = f"/api/v1/analyses/{extra.analysis_id}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cancelled = json.loads(client.request(first, cancelled_url))
            if cancelled["job"]["state"] == "cancelled" and not cancelled["result_available"]:
                break
            # A fast worker may have already committed before the cancel request.
            if response.state == "succeeded" and cancelled["result_available"]:
                break
            if cancelled["job"]["state"] != "cancel_requested":
                raise RuntimeError("The cancelled analysis produced an unexpected state")
            time.sleep(0.5)
        else:
            raise TimeoutError("The worker did not acknowledge cancellation")
        print(
            json.dumps(
                {
                    "analysis_id": analysis_id,
                    "model": parsed.model.model_version_id,
                    "cancellation": cancelled["job"]["state"],
                    "report_sha256": hashlib.sha256(reports[0]).hexdigest(),
                }
            )
        )
    finally:
        client.request(second, session_url, method="DELETE", expected=204)
    client.request(first, session_url, expected=401)
    return analysis_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument(
        "--nodes", type=address, nargs=2, required=True, metavar=("FIRST", "SECOND")
    )
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--model", default="active")
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    verify(
        Client(args.origin, args.ca_file),
        args.nodes[0],
        args.nodes[1],
        args.token_file.read_text().strip(),
        args.export.read_bytes(),
        args.model,
        args.timeout,
    )


if __name__ == "__main__":
    main()
