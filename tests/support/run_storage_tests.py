"""Run tests against a private, disposable native SeaweedFS 4.48 instance.

Download and verify the official archive explicitly before invoking this helper.
REGIMEX_TEST_SEAWEEDFS_BIN or --binary names its extracted Linux amd64 binary.
Use --tls for a temporary private CA, and -- followed by a command to run another
test selection. Without a command, the shared local/S3 contract suite runs.
No cloud credentials or existing provider data are used.
"""

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import platform
import secrets
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener
from xml.etree import ElementTree

ARCHIVE_URL = "https://github.com/seaweedfs/seaweedfs/releases/download/4.48/linux_amd64.tar.gz"
ARCHIVE_SHA256 = "4a7d108384d044d95212d1342cdda9533fa55842c1c9b41f606ca3c8a9561124"
BINARY_SHA256 = "8c07a1ccc4ec058cd90989ac0533c30436832e41c63de2b26f37253d9f744c4d"


def _private_file(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8") as output:
        path.chmod(0o600)
        output.write(value)


def _clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "S3_", "WEED_", "REGIMEX_TEST_S3_")) and key != "BOTO_CONFIG"
    }


def _ports(count: int) -> list[int]:
    held: list[socket.socket] = []
    try:
        for _ in range(count):
            candidate = socket.socket()
            held.append(candidate)
            candidate.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in held]
    finally:
        for candidate in held:
            candidate.close()


def _parent_death_signal(parent: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot set provider parent-death signal")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _assert_loopback(pid: int) -> None:
    inodes: set[str] = set()
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = entry.readlink().as_posix()
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    listeners = []
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in path.read_text().splitlines()[1:]:
            columns = line.split()
            if columns[3] == "0A" and columns[9] in inodes:
                listeners.append(columns[1])
    if not listeners or any(not address.startswith("0100007F:") for address in listeners):
        raise RuntimeError("test provider has an unexpected non-loopback listener")


def _signed_request(
    endpoint: str,
    bucket: str,
    access: str,
    secret: str,
    *,
    method: str,
    body: bytes | None,
    context: ssl.SSLContext | None,
) -> bytes:
    moment = datetime.now(UTC)
    timestamp, date = moment.strftime("%Y%m%dT%H%M%SZ"), moment.strftime("%Y%m%d")
    host = endpoint.split("://", 1)[1]
    payload_digest = hashlib.sha256(body or b"").hexdigest()
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{payload_digest}\nx-amz-date:{timestamp}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        (method, "/" + bucket, "versioning=", canonical_headers, signed_headers, payload_digest)
    )
    scope = date + "/us-east-1/s3/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            timestamp,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        )
    )
    signing_key = ("AWS4" + secret).encode()
    for piece in (date, "us-east-1", "s3", "aws4_request"):
        signing_key = hmac.digest(signing_key, piece.encode(), "sha256")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    request = Request(
        endpoint + "/" + bucket + "?versioning",
        method=method,
        data=body,
        headers={
            "Authorization": authorization,
            "Host": host,
            "x-amz-date": timestamp,
            "x-amz-content-sha256": payload_digest,
            "Content-Type": "application/xml",
        },
    )
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
    with opener.open(request, timeout=3) as response:
        payload = response.read(65536)
        if not isinstance(payload, bytes):
            raise RuntimeError("provider returned an invalid bootstrap response")
        return payload


def _ready(
    process: subprocess.Popen[bytes],
    endpoint: str,
    bucket: str,
    access: str,
    secret: str,
    certificate: Path | None,
) -> None:
    context = ssl.create_default_context(cafile=str(certificate)) if certificate else None
    configuration = (
        b'<VersioningConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<Status>Enabled</Status></VersioningConfiguration>"
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("test provider exited during startup")
        try:
            _signed_request(
                endpoint, bucket, access, secret, method="PUT", body=configuration, context=context
            )
            status = _signed_request(
                endpoint, bucket, access, secret, method="GET", body=None, context=context
            )
            root = ElementTree.fromstring(status)
            if not any(
                item.tag.rsplit("}", 1)[-1] == "Status" and item.text == "Enabled"
                for item in root.iter()
            ):
                raise RuntimeError("provider did not enable bucket versioning")
            break
        except HTTPError, URLError, TimeoutError, ConnectionError:
            time.sleep(0.2)
    else:
        raise RuntimeError("test provider startup exceeded 30 seconds")
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
    try:
        with opener.open(endpoint + "/" + bucket, timeout=3):
            raise RuntimeError("test provider accepted anonymous bucket access")
    except HTTPError as error:
        if error.code != 403:
            raise RuntimeError("test provider did not enforce private bucket access") from None
    _assert_loopback(process.pid)


@contextmanager
def provider(binary: Path, *, tls: bool) -> Iterator[dict[str, str]]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("the pinned native provider fixture requires Linux amd64")
    binary = binary.resolve(strict=True)
    with binary.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != BINARY_SHA256:
            raise ValueError("provider binary differs from the verified SeaweedFS 4.48 release")
    with tempfile.TemporaryDirectory(prefix="regimex-storage-tests-") as temporary:
        root = Path(temporary).resolve()
        root.chmod(0o700)
        configuration = root / "configuration"
        configuration.mkdir(mode=0o700)
        for name in ("security", "master", "volume", "admin"):
            _private_file(configuration / (name + ".toml"), "# Isolated test defaults.\n")
        _private_file(
            configuration / "filer.toml",
            "[leveldb2]\nenabled = true\ndir = " + json.dumps(str(root / "filer-db")) + "\n",
        )
        access, secret = secrets.token_hex(12), secrets.token_urlsafe(36)
        bucket = "regimex-test-" + secrets.token_hex(6)
        s3_configuration = root / "s3.json"
        _private_file(
            s3_configuration,
            json.dumps(
                {
                    "identities": [
                        {
                            "name": "regimex-tests",
                            "credentials": [{"accessKey": access, "secretKey": secret}],
                            "actions": ["Admin", "Read", "Write", "List", "Tagging"],
                        }
                    ]
                }
            ),
        )
        allocated = _ports(11)
        names = ("master", "volume", "filer", "s3", "admin")
        arguments = [
            str(binary),
            f"-config_dir={configuration}",
            "mini",
            f"-dir={root / 'data'}",
            "-ip=127.0.0.1",
            "-ip.bind=127.0.0.1",
            "-master.telemetry=false",
            "-master.volumeSizeLimitMB=64",
            "-volume.max=8",
            "-webdav=false",
            "-admin.ui=false",
            "-s3.iam=false",
            "-s3.port.iceberg=0",
            "-s3.port.lance=0",
            "-s3.autoCreateBucket=false",
            "-s3.allowDeleteBucketNotEmpty=false",
            f"-s3.config={s3_configuration}",
            f"-bucket={bucket}",
            f"-filer.localSocket={root / 'filer.sock'}",
            f"-s3.localSocket={root / 's3.sock'}",
        ]
        for index, name in enumerate(names):
            arguments.extend(
                (f"-{name}.port={allocated[index]}", f"-{name}.port.grpc={allocated[index + 5]}")
            )
        certificate: Path | None = None
        endpoint = f"http://127.0.0.1:{allocated[3]}"
        if tls:
            certificate, key = root / "ca.pem", root / "key.pem"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key),
                    "-out",
                    str(certificate),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=127.0.0.1",
                    "-addext",
                    "subjectAltName=IP:127.0.0.1",
                    "-addext",
                    "basicConstraints=critical,CA:TRUE",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            key.chmod(0o600)
            arguments.extend(
                (
                    f"-s3.port.https={allocated[10]}",
                    f"-s3.cert.file={certificate}",
                    f"-s3.key.file={key}",
                )
            )
            endpoint = f"https://127.0.0.1:{allocated[10]}"
        environment = _clean_environment()
        log_path = root / "provider.log"
        _private_file(log_path, "")
        parent = os.getpid()
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                arguments,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                preexec_fn=lambda: _parent_death_signal(parent),
            )
        try:
            _ready(process, endpoint, bucket, access, secret, certificate)
            environment.update(
                AWS_ACCESS_KEY_ID=access,
                AWS_SECRET_ACCESS_KEY=secret,
                AWS_DEFAULT_REGION="us-east-1",
                AWS_EC2_METADATA_DISABLED="true",
                REGIMEX_TEST_S3_ENDPOINT=endpoint,
                REGIMEX_TEST_S3_BUCKET=bucket,
                REGIMEX_TEST_S3_REGION="us-east-1",
            )
            empty_credentials = root / "empty-credentials"
            _private_file(empty_credentials, "")
            environment.update(
                AWS_CONFIG_FILE=str(empty_credentials),
                AWS_SHARED_CREDENTIALS_FILE=str(empty_credentials),
                NO_PROXY="127.0.0.1,localhost,::1",
                no_proxy="127.0.0.1,localhost,::1",
            )
            if certificate is not None:
                environment["REGIMEX_TEST_S3_CA_BUNDLE"] = str(certificate)
            print("Verified SeaweedFS 4.48 fixture:", endpoint, flush=True)
            yield environment
            _assert_loopback(process.pid)
        finally:
            _stop(process)


def _interrupt(signum: int, frame: FrameType | None) -> NoReturn:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=os.environ.get("REGIMEX_TEST_SEAWEEDFS_BIN"))
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.binary is None:
        parser.error("set REGIMEX_TEST_SEAWEEDFS_BIN or --binary to the verified native binary")
    command = options.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_object_store_contract.py",
            "tests/test_s3_object_store.py",
        ]
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        with provider(options.binary, tls=options.tls) as environment:
            process = subprocess.Popen(command, env=environment)
            try:
                return process.wait()
            finally:
                _stop(process)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
