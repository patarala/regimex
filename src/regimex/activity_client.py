"""Track an explicitly selected local process, with a receipt for offline replay."""

import asyncio
import fcntl
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from regimex.contracts import StrictModel
from regimex.service_client import connect, read_file, request, service_origin
from regimex.storage import write_new_file
from regimex.terminal_activity import ActivityFinish, ActivityStart, TerminalActivity

# A service may encode each non-BMP result character as two JSON Unicode escapes.
_ACTIVITY_RESPONSE_LIMIT = 256 * 1024


class ActivityReceipt(StrictModel):
    origin: str
    start: ActivityStart
    finish: ActivityFinish | None = None


def _save_finish(path: Path, receipt: ActivityReceipt, finish: ActivityFinish) -> ActivityReceipt:
    complete = receipt.model_copy(update={"finish": finish})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete_on_close=False) as tmp:
        tmp.write(complete.model_dump_json().encode())
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    return complete


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    # A sibling lock stays on the same inode when the receipt is atomically updated.
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("The local activity is still running or syncing.") from None
        yield


async def _sync(receipt: ActivityReceipt, token_file: Path, origin: str) -> str:
    # Select the credential destination from CLI configuration, never from an imported file.
    if receipt.origin != service_origin(origin):
        raise ValueError("The receipt belongs to a different service. Check --origin.")
    async with connect(origin, token_file, timeout=10) as client:
        start = receipt.start
        raw = await request(
            client,
            "terminal-activities",
            expected_status=201,
            limit=_ACTIVITY_RESPONSE_LIMIT,
            content=start.model_dump_json(exclude_none=True).encode(),
        )
        recorded = TerminalActivity.model_validate_json(raw)
        if (
            recorded.activity_id != start.activity_id
            or recorded.title != start.title
            or recorded.note != start.note
            or (start.started_at is not None and recorded.started_at != start.started_at)
        ):
            raise ValueError("The service returned a different activity start.")
        if receipt.finish is not None:
            raw = await request(
                client,
                f"terminal-activities/{start.activity_id}/finish",
                content=receipt.finish.model_dump_json(exclude_none=True).encode(),
                limit=_ACTIVITY_RESPONSE_LIMIT,
            )
            completed = TerminalActivity.model_validate_json(raw)
            if (
                completed.activity_id != start.activity_id
                or completed.status != receipt.finish.status
                or completed.exit_code != receipt.finish.exit_code
                or completed.result_text != receipt.finish.result_text
                or (
                    receipt.finish.finished_at is not None
                    and completed.finished_at != receipt.finish.finished_at
                )
            ):
                raise ValueError("The service returned a different activity outcome.")
        return str(client.base_url.join(f"/activity/{start.activity_id}"))


def sync_activity(
    path: Path,
    *,
    token_file: Path,
    origin: str,
    interrupted: bool,
) -> str:
    with _exclusive(path):
        try:
            receipt = ActivityReceipt.model_validate_json(read_file(path, 512 * 1024))
        except ValidationError:
            raise ValueError("Invalid activity receipt.") from None
        if receipt.origin != service_origin(origin):
            raise ValueError("The receipt belongs to a different service. Check --origin.")
        if interrupted and receipt.finish is None:
            receipt = _save_finish(
                path,
                receipt,
                ActivityFinish(
                    status="interrupted",
                    exit_code=None,
                    finished_at=datetime.now(UTC),
                ),
            )
        return asyncio.run(_sync(receipt, token_file, origin))


def _result_text(path: Path | None, token_file: Path, warn: Callable[[str], None]) -> str | None:
    if path is None:
        return None
    try:
        if token_file.exists() and path.samefile(token_file):
            raise ValueError("the bearer token file cannot be attached as a result")
        text = read_file(path, 64_000).decode("utf-8")
        if len(text) > 16_000 or "\0" in text:
            raise ValueError("expected UTF-8 text up to 16,000 characters without NUL bytes")
        return text
    except (OSError, ValueError) as exc:
        warn(f"Could not attach the result file: {exc}")
        return None


def run_activity(
    command: Sequence[str],
    *,
    title: str,
    note: str | None,
    origin: str,
    token_file: Path,
    receipt_dir: Path,
    result_file: Path | None,
    progress: Callable[[str], None],
    warn: Callable[[str], None],
) -> int:
    base = service_origin(origin)
    start = ActivityStart(activity_id=uuid4(), title=title, note=note, started_at=datetime.now(UTC))
    receipt = ActivityReceipt(origin=base, start=start)
    path = receipt_dir / f"{start.activity_id}.json"
    receipt_dir.mkdir(mode=0o700, exist_ok=True, parents=True)
    with _exclusive(path):
        write_new_file(path, receipt.model_dump_json().encode())
        progress(f"Activity receipt: {path}")

        def announce_start() -> None:
            try:
                url = asyncio.run(_sync(receipt, token_file, base))
            except (OSError, ValueError, RuntimeError) as exc:
                warn(f"Activity could not be synced at startup: {exc}")
            else:
                progress(f"Open in browser: {url}")

        # Persist the child's actual finish before waiting for a slow network response.
        with ThreadPoolExecutor(max_workers=1) as reporter:
            try:
                child = subprocess.Popen(command)
            except FileNotFoundError:
                warn("Local executable was not found.")
                code = 127
                result_file = None
            except OSError:
                warn("Could not start the local executable.")
                code = 126
                result_file = None
            else:
                with child:
                    try:
                        reporter.submit(announce_start)
                        code = child.wait()
                    except KeyboardInterrupt:
                        child.terminate()
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        code = -2
            finish = ActivityFinish(
                status="interrupted" if code < 0 else "succeeded" if code == 0 else "failed",
                exit_code=code,
                finished_at=datetime.now(UTC),
                result_text=_result_text(result_file, token_file, warn),
            )
            complete = _save_finish(path, receipt, finish)
        try:
            url = asyncio.run(_sync(complete, token_file, base))
        except (OSError, ValueError, RuntimeError) as exc:
            warn(f"Activity saved locally. Sync {path} when the service is available: {exc}")
        else:
            progress(f"Saved activity: {url}")
    return 128 - code if code < 0 else code
