"""Share the installed plotting runtime across sandboxed analysis previews."""

import base64
import hashlib
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class PlotRuntime:
    script: bytes
    digest: str
    integrity: str


@cache
def plot_runtime() -> PlotRuntime | None:
    # The API image contains just the static asset. Source installations with
    # the visualization extra can use the same file without importing Plotly.
    bundled = files("regimex.web").joinpath("vendor", "plotly.min.js")
    if bundled.is_file():
        script = bundled.read_bytes()
    else:
        try:
            installed = distribution("plotly")
        except PackageNotFoundError:
            return None
        try:
            script = installed.locate_file("plotly/package_data/plotly.min.js").read_bytes()
        except FileNotFoundError:
            return None
    digest = hashlib.sha256(script)
    integrity = "sha256-" + base64.b64encode(digest.digest()).decode("ascii")
    return PlotRuntime(script, digest.hexdigest(), integrity)


def preview_html(document: bytes) -> bytes:
    runtime = plot_runtime()
    if runtime is None:
        return document
    opening = b"\n<script>"
    start = document.find(opening)
    if start < 0:
        return document
    body = start + len(opening)
    end = body + len(runtime.script)
    # Keep a report with a different renderer/runtime entirely self-contained.
    if not document.startswith(runtime.script, body) or not document.startswith(b"</script>", end):
        return document
    tag = (
        f'\n<script src="/assets/plotly/{runtime.digest}.js" '
        f'integrity="{runtime.integrity}" crossorigin="anonymous">'
    ).encode("ascii")
    source = memoryview(document)
    return b"".join((source[:start], tag, source[end:]))
