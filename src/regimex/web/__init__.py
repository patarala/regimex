"""Bundled standalone upload page; authentication stays at the API boundary."""

import hashlib
import re
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

from regimex.web.report_runtime import plot_runtime

_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "frame-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
_HEADERS = {
    "Content-Security-Policy": _CSP,
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
}
_ASSET_NAME = re.compile(r"[a-z][a-z0-9_-]*\.(js|css)")
_MEDIA_TYPES = {"js": "text/javascript", "css": "text/css"}


def install_web_routes(app: FastAPI) -> None:
    @app.get("/examples/mixed-market.json", include_in_schema=False)
    def mixed_market_example() -> Response:
        return Response(
            files("regimex.web").joinpath("mixed-market.json").read_bytes(),
            media_type="application/json",
            headers=_HEADERS,
        )

    @app.get("/assets/plotly/{digest}.js", include_in_schema=False)
    def report_plotting_script(digest: str) -> Response:
        runtime = plot_runtime()
        if runtime is None or digest != runtime.digest:
            raise HTTPException(404, "Plotting runtime was not found.")
        return Response(
            runtime.script,
            media_type="text/javascript",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{runtime.digest}"',
                "X-Content-Type-Options": "nosniff",
                # The sandboxed preview has an opaque origin. This public file
                # contains only installed JavaScript and needs CORS for SRI.
                "Access-Control-Allow-Origin": "*",
            },
        )

    def standalone() -> HTMLResponse:
        return HTMLResponse(
            files("regimex.web").joinpath("index.html").read_bytes(), headers=_HEADERS
        )

    for path in (
        "/",
        "/training",
        "/training/new",
        "/training/compare",
        "/training/{run_id:uuid}",
        "/connections",
        "/connections/{integration_id:uuid}",
    ):
        app.add_api_route(path, standalone, methods=["GET"], include_in_schema=False)

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str) -> Response:
        match = _ASSET_NAME.fullmatch(name)
        if match is None:
            raise HTTPException(404, "Asset was not found.")
        resource = files("regimex.web").joinpath(name)
        if not resource.is_file() or (isinstance(resource, Path) and resource.is_symlink()):
            raise HTTPException(404, "Asset was not found.")
        try:
            body = resource.read_bytes()
        except FileNotFoundError, IsADirectoryError:
            # A release may replace the file between lookup and read.
            raise HTTPException(404, "Asset was not found.") from None
        return Response(
            body,
            media_type=_MEDIA_TYPES[match[1]],
            headers={**_HEADERS, "ETag": f'"{hashlib.sha256(body).hexdigest()}"'},
        )
