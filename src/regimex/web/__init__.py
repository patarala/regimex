"""Bundled standalone upload page; authentication stays at the API boundary."""

import hashlib
from importlib.resources import files

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
_ASSETS = {
    "app.css": "text/css",
    "app.js": "text/javascript",
    "theme.js": "text/javascript",
    "session.js": "text/javascript",
    "history.js": "text/javascript",
    "experiment-history.js": "text/javascript",
    "models.js": "text/javascript",
    "analysis.js": "text/javascript",
    "uploads.js": "text/javascript",
    "comparison.js": "text/javascript",
    "queue-status.js": "text/javascript",
    "ui.js": "text/javascript",
    "integrations.js": "text/javascript",
    "training.js": "text/javascript",
    "training-archive.js": "text/javascript",
    "training-evaluation.js": "text/javascript",
    "training-form.js": "text/javascript",
    "training-json.js": "text/javascript",
    "training-metrics.js": "text/javascript",
    "training-name.js": "text/javascript",
    "training-policy.js": "text/javascript",
    "training-progress.js": "text/javascript",
    "training-size.js": "text/javascript",
    "storage.js": "text/javascript",
    "deletion.js": "text/javascript",
}


def install_web_routes(app: FastAPI) -> None:
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
        "/training/{run_id:uuid}",
        "/connections",
        "/connections/{integration_id:uuid}",
    ):
        app.add_api_route(path, standalone, methods=["GET"], include_in_schema=False)

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str) -> Response:
        media_type = _ASSETS.get(name)
        if media_type is None:
            raise HTTPException(404, "Asset was not found.")
        body = files("regimex.web").joinpath(name).read_bytes()
        return Response(
            body,
            media_type=media_type,
            headers={**_HEADERS, "ETag": f'"{hashlib.sha256(body).hexdigest()}"'},
        )
