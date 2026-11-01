"""Bundled standalone upload page; authentication stays at the API boundary."""

import hashlib
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

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
_ASSETS = {"app.js": "text/javascript", "app.css": "text/css"}


def install_web_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def standalone() -> HTMLResponse:
        return HTMLResponse(
            files("regimex.web").joinpath("index.html").read_bytes(), headers=_HEADERS
        )

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
