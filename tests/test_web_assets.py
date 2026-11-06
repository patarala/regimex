"""The browser can fetch every packaged module through the real asset routes."""

from importlib.resources import files

import pytest

pytest.importorskip("fastapi", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from regimex.web import install_web_routes


def test_packaged_browser_assets_are_available_without_authentication() -> None:
    app = FastAPI()
    install_web_routes(app)
    assets = {
        entry.name: entry
        for entry in files("regimex.web").iterdir()
        if entry.name.endswith((".js", ".css"))
    }
    assert "app.js" in assets and "app.css" in assets
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'src="/assets/app.js"' in page.text
        for name, resource in assets.items():
            response = client.get("/assets/" + name)
            assert response.status_code == 200, f"Browser asset is unavailable: {name}"
            media_type = "text/javascript" if name.endswith(".js") else "text/css"
            assert response.headers["content-type"].startswith(media_type)
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.content == resource.read_bytes()
        assert client.get("/assets/__init__.py").status_code == 404
        assert client.get("/assets/missing.js").status_code == 404
