"""The browser can fetch every packaged module through the real asset routes."""

from importlib.resources import files

import pytest

pytest.importorskip("fastapi", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from regimex.web import install_web_routes, report_runtime


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


def test_workspace_deep_links_serve_the_app_shell_only_on_known_routes() -> None:
    app = FastAPI()
    install_web_routes(app)
    resource_id = "12345678-1234-1234-1234-123456789abc"
    with TestClient(app) as client:
        home = client.get("/")
        policy = home.headers["content-security-policy"]
        for path in (
            "/training",
            "/training/new",
            f"/training/compare?left={resource_id}&right=00000000-0000-0000-0000-000000000001",
            f"/training/{resource_id}?view=archived",
            "/connections",
            f"/connections/{resource_id}",
        ):
            page = client.get(path, follow_redirects=False)
            assert page.status_code == 200, path
            assert page.content == home.content
            assert page.headers["content-security-policy"] == policy
        for path in ("/training/not-a-run", "/connections/new", "/missing", "/api/v1/missing"):
            assert client.get(path).status_code == 404, path


def test_plotting_asset_only_serves_an_installed_digest() -> None:
    app = FastAPI()
    install_web_routes(app)
    with TestClient(app) as client:
        assert client.get("/assets/plotly/unknown.js").status_code == 404
        runtime = report_runtime.plot_runtime()
        if runtime is not None:
            response = client.get(f"/assets/plotly/{runtime.digest}.js")
            assert response.status_code == 200 and response.content == runtime.script
            assert "immutable" in response.headers["cache-control"]
            assert response.headers["access-control-allow-origin"] == "*"


def test_preview_preserves_documents_with_other_or_unavailable_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = report_runtime.PlotRuntime(b"local", "digest", "integrity")
    monkeypatch.setattr(report_runtime, "plot_runtime", lambda: runtime)
    for document in (b"no chart", b"\n<script>older</script>", b"\n<script>local-extra</script>"):
        assert report_runtime.preview_html(document) is document
    document = b"\n<script>local</script>"
    assert b"/assets/plotly/digest.js" in report_runtime.preview_html(document)
    monkeypatch.setattr(report_runtime, "plot_runtime", lambda: None)
    assert report_runtime.preview_html(document) is document
