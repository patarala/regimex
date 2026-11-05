"""Missing or unsafe installed notices cannot produce a partially licensed report."""

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from regimex import report_licenses
from regimex.model_registry import RegistryError


@pytest.mark.parametrize("text", [None, " \n", "notice --> <script>", "notice --!>", "notice\x00"])
def test_missing_or_unsafe_notice_is_rejected(
    monkeypatch: pytest.MonkeyPatch, text: str | None
) -> None:
    monkeypatch.setattr(
        report_licenses, "distribution", lambda _name: SimpleNamespace(read_text=lambda _path: text)
    )
    with pytest.raises(RegistryError, match="license"):
        report_licenses.license_notices()


def test_absent_distribution_is_reported_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> None:
        raise PackageNotFoundError("regimex")

    monkeypatch.setattr(report_licenses, "distribution", missing)
    with pytest.raises(RegistryError, match=r"regimex.*license"):
        report_licenses.license_notices()


def test_requested_plotly_notice_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def metadata(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            read_text=lambda _path: "Project notice.\n" if name == "regimex" else None
        )

    monkeypatch.setattr(report_licenses, "distribution", metadata)
    with pytest.raises(RegistryError, match=r"plotly.*license"):
        report_licenses.license_notices(include_plotly=True)


def test_project_only_notices_do_not_access_plotly_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def metadata(name: str) -> SimpleNamespace:
        assert name == "regimex"
        return SimpleNamespace(read_text=lambda _path: "An unchanged project notice.\n")

    monkeypatch.setattr(report_licenses, "distribution", metadata)
    assert report_licenses.license_notices() == "# Regimex\n\nAn unchanged project notice.\n"
