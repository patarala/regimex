"""A missing or substituted training candidate must keep the automatic holdout labels sealed."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from regimex.storage import digest, write_json


def test_resumable_evidence_only_publishes_complete_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("compare_profile_21")
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        comparison.write(path, {"score": float("nan")})
    assert not path.exists()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(comparison.record, path, {"score": 0.9}) for _ in range(8)]
        for future in futures:
            future.result()
    assert path.read_bytes() == b'{"score":0.9}\n'
    with pytest.raises(ValueError, match="Previous evidence differs"):
        comparison.record(path, {"score": 0.5})
    assert path.read_bytes() == b'{"score":0.9}\n'


@pytest.mark.parametrize("substitute", [False, True], ids=["incomplete", "wrong-bundle"])
def test_all_candidates_must_be_frozen_before_accessing_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, substitute: bool
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    confirmation = importlib.import_module("confirm_training_populations")
    for directory in ("development", "confirmation", "fitted"):
        (tmp_path / directory).mkdir()
    plan = tmp_path / "development" / "plan.json"
    write_json(plan, {})
    receipt = {"development_plan_sha256": digest(plan)}
    reservation = tmp_path / "confirmation" / "reservation.json"
    write_json(reservation, receipt)
    candidates = {name: {"bundle_path": f"{name}/bundle"} for name in confirmation.ARMS}
    if substitute:
        candidates["short-96"]["bundle_path"] = "mixed-96/bundle"
    else:
        del candidates["mixed-96-enriched"]
    write_json(
        tmp_path / "fitted" / "freeze.json",
        {
            "reservation_sha256": digest(reservation),
            "development_plan_sha256": digest(plan),
            "candidates": candidates,
        },
    )

    monkeypatch.setattr(confirmation, "checked_reservation", lambda _: receipt)

    def opened(*args: object, **kwargs: object) -> None:
        pytest.fail("Opened the holdout before verifying every frozen model")

    monkeypatch.setattr(confirmation, "Corpus", opened)
    with pytest.raises(ValueError, match=r"Unexpected bundle location|must bind every candidate"):
        confirmation.score(tmp_path)
