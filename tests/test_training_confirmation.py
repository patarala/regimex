"""A missing or substituted training candidate must keep the automatic holdout labels sealed."""

import importlib
from pathlib import Path

import pytest

from regimex.storage import digest, write_json


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
