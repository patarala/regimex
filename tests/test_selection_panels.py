"""Selection panels must spend the registered sample budget before opening the holdout."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from regimex.storage import digest, write_json


@pytest.fixture
def panels(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_selection_panels")


def test_panels_share_only_the_prespecified_short_strata(panels: ModuleType) -> None:
    members = [
        SimpleNamespace(
            step_count=steps,
            instrument_ids=tuple(range(1 + index % 2)),
            split_group=SimpleNamespace(seed=index),
            source_namespace=str(steps),
            snapshot_id=str(index),
        )
        for steps, size in zip(panels.LENGTHS, (24, 24, 16), strict=True)
        for index in reversed(range(size))
    ]
    a = set(panels.panel_ids(members, panels.PANELS["short"]))
    b = set(panels.panel_ids(members, panels.PANELS["balanced"]))
    assert len(a) == len(b) == 48
    assert len(a & b) == 32
    assert len(a | b) == 64
    for selected, wanted in ((a, (24, 24, 0)), (b, (16, 16, 16))):
        for steps, count in zip(panels.LENGTHS, wanted, strict=True):
            for instruments in (1, 2):
                assert (
                    sum(
                        panels.experiment_id(m) in selected
                        for m in members
                        if m.step_count == steps and len(m.instrument_ids) == instruments
                    )
                    == count // 2
                )


def test_fixed_ensemble_is_ineligible_and_choices_precede_holdout_access(
    panels: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scores = {name: {"interval_macro_f1": 0.5} for name in panels.ARMS}
    scores["ensemble"] = {"interval_macro_f1": 0.99}
    assert panels.ranking(scores) == list(panels.ARMS)
    reservation = tmp_path / "reservation.json"
    write_json(reservation, {})
    panel = {"scores": scores, "rank": list(panels.ARMS), "choice": "ensemble"}
    write_json(
        tmp_path / "selection.json",
        {
            "reservation_sha256": digest(reservation),
            "panels": {"short": panel, "balanced": panel},
        },
    )

    def opened(*args: object) -> None:
        pytest.fail("The holdout was opened before verifying both panel selections")

    monkeypatch.setattr(panels, "open_corpus", opened)
    with pytest.raises(ValueError, match="recorded validation ranking"):
        panels.score(tmp_path, tmp_path)
    assert not (tmp_path / "opened.json").exists()
