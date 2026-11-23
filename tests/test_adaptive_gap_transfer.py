"""Verify that masking measures the same price path and preserves both short-pause references."""

from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("check_adaptive_gap_transfer")


@pytest.mark.parametrize("index", (0, 3, 6, 9))
def test_paired_masks_preserve_actual_short_pauses_and_direction_schedules(
    replay: ModuleType, index: int
) -> None:
    import numpy as np

    pairs = list(replay.pairs(4317, index))
    assert len(pairs) == len(replay.SHAPES) * len(replay.MARKETS)
    assert len({(full.shape, full.market) for full, _, _ in pairs}) == len(pairs)
    for full, partial, (a, b) in pairs:
        assert 32 <= b - a <= 64
        assert full.rows[:a] == partial.rows[:a] and full.rows[b:] == partial.rows[b:]
        assert full.directions is partial.directions
        assert all(r.last_price is None and r.trade_count is None for r in partial.rows[a:b])
        masked = replay.references(partial)
        assert [(r.start, r.stop) for r in masked if r.status == "insufficient_data"] == [(a, b)]
        if full.shape == "reversal":
            edge = int(np.flatnonzero(np.diff(full.directions))[0] + 1)
            assert edge in (a, b)
        if full.shape in replay.SHORT_SHAPES:
            for case in (full, partial):
                holds = [r for r in replay.references(case) if r.direction == "sideways"]
                assert len(holds) == 1 and 32 <= holds[0].stop - holds[0].start <= 63
                assert b <= holds[0].start or holds[0].stop <= a
        if full.shape == "slow-continuation":
            assert len(set(full.directions)) == 1
            assert full.directions[0] != replay.DIRECTION_NAMES.index("sideways")


def test_churn_counts_each_common_step_once_at_its_nearest_gap_edge(replay: ModuleType) -> None:
    import numpy as np

    full = np.zeros(360, dtype=int)
    full[5:11] = -1
    masked = full.copy()
    masked[160:192] = -1
    masked[[0, 96, 128, 159, 192, 224, 320]] = 1
    buckets = replay.changed_steps(full, masked, (160, 192))
    assert [v[0] for v in buckets.values()] == [3, 2, 0, 2]
    assert sum(v[1] for v in buckets.values()) == 360 - 32 - 6


def test_transfer_requires_interval_quality_and_stability(replay: ModuleType) -> None:
    def result(score: float, sideways: float, changed: float) -> dict[str, object]:
        return {
            "interval": {
                "pooled": {
                    "macro_f1": score,
                    "per_class": {
                        key: {"f1": sideways if key == "sideways" else 0.9}
                        for key in replay.DIRECTION_NAMES
                    },
                }
            },
            "changed": {"pooled": {"fraction": changed}},
        }

    control = result(0.85, 0.80, 0.015)
    exact = replay.gate(control, result(0.83, 0.76, 0.020))
    assert exact["pooled"] and exact["stability"] and all(exact["per_class"].values())
    regressed = replay.gate(control, result(0.9, 0.70, 0.021))
    assert (
        regressed["pooled"]
        and not regressed["stability"]
        and not regressed["per_class"]["sideways"]
    )
