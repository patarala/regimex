from collections import defaultdict
from importlib import import_module
from pathlib import Path

import pytest


def test_matched_paths_share_native_execution_times_and_carry_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    design = import_module("trend_pause_cases")
    grouped = defaultdict(list)
    for case in design.cases(172, 0):
        grouped[case.market].append(case)

    for paired in grouped.values():
        assert [len(case.expected) for case in paired] == [0, 0, 0, 1, 3]
        assert paired[-2].expected[0] == paired[-1].expected[1]
        for case in paired:
            assert 0 < case.active_steps < design.STEPS
            assert [r.trade_count for r in case.rows] == [r.trade_count for r in paired[0].rows]
            last = 100.0
            for row in case.rows:
                if row.trade_count:
                    assert row.price_state == "observed"
                    assert row.last_price / case.tick == pytest.approx(
                        round(row.last_price / case.tick)
                    )
                    last = row.last_price
                else:
                    assert row.price_state == "carried"
                    assert row.last_price == last


def test_selection_enforces_control_budget_before_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = import_module("compare_trend_pauses")
    high_recall, within_budget = comparison.Method("loose"), comparison.Method("strict")

    def counts(matched: int, false: int) -> dict[str, int]:
        return dict(
            matched=matched,
            slowdown_false=false,
            slowdown_steps=2000,
            extra_on_positive=0,
            steady_false=0,
        )

    assert (
        comparison.choose(
            [high_recall, within_budget], {"loose": counts(10, 2), "strict": counts(8, 1)}
        )
        == within_budget
    )
    assert (
        comparison.choose(
            [high_recall, within_budget], {"loose": counts(10, 3), "strict": counts(8, 2)}
        )
        == within_budget
    )
