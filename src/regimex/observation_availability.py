"""Execution evidence shared by export validation and interval availability."""

from regimex.contracts import Observation


def known_no_executions(row: Observation) -> bool:
    """Recognize observed inactivity without filling missing numeric aggregates.

    Plain observations expose count and volume; observation exports additionally
    declare whether the price was carried or the bin had no execution price.
    Unknown counts, volumes and prices alone establish neither activity nor a
    pause. The export contract rejects states that contradict known totals.
    """
    return (
        row.trade_count == 0
        or row.trade_volume == 0
        or getattr(row, "price_state", None) in ("carried", "no_execution")
    )
