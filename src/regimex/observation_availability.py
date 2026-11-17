"""Execution evidence shared by export validation and interval availability."""

from regimex.contracts import Observation


def known_no_executions(row: Observation) -> bool:
    """A carried price can establish a pause even when trade totals are missing."""
    return (
        row.trade_count == 0
        or row.trade_volume == 0
        or getattr(row, "price_state", None) in ("carried", "no_execution")
    )
