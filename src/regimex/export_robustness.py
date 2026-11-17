"""Paired direction and availability diagnostics under fixed incomplete-export scenarios."""

import numpy as np
from numpy.typing import NDArray

from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import scores
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.features import IntArray
from regimex.storage import json_bytes

WINDOW = (192, 256)
SCENARIOS = {
    "price_gap": "Hide the execution price and its state in the window. Keep execution totals.",
    "execution_gap": "Hide execution prices, their states, counts and volumes in the window.",
    "price_only": "Keep the price and its state. Hide all trade/order totals and quote history.",
}
NO_TRADES = len(DIRECTION_NAMES)
INSUFFICIENT = NO_TRADES + 1


def masked_export(export: ExportV2, scenario: str) -> ExportV2:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown missing-data scenario: {scenario}")
    payload = export.model_dump(mode="json", exclude={"content_sha256"})
    caps = payload["capabilities"]
    payload["source"]["snapshot_id"] = f"{export.source.snapshot_id}-{scenario}"
    if scenario == "price_only":
        caps.update(trade_aggregates="unavailable", accepted_order_aggregates="unavailable")
        caps["historical_quotes"].update(
            mode="unavailable",
            coverage="unavailable",
            provenance="This export retains only execution prices and their states.",
            complete_event_history_sha256=None,
            reconstruction_policy=None,
        )
    else:
        caps["execution_prices"] = "partial"
        if scenario == "execution_gap":
            caps["trade_aggregates"] = "partial"
    for row in payload["observations"]:
        if scenario == "price_only":
            for field in (
                "trade_count",
                "trade_volume",
                "buy_order_count",
                "buy_order_volume",
                "sell_order_count",
                "sell_order_volume",
                "best_bid",
                "best_ask",
            ):
                row[field] = None
            row.update(bid_state="unknown", ask_state="unknown")
        elif WINDOW[0] <= row["step"] < WINDOW[1]:
            row.update(last_price=None, price_state="unknown")
            if scenario == "execution_gap":
                row.update(trade_count=None, trade_volume=None)
    return seal_export(ExportPayloadV2.model_validate_json(json_bytes(payload)))


def statuses(values: IntArray) -> dict[str, int]:
    return {
        "observed": int((values < NO_TRADES).sum()),
        "no_trades": int((values == NO_TRADES).sum()),
        "insufficient_data": int((values == INSUFFICIENT).sum()),
    }


def paired_metrics(
    reference: IntArray,
    pristine: IntArray,
    masked: IntArray,
    window: NDArray[np.bool_],
    zero: NDArray[np.bool_],
) -> dict[str, object]:
    """Compare direction on identical eligible points and count coverage on the full grid."""
    pristine_direction = pristine < NO_TRADES
    masked_direction = masked < NO_TRADES
    common = (reference < NO_TRADES) & pristine_direction & masked_direction
    changed = pristine != masked
    groups = {
        "window": window,
        "window_original_zero": window & zero,
        "window_original_no_trades": window & (pristine == NO_TRADES),
    }
    return {
        "grid_points": len(reference),
        "pristine_status": statuses(pristine),
        "masked_status": statuses(masked),
        "lost_coverage": int((pristine_direction & ~masked_direction).sum()),
        "gained_coverage": int((~pristine_direction & masked_direction).sum()),
        "common_points": int(common.sum()),
        "changed_direction": int((common & changed).sum()),
        "common_outside_window": int((common & ~window).sum()),
        "changed_outside_window": int((common & ~window & changed).sum()),
        "pristine_macro_f1": scores(reference[common], pristine[common])["macro_f1"]
        if common.any()
        else None,
        "masked_macro_f1": scores(reference[common], masked[common])["macro_f1"]
        if common.any()
        else None,
        "groups": {
            name: {"pristine": statuses(pristine[group]), "masked": statuses(masked[group])}
            for name, group in groups.items()
        },
    }
