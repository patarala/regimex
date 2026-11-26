"""Observation feature layout and coefficient count for the logistic baseline."""

from typing import TYPE_CHECKING

from regimex.directions import DIRECTION_NAMES

if TYPE_CHECKING:
    from regimex.model_registry import LogisticParameters


FEATURE_VERSION = "3.1"
LEGACY_FEATURE_VERSION = "3.0"
NUMERIC_NAMES = (
    "relative_log_price",
    "log_return",
    "log_trade_count",
    "log_trade_volume",
    "log_buy_count",
    "log_buy_volume",
    "log_sell_count",
    "log_sell_volume",
    "count_imbalance",
    "volume_imbalance",
    "relative_spread",
    "absolute_return",
)
FEATURE_NAMES = (*NUMERIC_NAMES, *(f"{name}_missing" for name in NUMERIC_NAMES))


def trainable_parameter_count(parameters: LogisticParameters) -> int:
    features = parameters.summaries.width(len(FEATURE_NAMES), parameters.context_sizes)
    return len(DIRECTION_NAMES) * (features + 1)
