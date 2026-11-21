"""Feature and tensor layout shared by the TCN codec and architecture catalog."""

from math import prod
from typing import TYPE_CHECKING

from regimex.directions import DIRECTION_NAMES

if TYPE_CHECKING:
    from regimex.model_registry import TCNParameters


CONTEXT_SIZES = (32, 64, 128, 256, 512)
DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
TEMPORAL_NUMERIC_NAMES = (
    "relative_log_price_asinh",
    "log_return_asinh",
    "absolute_log_return_asinh",
    "sequence_standardized_log_price",
    "sequence_log_price_scale",
    *(
        f"context_{size}_{statistic}"
        for size in CONTEXT_SIZES
        for statistic in ("slope_change_asinh", "linear_fit", "endpoint_change_asinh", "coverage")
    ),
)
SIDE_WINDOW_SIZES = (16, 32)
SIDE_SLOPE_NAMES = tuple(
    f"{side}_{size}_slope_change_asinh" for size in SIDE_WINDOW_SIZES for side in ("past", "future")
)
TEMPORAL_NAMES = (
    *TEMPORAL_NUMERIC_NAMES,
    *(f"{name}_missing" for name in TEMPORAL_NUMERIC_NAMES),
)


def feature_names(side_slopes: bool) -> tuple[str, ...]:
    if not side_slopes:
        return TEMPORAL_NAMES
    numeric = TEMPORAL_NUMERIC_NAMES + SIDE_SLOPE_NAMES
    return (*numeric, *(f"{name}_missing" for name in numeric))


def state_shapes(parameters: TCNParameters) -> dict[str, tuple[int, ...]]:
    hidden = parameters.channels
    shapes = {
        "input.weight": (hidden, len(feature_names(parameters.side_slopes)), 1),
        "input.bias": (hidden,),
        "head.weight": (len(DIRECTION_NAMES), hidden, 1),
        "head.bias": (len(DIRECTION_NAMES),),
    }
    for index in range(len(parameters.dilations)):
        shapes.update(
            {
                f"blocks.{index}.conv.weight": (hidden, hidden, 3),
                f"blocks.{index}.conv.bias": (hidden,),
                f"blocks.{index}.mix.weight": (hidden, hidden, 1),
                f"blocks.{index}.mix.bias": (hidden,),
            }
        )
    return shapes


def trainable_parameter_count(parameters: TCNParameters) -> int:
    return sum(prod(shape) for shape in state_shapes(parameters).values())
