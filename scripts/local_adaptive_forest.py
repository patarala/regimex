"""Learn direction from price-dependent pieces and short windows within each available block."""

import direction_context_model as windows
import numpy as np
from numpy.typing import NDArray

from regimex.backends import adaptive_forest as pieces
from regimex.backends.adaptive_forest import DirectionContextModel, fit
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig

_PIECE_NAMES = tuple(
    f"penalty_{penalty}_{measure}"
    for penalty in (4, 12, 36)
    for measure in (
        "movement",
        "fit",
        "adjusted_movement",
        "adjusted_fit",
        "log_length",
        "previous_adjusted_movement",
        "next_adjusted_movement",
    )
)
_WINDOW_NAMES = tuple(
    f"{scale}_{measure}"
    for scale in (1, 2, 4)
    for measure in (
        "center_within_level_movement",
        "center_within_level_correlation",
        "best_fit_within_level_movement",
    )
)
FEATURE_NAMES = (*_PIECE_NAMES, *_WINDOW_NAMES)
_PIECE_COLUMNS = [pieces.FEATURE_NAMES.index(name) for name in _PIECE_NAMES]
_WINDOW_COLUMNS = [windows.FEATURE_NAMES.index(name) for name in _WINDOW_NAMES]

__all__ = ["FEATURE_NAMES", "DirectionContextModel", "features", "fit"]


def features(rows: tuple[Observation, ...], config: AnalysisConfig) -> NDArray[np.float64]:
    # The window features can change inside a single fitted piece, allowing the
    # decoder to recover a short pause even when all three partitions missed it.
    return np.column_stack(
        (
            pieces.features(rows, config)[:, _PIECE_COLUMNS],
            windows.features(rows, config)[:, _WINDOW_COLUMNS],
        )
    )
