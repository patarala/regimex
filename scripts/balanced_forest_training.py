"""Fit a forest with duration-balanced examples and reflected price movement."""

import numpy as np

from regimex.backends import adaptive_forest
from regimex.backends.adaptive_forest import DirectionContextModel, Floats, Ints

_SIGNED = tuple(
    index for index, name in enumerate(adaptive_forest.FEATURE_NAMES) if name.endswith("_movement")
)
_REFLECTED_CLASSES = np.array([0, 2, 1], dtype=np.int64)
_SEED = 173


def interval_weights(labels: Ints) -> Floats:
    """Measure each labeled run on one instrument's original grid."""
    if labels.ndim != 1 or labels.dtype != np.int64 or np.any((labels < -1) | (labels > 2)):
        raise ValueError("Expected one sequence of direction labels")
    if not len(labels):
        return np.empty(0, dtype=np.float64)
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    lengths = np.diff(edges)
    # A step in a 32-step interval counts twice as much as one in a 128-step
    # interval. The cap keeps a few short stretches from dominating training.
    weights = np.repeat(np.clip(np.sqrt(128 / lengths), 0.5, 2.0), lengths)
    weights[labels < 0] = 0.0
    return weights


def _sample(targets: Ints, weights: Floats) -> Ints:
    if (
        targets.ndim != 1
        or targets.dtype != np.int64
        or weights.shape != targets.shape
        or not np.isfinite(weights).all()
        or np.any((targets < -1) | (targets > 2))
        or np.any(weights[targets == -1] != 0)
        or np.any(weights[targets >= 0] <= 0)
    ):
        raise ValueError("Weights must be positive on classified rows and zero on unavailable rows")
    rng = np.random.default_rng(_SEED)
    samples = []
    for target in range(3):
        rows = np.flatnonzero(targets == target)
        if not len(rows):
            raise ValueError("Training requires all three directions")
        cumulative = weights[rows].cumsum()
        # One draw in each equal-mass stratum gives low-variance integer
        # multiplicities and preserves the total number of examples of this class.
        positions = (np.arange(len(rows)) + rng.random()) * (cumulative[-1] / len(rows))
        samples.append(rows[cumulative.searchsorted(positions, side="right")])
    return np.concatenate(samples)


def fit(x: Floats, targets: Ints, weights: Floats) -> DirectionContextModel:
    """Change only training evidence; inference uses the ordinary adaptive forest."""
    if x.shape != (len(targets), len(adaptive_forest.FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError("Expected aligned finite adaptive features")
    indices = _sample(targets, weights)
    count = len(indices)
    augmented = np.empty((2 * count, x.shape[1]), dtype=np.float64)
    np.take(x, indices, axis=0, out=augmented[:count])
    augmented[count:] = augmented[:count]
    augmented[count:, _SIGNED] *= -1
    selected = targets[indices]
    return adaptive_forest.fit(
        augmented, np.concatenate((selected, _REFLECTED_CLASSES[selected])), seed=_SEED
    )
