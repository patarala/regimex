"""Learn direction probabilities from locally fitted stretches of the price path."""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, observation_blocks

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
_PENALTIES = (4.0, 12.0, 36.0)
_PIECE_WIDTH = 10
_WIDTH = _PIECE_WIDTH * len(_PENALTIES)


def _errors(prefix: Floats, start: int | Ints, stop: int | Ints) -> Floats:
    """Residual sum of squares for separate intercepts and slopes on each span."""
    count, x, y, xx, xy, yy = (prefix[stop] - prefix[start]).T
    n = np.maximum(count, 1.0)
    spread = xx - x * x / n
    covariance = xy - x * y / n
    return np.asarray(np.maximum(0.0, yy - y * y / n - covariance**2 / np.maximum(spread, 1e-12)))


def _partition(prefix: Floats, size: int, width: int, penalty: float) -> list[int]:
    # Greedy binary splits give each observation a data-dependent context. A
    # refinement pass lets a split move after its neighboring piece is found.
    cuts = [0, size]
    pending = [(0, size)]
    while pending:
        left, right = pending.pop()
        options = np.arange(left + width, right - width + 1)
        if not len(options):
            continue
        residual = _errors(prefix, left, options) + _errors(prefix, options, right)
        choice = int(residual.argmin())
        if float(_errors(prefix, left, right)) - residual[choice] > penalty:
            cut = int(options[choice])
            cuts.append(cut)
            pending.extend(((left, cut), (cut, right)))
    cuts.sort()
    for index in range(1, len(cuts) - 1):
        left, right = cuts[index - 1], cuts[index + 1]
        options = np.arange(left + width, right - width + 1)
        residual = _errors(prefix, left, options) + _errors(prefix, options, right)
        cuts[index] = int(options[residual.argmin()])
    return cuts


def _line(x: Floats, y: Floats) -> tuple[float, float]:
    centered = y - y.mean()
    xy = float(x @ centered)
    xx = max(float(x @ x), 1e-12)
    movement = xy / xx
    fit = min(1.0, xy**2 / max(xx * float(centered @ centered), 1e-24))
    return movement, fit


def _describe(values: Floats, scale: float) -> tuple[float, ...]:
    present = np.isfinite(values)
    x = np.flatnonzero(present).astype(np.float64)
    y = values[present]
    length = len(values)
    if len(y) < 2:
        return (0.0,) * 6
    x = (x - x.mean()) / max(1, length - 1)
    movement, fit = _line(x, y)
    # One abrupt change in level can dominate a line. Removing its excess over
    # the median increment gives a second estimate of sustained movement.
    differences = np.diff(y)
    typical = float(np.median(differences))
    jump = int(np.abs(differences - typical).argmax())
    adjusted = y.copy()
    adjusted[jump + 1 :] -= differences[jump] - typical
    adjusted_move, adjusted_fit = _line(x, adjusted)
    edge = max(1, len(y) // 5)
    median_move = float(np.median(y[-edge:]) - np.median(y[:edge]))
    return (
        float(np.arcsinh(movement / scale)),
        fit,
        float(np.arcsinh(adjusted_move / scale)),
        adjusted_fit,
        float(np.arcsinh(median_move / scale)),
        float(np.log1p(length)),
    )


def features(rows: tuple[Observation, ...], config: AnalysisConfig) -> Floats:
    """Extract a row at each original step, keeping each available block independent."""
    result = np.zeros((len(rows), _WIDTH))
    for start, stop, status in observation_blocks(rows, config):
        if status != "observed":
            continue
        raw = np.array([row.last_price for row in rows[start:stop]], dtype=np.float64)
        present = np.isfinite(raw)
        log_price = np.log(raw)
        log_price -= log_price[present][0]
        size = len(raw)
        x = np.arange(size, dtype=np.float64) * present
        y = np.where(present, log_price, 0.0)
        moments = np.column_stack((present, x, y, x * x, x * y, y * y))
        prefix = np.vstack((np.zeros(6), moments.cumsum(axis=0)))
        diff = np.diff(log_price)
        diff = diff[np.isfinite(diff)]
        noise = config.noise_floor
        if len(diff):
            noise = max(noise, 1.4826 * float(np.median(np.abs(diff - np.median(diff)))))
        for level, multiplier in enumerate(_PENALTIES):
            cuts = _partition(
                prefix, size, config.min_duration, multiplier * np.log1p(size) * noise**2
            )
            pieces = [
                _describe(log_price[a:b], np.log1p(config.minimum_relative_change))
                for a, b in pairwise(cuts)
            ]
            first = level * _PIECE_WIDTH
            for i, (a, b) in enumerate(pairwise(cuts)):
                # A shallow piece between two sustained moves in the same direction
                # can belong to their trend. Let training decide from its neighbors.
                left = pieces[i - 1] if i else (0.0,) * 6
                right = pieces[i + 1] if i + 1 < len(pieces) else (0.0,) * 6
                result[start + a : start + b, first : first + _PIECE_WIDTH] = (
                    *pieces[i],
                    left[2],
                    left[3],
                    right[2],
                    right[3],
                )
    return result


@dataclass(frozen=True)
class DirectionContextModel:
    """An ensemble of shallow randomized trees stored as plain numeric arrays."""

    offsets: Ints
    children: Ints
    columns: Ints
    thresholds: Floats
    votes: Floats

    def predict(self, x: Floats) -> Floats:
        if x.ndim != 2 or x.shape[1] != _WIDTH or not np.isfinite(x).all():
            raise ValueError("Expected finite adaptive context features")
        # The fitted sklearn trees compare their thresholds with float32 inputs.
        values = x.astype(np.float32)
        probabilities = np.zeros((len(x), 3))
        for root in self.offsets:
            nodes = np.full(len(x), root, dtype=np.int64)
            while True:
                selected = np.flatnonzero(self.columns[nodes] >= 0)
                if not len(selected):
                    break
                current = nodes[selected]
                right = values[selected, self.columns[current]] > self.thresholds[current]
                nodes[selected] = self.children[current, right.astype(np.int64)]
            probabilities += self.votes[nodes]
        return probabilities / len(self.offsets)

    def save(self, path: Path) -> None:
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                offsets=self.offsets,
                children=self.children,
                columns=self.columns,
                thresholds=self.thresholds,
                votes=self.votes,
            )

    @classmethod
    def load(cls, path: Path) -> DirectionContextModel:
        with np.load(path, allow_pickle=False) as arrays:
            model = cls(
                **{
                    key: arrays[key]
                    for key in ("offsets", "children", "columns", "thresholds", "votes")
                }
            )
        n = len(model.columns)
        if (
            model.columns.shape != (n,)
            or model.offsets.ndim != 1
            or not len(model.offsets)
            or model.offsets[0] != 0
            or np.any(np.diff(model.offsets) <= 0)
            or model.offsets[-1] >= n
            or model.children.shape != (n, 2)
            or model.thresholds.shape != (n,)
            or model.votes.shape != (n, 3)
            or not np.isfinite(model.thresholds).all()
            or not np.isfinite(model.votes).all()
            or np.any(model.votes < 0)
            or not np.allclose(model.votes.sum(axis=1), 1)
            or np.any(model.columns >= _WIDTH)
            or any(a.dtype != np.int64 for a in (model.offsets, model.columns, model.children))
        ):
            raise ValueError("Invalid direction forest")
        for start, end in zip(model.offsets, [*model.offsets[1:], n], strict=True):
            nodes = np.arange(start, end)[model.columns[start:end] >= 0]
            branches = model.children[nodes]
            if np.any(branches <= nodes[:, None]) or np.any(branches >= end):
                raise ValueError("Invalid direction forest children")
        return model


def fit(x: Floats, targets: Ints) -> DirectionContextModel:
    if x.ndim != 2 or x.shape != (len(targets), _WIDTH) or not np.isfinite(x).all():
        raise ValueError("Expected aligned finite training rows and direction targets")
    labelled = targets != -1
    if not np.array_equal(np.unique(targets[labelled]), np.arange(3)):
        raise ValueError("Training requires all three directions")
    # Identical piece descriptions can cover hundreds of steps. Preserve their
    # target frequencies while fitting on the much smaller distinct table.
    table, weights = np.unique(
        np.column_stack((x[labelled], targets[labelled])), axis=0, return_counts=True
    )
    forest = ExtraTreesClassifier(
        n_estimators=40,
        max_depth=12,
        max_features=0.8,
        min_weight_fraction_leaf=0.0002,
        random_state=173,
        n_jobs=1,
    )
    forest.fit(table[:, :-1], table[:, -1].astype(np.int64), sample_weight=weights)
    offsets, edges, columns, thresholds, votes = [], [], [], [], []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        offsets.append(offset)
        edges.append(np.column_stack((tree.children_left, tree.children_right)) + offset)
        columns.append(tree.feature)
        thresholds.append(tree.threshold)
        values = tree.value[:, 0, :]
        votes.append(values / values.sum(axis=1, keepdims=True))
        offset += tree.node_count
    return DirectionContextModel(
        np.array(offsets),
        np.concatenate(edges),
        np.concatenate(columns),
        np.concatenate(thresholds),
        np.concatenate(votes),
    )
