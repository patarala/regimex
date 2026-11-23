"""Learn direction probabilities from locally fitted stretches of the price path."""

import math
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import MAX_STEPS, Observation
from regimex.directions import AnalysisConfig, observation_blocks
from regimex.resource_limits import ResourceBudgetExceeded

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
_PENALTIES = (4.0, 12.0, 36.0)
_MEASURES = (
    "movement",
    "fit",
    "adjusted_movement",
    "adjusted_fit",
    "median_movement",
    "log_length",
    "previous_adjusted_movement",
    "previous_adjusted_fit",
    "next_adjusted_movement",
    "next_adjusted_fit",
)
FEATURE_NAMES = tuple(f"penalty_{penalty:g}_{name}" for penalty in _PENALTIES for name in _MEASURES)
_PIECE_WIDTH = len(_MEASURES)
_WIDTH = len(FEATURE_NAMES)
_MAX_TREES = 128
_MAX_DEPTH = 16
_MIN_LEAF_FRACTION = 0.0002


def max_nodes_per_tree(max_depth: int) -> int:
    # Each leaf needs a fixed fraction of the total sample weight. A binary tree
    # with L leaves has 2L - 1 nodes, even if the depth limit would allow more.
    return min((1 << (max_depth + 1)) - 1, 2 * int(1 / _MIN_LEAF_FRACTION) - 1)


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


def _scaled_movement(value: float, scale: float) -> float:
    ratio = float(value) / float(scale)
    if math.isfinite(ratio):
        return float(np.arcsinh(ratio))
    # The ratio overflowed, but asinh is still finite and rounds to log(2 * |ratio|).
    return math.copysign(math.log(2) + math.log(abs(value)) - math.log(scale), value)


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
        _scaled_movement(movement, scale),
        fit,
        _scaled_movement(adjusted_move, scale),
        adjusted_fit,
        _scaled_movement(median_move, scale),
        float(np.log1p(length)),
    )


type PartitionRule = Callable[[Floats, int, AnalysisConfig, float], tuple[list[int], ...]]


def _greedy_partitions(
    prefix: Floats, start: int, config: AnalysisConfig, noise_variance: float
) -> tuple[list[int], ...]:
    size = len(prefix) - 1
    return tuple(
        _partition(
            prefix, size, config.min_duration, float(multiplier * np.log1p(size)) * noise_variance
        )
        for multiplier in _PENALTIES
    )


def price_blocks(
    rows: tuple[Observation, ...], config: AnalysisConfig
) -> Iterator[tuple[int, Floats, Floats, float]]:
    """Prepare log prices, line-fit moments and noise from each available block."""
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
        yield start, log_price, prefix, noise


def features(
    rows: tuple[Observation, ...],
    config: AnalysisConfig,
    *,
    partition: PartitionRule = _greedy_partitions,
) -> Floats:
    """Extract a row at each original step, keeping each available block independent."""
    result = np.zeros((len(rows), _WIDTH))
    scale = float(np.log1p(config.minimum_relative_change))
    for start, log_price, prefix, noise in price_blocks(rows, config):
        try:
            variance = noise**2
        except OverflowError:
            variance = math.inf  # No finite reduction in residual error can justify a split.
        for level, cuts in enumerate(partition(prefix, start, config, variance)):
            pieces = [_describe(log_price[a:b], scale) for a, b in pairwise(cuts)]
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

    def __post_init__(self) -> None:
        for array in (self.offsets, self.children, self.columns, self.thresholds, self.votes):
            array.setflags(write=False)

    @property
    def numeric_bytes(self) -> int:
        return sum(
            a.nbytes
            for a in (self.offsets, self.children, self.columns, self.thresholds, self.votes)
        )

    def predict(self, x: Floats) -> Floats:
        return next(self.predict_prefixes(x, (len(self.offsets),)))

    def predict_prefixes(self, x: Floats, tree_counts: tuple[int, ...]) -> Iterator[Floats]:
        """Average successive tree votes without traversing a tree twice."""
        if (
            not tree_counts
            or tree_counts[0] < 1
            or tree_counts[-1] > len(self.offsets)
            or any(a >= b for a, b in pairwise(tree_counts))
        ):
            raise ValueError("Tree counts must be increasing prefixes of this forest")
        if x.ndim != 2 or x.shape[1] != _WIDTH or not np.isfinite(x).all():
            raise ValueError("Expected finite adaptive context features")
        if not len(x):
            for _ in tree_counts:
                yield np.empty((0, 3))
            return
        # The fitted sklearn trees compare their thresholds with float32 inputs.
        values = x.astype(np.float32)
        changes = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
        lengths = None
        if len(changes) < len(values) - 1:
            # A fitted piece repeats the same feature row until one of the partitions changes.
            starts = np.r_[0, changes]
            lengths = np.diff(starts, append=len(values))
            values = values[starts]
        probabilities = np.zeros((len(values), 3))
        # When only a few different pieces remain, traverse multiple trees together.
        # Once the input fills a whole instrument's worth of node slots, use one tree per pass.
        trees_per_pass = max(1, MAX_STEPS // len(values))
        visited = 0
        for count in tree_counts:
            for first in range(visited, count, trees_per_pass):
                roots = self.offsets[first : min(first + trees_per_pass, count)]
                nodes = np.repeat(roots, len(values))
                while True:
                    selected = np.flatnonzero(self.columns[nodes] >= 0)
                    if not len(selected):
                        break
                    current = nodes[selected]
                    rows = selected if len(roots) == 1 else selected % len(values)
                    right = values[rows, self.columns[current]] > self.thresholds[current]
                    nodes[selected] = self.children[current, right.astype(np.int64)]
                # Preserve floating-point accumulation order across tree batches.
                for leaves in nodes.reshape(len(roots), len(values)):
                    probabilities += self.votes[leaves]
            visited = count
            average = probabilities / count
            yield np.repeat(average, lengths, axis=0) if lengths is not None else average

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
    def load(
        cls,
        path: Path,
        *,
        trees: int | None = None,
        max_depth: int = _MAX_DEPTH,
        max_bytes: int = 128 * 1024**2,
    ) -> DirectionContextModel:
        if not 1 <= max_depth <= _MAX_DEPTH or (trees is not None and not 1 <= trees <= _MAX_TREES):
            raise ValueError("Unsupported direction forest dimensions")
        max_nodes = (trees or _MAX_TREES) * max_nodes_per_tree(max_depth)
        layout = {
            "offsets": ((), np.int64),
            "children": ((2,), np.int64),
            "columns": ((), np.int64),
            "thresholds": ((), np.float64),
            "votes": ((3,), np.float64),
        }
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) != len(layout) or {m.filename for m in members} != {
                    name + ".npy" for name in layout
                }:
                    raise ValueError("Unexpected direction forest archive members")
                arrays = {}
                sizes = {}
                total = 0
                for member in members:
                    name = member.filename[:-4]
                    trailing, expected_dtype = layout[name]
                    if (
                        member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or member.flag_bits & 1
                    ):
                        raise ValueError("Unsupported direction forest archive encoding")
                    with archive.open(member) as stream:
                        if np.lib.format.read_magic(stream) != (1, 0):
                            raise ValueError("Expected direction forest NPY format 1.0")
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            stream, max_header_size=4096
                        )
                        limit = _MAX_TREES if name == "offsets" else max_nodes
                        if (
                            not shape
                            or not 1 <= shape[0] <= limit
                            or shape[1:] != trailing
                            or fortran
                            or dtype != expected_dtype
                        ):
                            raise ValueError("Invalid direction forest dimensions or dtype")
                        size = math.prod(shape) * dtype.itemsize
                        if size + stream.tell() != member.file_size:
                            raise ValueError("Invalid direction forest array size")
                        total += size
                        if total > max_bytes:
                            raise ResourceBudgetExceeded(
                                "Direction forest weights exceed the inference memory budget"
                            )
                        sizes[name] = (shape, dtype, stream.tell(), size)
                # All allocation sizes are known before decompressing any array payload.
                for name, (shape, dtype, header_bytes, size) in sizes.items():
                    with archive.open(name + ".npy") as stream:
                        stream.read(header_bytes)
                        payload = stream.read(size + 1)
                        if len(payload) != size:
                            raise ValueError("Incomplete direction forest array")
                        arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape)
        except (OSError, zipfile.BadZipFile, EOFError) as exc:
            raise ValueError("Invalid direction forest archive") from exc
        model = cls(**arrays)
        model.validate(trees=trees, max_depth=max_depth)
        return model

    def validate(self, *, trees: int | None = None, max_depth: int = _MAX_DEPTH) -> None:
        n = len(self.columns)
        if (
            self.columns.shape != (n,)
            or self.offsets.ndim != 1
            or not 1 <= len(self.offsets) <= _MAX_TREES
            or (trees is not None and len(self.offsets) != trees)
            or self.offsets[0] != 0
            or np.any(np.diff(self.offsets) <= 0)
            or self.offsets[-1] >= n
            or self.children.shape != (n, 2)
            or self.thresholds.shape != (n,)
            or self.votes.shape != (n, 3)
            or any(a.dtype != np.int64 for a in (self.offsets, self.columns, self.children))
            or self.thresholds.dtype != np.float64
            or self.votes.dtype != np.float64
            or not np.isfinite(self.thresholds).all()
            or not np.isfinite(self.votes).all()
            or np.any(self.votes < 0)
            or not np.allclose(self.votes.sum(axis=1), 1, rtol=0, atol=1e-12)
            or np.any((self.columns < 0) & (self.columns != -2))
            or np.any(self.columns >= _WIDTH)
        ):
            raise ValueError("Invalid direction forest")
        for start, end in zip(self.offsets, [*self.offsets[1:], n], strict=True):
            if end - start > max_nodes_per_tree(max_depth):
                raise ValueError("Direction forest exceeds the configured tree depth or leaf limit")
            nodes = np.arange(start, end)
            split = self.columns[start:end] >= 0
            branches = self.children[nodes[split]]
            if (
                np.any(branches <= nodes[split, None])
                or np.any(branches >= end)
                or np.any(self.children[nodes[~split]] != start - 1)
            ):
                raise ValueError("Invalid direction forest children")
            parents = np.bincount(branches.ravel() - start, minlength=end - start)
            if np.any(parents[1:] != 1):
                raise ValueError("Each direction forest node must have one parent")
            level = np.array([start])
            for _ in range(max_depth):
                level = self.children[level[self.columns[level] >= 0]].ravel()
                if not len(level):
                    break
            if np.any(self.columns[level] >= 0):
                raise ValueError("Direction forest exceeds the configured tree depth")


@dataclass(frozen=True)
class ForestTrainingRows:
    table: Floats
    counts: Ints
    interval_weights: Floats


def training_rows(x: Floats, targets: Ints) -> ForestTrainingRows:
    """Collapse repeated feature/target pairs within one instrument, preserving both weights."""
    if x.ndim != 2 or x.shape != (len(targets), _WIDTH) or not np.isfinite(x).all():
        raise ValueError("Expected aligned finite training rows and direction targets")
    if targets.dtype != np.int64 or np.any((targets < -1) | (targets > 2)):
        raise ValueError("Expected original-grid direction targets or -1 for unavailable steps")
    if not len(targets):
        return ForestTrainingRows(np.empty((0, _WIDTH + 1)), targets, np.empty(0))
    boundaries = targets[1:] != targets[:-1]
    label_starts = np.r_[0, np.flatnonzero(boundaries) + 1]
    lengths = np.diff(label_starts, append=len(targets))
    point_weights = np.repeat(np.clip(np.sqrt(128 / lengths), 0.5, 2.0), lengths)
    changes = boundaries | np.any(x[1:] != x[:-1], axis=1)
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    counts = np.diff(starts, append=len(targets))
    keep = targets[starts] >= 0
    selected = starts[keep]
    return ForestTrainingRows(
        np.column_stack((x[selected], targets[selected])),
        counts[keep],
        point_weights[selected] * counts[keep],
    )


def fit_training_rows(
    rows: list[ForestTrainingRows],
    *,
    interval_weighting: bool,
    trees: int,
    max_depth: int,
    seed: int,
) -> DirectionContextModel:
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

    table = np.concatenate([row.table for row in rows])
    counts = np.concatenate([row.counts for row in rows])
    labels = table[:, -1].astype(np.int64)
    if not np.array_equal(np.unique(labels), np.arange(3)):
        raise ValueError("Training requires all three directions")
    if interval_weighting:
        weights = np.concatenate([row.interval_weights for row in rows])
        # Retain the original class frequencies while giving shorter intervals more weight.
        normalization = np.bincount(labels, weights=counts) / np.bincount(labels, weights=weights)
        weights *= normalization[labels]
    else:
        weights = counts.astype(np.float64)
    table, groups = np.unique(table, axis=0, return_inverse=True)
    weights = np.bincount(groups, weights=weights)
    forest = ExtraTreesClassifier(
        n_estimators=trees,
        max_depth=max_depth,
        max_features=0.8,
        min_weight_fraction_leaf=_MIN_LEAF_FRACTION,
        random_state=seed,
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


def fit(
    x: Floats, targets: Ints, *, trees: int = 40, max_depth: int = 12, seed: int = 173
) -> DirectionContextModel:
    return fit_training_rows(
        [training_rows(x, targets)],
        interval_weighting=False,
        trees=trees,
        max_depth=max_depth,
        seed=seed,
    )
