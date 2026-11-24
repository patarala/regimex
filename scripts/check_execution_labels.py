"""Compare offline direction labels on original export observations."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.data_catalog import SnapshotGrid
from regimex.directions import AnalysisConfig, DirectionInterval, observation_blocks
from regimex.exports import load_export_v2
from regimex.observation_availability import known_no_executions
from regimex.segmentation import _directions, _interval, _log_prices
from regimex.storage import json_bytes, write_new_file

PROFILE = AnalysisConfig(version="2.2")
CANDIDATES = ("execution-median", "persistence", "combined")
ACTIVITY_CAPS = {"activity-20": 0.20, "activity-50": 0.50, "activity-80": 0.80}
METHODS = ("current", "median9", *CANDIDATES, *ACTIVITY_CAPS)
Floats = NDArray[np.float64]


def execution_prices(values: Floats, rows: tuple[Observation, ...]) -> Floats:
    # Repeated carried closes give a single execution many votes in a grid median.
    trades = np.array(
        [i for i, row in enumerate(rows) if np.isfinite(values[i]) and not known_no_executions(row)]
    )
    if len(trades) < 3:
        return values
    prices = values[trades].copy()
    windows = np.lib.stride_tricks.sliding_window_view(prices, 3)
    prices[1:-1] = np.median(windows, axis=1)
    known = np.flatnonzero(np.isfinite(values))
    result = values.copy()
    # Between executions the fit uses elapsed grid time. Missing observations stay missing.
    result[known] = np.interp(known, trades, prices)
    return result


def persistent_noise(values: Floats, config: AnalysisConfig) -> float:
    lag = max(1, config.min_duration // 4)
    contrast = values[2 * lag :] - 2 * values[lag:-lag] + values[: -2 * lag]
    finite = np.isfinite(contrast)
    if finite.sum() < config.min_duration:
        return config.noise_floor
    centered = contrast - np.median(contrast[finite])
    absolute = np.abs(centered[finite])
    # For independent Gaussian price noise a second difference has variance 6 sigma².
    scale = np.quantile(absolute, 0.75) / (1.15035 * np.sqrt(6.0))
    limit = np.quantile(absolute, 0.9)
    centered = np.clip(centered, -limit, limit)
    adjacent = finite[:-1] & finite[1:]
    left, right = centered[:-1][adjacent], centered[1:][adjacent]
    energy = np.sqrt((left @ left) * (right @ right))
    correlation = np.clip((left @ right) / energy, 0.0, 0.8) if energy > 0 else 0.0
    inflation = np.sqrt((1.0 + correlation) / (1.0 - correlation))
    return max(config.noise_floor, float(scale * inflation))


def label_sequence(
    rows: tuple[Observation, ...], method: str, config: AnalysisConfig = PROFILE
) -> tuple[DirectionInterval, ...]:
    """Compare candidate fits while retaining raw observation statistics and availability."""
    if method not in METHODS:
        raise ValueError(f"Unknown label method {method!r}")
    result = []
    for start, stop, status in observation_blocks(rows, config):
        if status != "observed":
            result.append(_interval(rows, start, stop, None, status))
            continue
        block = rows[start:stop]
        values = _log_prices(block, config)
        penalty = config
        if method in ACTIVITY_CAPS:
            known = np.isfinite(values)
            carried = sum(known[i] and known_no_executions(row) for i, row in enumerate(block))
            weight = min(1.0, carried / int(known.sum()) / ACTIVITY_CAPS[method])
            if weight:
                floor = config.noise_floor
                floor += weight * (persistent_noise(values, config) - floor)
                penalty = config.model_copy(update={"noise_floor": floor})
                values = values + weight * (execution_prices(values, block) - values)
        elif method in ("persistence", "combined"):
            penalty = config.model_copy(update={"noise_floor": persistent_noise(values, config)})
        if method in ("execution-median", "combined"):
            values = execution_prices(values, block)
        elif method == "median9":
            observed = np.flatnonzero(np.isfinite(values))
            prices = np.array([row.last_price for row in block], dtype=np.float64)
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(prices, 4, constant_values=np.nan), 9
            )
            medians = np.nanmedian(windows[observed], axis=1)
            staged = list(block)
            for index, price in zip(observed, medians, strict=True):
                staged[index] = block[index].model_copy(update={"last_price": float(price)})
            values = _log_prices(tuple(staged), config)
        for left, right, direction in _directions(values, penalty):
            result.append(_interval(rows, start + left, start + right, direction, status))
    return tuple(result)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("export", type=Path, help="Regimex export-v2 file")
    cli.add_argument("output", type=Path, help="New JSON file with the candidate intervals")
    cli.add_argument("--method", choices=METHODS, required=True)
    args = cli.parse_args()
    export = load_export_v2(args.export)
    grid = SnapshotGrid.from_export(export)
    intervals = []
    for instrument in grid.instruments:
        rows = tuple(
            row for row in grid.observations if row.instrument_id == instrument.instrument_id
        )
        intervals.extend(label_sequence(rows, args.method))
    report = {
        "method": args.method,
        "profile": PROFILE.model_dump(mode="json"),
        "export_sha256": export.content_sha256,
        "intervals": [i.model_dump(mode="json") for i in intervals],
    }
    write_new_file(args.output, json_bytes(report))


if __name__ == "__main__":
    main()
