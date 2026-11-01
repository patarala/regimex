"""Observation checks and saved predictions for current immutable source snapshots."""

import math
from collections import Counter
from dataclasses import dataclass
from statistics import fmean, pstdev

from pydantic import JsonValue

from regimex.application.report import validate_analysis_result
from regimex.application.service import AnalysisResult
from regimex.contracts import Observation
from regimex.data_catalog import (
    DatasetManifest,
    SourceSnapshot,
    content_hash,
    create_snapshot_member,
)
from regimex.directions import DIRECTION_NAMES
from regimex.exports import ExportObservation, ExportV2

REPORT_VERSION = "6.0"
OBSERVABLE_FIELDS = tuple(
    name for name in Observation.model_fields if name not in {"step", "instrument_id"}
)


@dataclass(frozen=True)
class Series:
    experiment_id: str
    instrument_id: str
    initial_price: float
    base_asset: str
    quote_asset: str
    rows: tuple[ExportObservation, ...]
    regimes: tuple[str, ...]
    returns: tuple[float | None, ...]
    imbalance: tuple[float | None, ...]
    time_grid: dict[str, JsonValue]
    source: dict[str, JsonValue]
    analysis_config: dict[str, JsonValue]
    direction_intervals: tuple[dict[str, JsonValue], ...]
    prediction: dict[str, JsonValue]

    def payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "instrument_id": self.instrument_id,
            "initial_price": self.initial_price,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "source": self.source,
            "time_grid": self.time_grid,
            "analysis_config": self.analysis_config,
            "steps": [row.step for row in self.rows],
            "observations": {
                field: [getattr(row, field) for row in self.rows] for field in OBSERVABLE_FIELDS
            },
            "availability": {
                field: [getattr(row, field) for row in self.rows]
                for field in ("price_state", "bid_state", "ask_state")
            },
            "intervals": self.direction_intervals,
            "prediction": self.prediction,
        }


def _observed_values(
    rows: tuple[ExportObservation, ...],
) -> tuple[
    tuple[float | None, ...],
    tuple[float | None, ...],
]:
    returns: list[float | None] = []
    imbalance: list[float | None] = []
    previous: float | None = None
    for row in rows:
        change = None
        if previous is not None and row.last_price is not None:
            change = math.log(row.last_price) - math.log(previous)
        returns.append(change)
        previous = row.last_price
        buy, sell = row.buy_order_count, row.sell_order_count
        imbalance.append(
            None if buy is None or sell is None or buy + sell == 0 else (buy - sell) / (buy + sell)
        )
    return tuple(returns), tuple(imbalance)


def validate_series(series: Series) -> None:
    for row, change in zip(series.rows, series.returns, strict=True):
        if row.step == 0 and row.last_price is not None:
            change = math.log(row.last_price) - math.log(series.initial_price)
        if row.trade_count == 0 and change is not None and abs(change) > 1e-12:
            raise ValueError(
                f"{series.experiment_id}/{series.instrument_id}: observed price changed "
                f"at step {row.step} while the observed trade count was zero"
            )


def observed_series(export: ExportV2, prediction: AnalysisResult) -> tuple[Series, ...]:
    result_sha256 = content_hash(prediction.model_dump(mode="json"))
    export, prediction = validate_analysis_result(export, prediction, result_sha256=result_sha256)
    series = []
    for instrument, sequence in zip(export.instruments, prediction.sequences, strict=True):
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        intervals = sequence.intervals
        labels = tuple(
            interval.direction.value if interval.direction is not None else "unclassified"
            for interval in intervals
            for _ in range(interval.start, interval.stop)
        )
        returns, imbalance = _observed_values(rows)
        series.append(
            Series(
                experiment_id=export.source.snapshot_id,
                instrument_id=instrument.instrument_id,
                initial_price=instrument.initial_price,
                base_asset=instrument.base_asset,
                quote_asset=instrument.quote_asset,
                rows=rows,
                regimes=labels,
                returns=returns,
                imbalance=imbalance,
                time_grid=export.time_grid.model_dump(mode="json"),
                source=export.source.model_dump(mode="json"),
                analysis_config=prediction.analysis.model_dump(mode="json"),
                direction_intervals=tuple(
                    interval.model_dump(mode="json") for interval in intervals
                ),
                prediction={
                    "analysis_id": str(prediction.analysis_id),
                    "snapshot_id": str(prediction.snapshot_id),
                    "result_sha256": result_sha256,
                },
            )
        )
    return tuple(series)


def _model_identity(result: AnalysisResult) -> dict[str, JsonValue]:
    return {
        **result.model.model_dump(mode="json"),
        "architecture_id": result.architecture_id,
        "implementation_version": result.implementation_version,
        "decoder_version": result.decoder_version,
        "analysis_profile_id": result.analysis_profile_id,
        "analysis_config": result.analysis.model_dump(mode="json"),
        "inference_runtime": result.inference_runtime.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class Inspection:
    summary: dict[str, object]
    samples: tuple[Series, ...]


def _distribution(values: list[float]) -> dict[str, object]:
    if not values:
        return {"observed": 0, "mean": None, "std": None, "quantiles": None}
    ordered = sorted(values)
    quantiles = []
    for fraction in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        position = fraction * (len(ordered) - 1)
        left = math.floor(position)
        right = min(left + 1, len(ordered) - 1)
        quantiles.append(ordered[left] + (ordered[right] - ordered[left]) * (position - left))
    return {
        "observed": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "quantiles": quantiles,
    }


def _select_samples(names: list[str], count: int, requested: str | None) -> list[str]:
    if requested is not None:
        if requested not in names:
            raise ValueError("requested experiment identifier is not in the dataset")
        return [requested]
    return [names[round(i * (len(names) - 1) / max(count - 1, 1))] for i in range(count)]


def inspect_snapshots(
    snapshots: tuple[SourceSnapshot, ...],
    *,
    results: tuple[AnalysisResult, ...],
    dataset: DatasetManifest | None = None,
    sample_count: int = 12,
    experiment_id: str | None = None,
) -> Inspection:
    """Validate all saved predictions before selecting examples for display."""
    if not snapshots:
        raise ValueError("a report requires at least one source snapshot")
    if not 1 <= sample_count <= 100:
        raise ValueError("sample count must be between 1 and 100")
    checked = tuple(
        SourceSnapshot.model_validate_json(item.model_dump_json()) for item in snapshots
    )
    names = [item.export.source.snapshot_id for item in checked]
    if len(set(names)) != len(names):
        raise ValueError("report snapshot identifiers must be unique across input namespaces")
    if dataset is not None:
        dataset = DatasetManifest.model_validate_json(dataset.model_dump_json())
        if len(dataset.members) != len(checked) or any(
            create_snapshot_member(snapshot, member.partition) != member
            for snapshot, member in zip(checked, dataset.members, strict=True)
        ):
            raise ValueError("report snapshots differ from ordered dataset membership")
    predictions = tuple(
        AnalysisResult.model_validate_json(item.model_dump_json()) for item in results
    )
    by_hash = {item.snapshot_sha256: item for item in predictions}
    if (
        len(predictions) != len(checked)
        or len(by_hash) != len(predictions)
        or set(by_hash) != {source.export.content_sha256 for source in checked}
    ):
        raise ValueError("a report requires exactly one saved result per source snapshot")
    if any(
        len({getattr(item, field) for item in predictions}) != len(predictions)
        for field in ("analysis_id", "snapshot_id")
    ):
        raise ValueError("different report snapshots require distinct saved resource identities")
    model = _model_identity(predictions[0])
    model_sha256 = content_hash(model)
    if any(content_hash(_model_identity(item)) != model_sha256 for item in predictions):
        raise ValueError("report results must use the same immutable model, profile and runtime")
    groups = {}
    prediction_identities: list[dict[str, JsonValue]] = []
    for source in checked:
        prediction = by_hash[source.export.content_sha256]
        groups[source.export.source.snapshot_id] = observed_series(source.export, prediction)
        prediction_identities.append(
            {
                "snapshot_id": source.export.source.snapshot_id,
                "snapshot_sha256": source.export.content_sha256,
                "analysis_id": str(prediction.analysis_id),
                "result_sha256": content_hash(prediction.model_dump(mode="json")),
            }
        )
    totals: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    class_names = (*DIRECTION_NAMES, "unclassified")
    returns: dict[str, list[float]] = {name: [] for name in class_names}
    activity: dict[str, dict[str, list[float]]] = {
        name: {
            field: []
            for field in ("trade_count", "buy_order_count", "sell_order_count", "count_imbalance")
        }
        for name in class_names
    }
    lengths: dict[str, list[int]] = {name: [] for name in class_names}
    experiments: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    for name, series_group in groups.items():
        totals[f"experiments_with_{len(series_group)}_instruments"] += 1
        experiments.append(
            {
                "experiment_id": name,
                "instrument_count": len(series_group),
                "observations": sum(len(series.rows) for series in series_group),
                "observed_trade_total": sum(
                    row.trade_count or 0 for series in series_group for row in series.rows
                ),
                "classified_steps": sum(
                    label != "unclassified" for series in series_group for label in series.regimes
                ),
                "interval_count": sum(len(series.direction_intervals) for series in series_group),
            }
        )
        for series in series_group:
            totals["instrument_sequences"] += 1
            totals["observations"] += len(series.rows)
            totals["steps_without_executions"] += sum(row.trade_count == 0 for row in series.rows)
            validate_series(series)
            for i, (row, regime) in enumerate(zip(series.rows, series.regimes, strict=True)):
                regime_counts[regime] += 1
                for field in OBSERVABLE_FIELDS:
                    missing[field] += getattr(row, field) is None
                change = series.returns[i]
                if change is not None:
                    returns[regime].append(100 * change)
                for field in activity[regime]:
                    value = (
                        series.imbalance[i] if field == "count_imbalance" else getattr(row, field)
                    )
                    if value is not None:
                        activity[regime][field].append(float(value))
            for interval in series.direction_intervals:
                direction, status = interval["direction"], interval["status"]
                start, stop = interval["start"], interval["stop"]
                assert isinstance(start, int) and isinstance(stop, int)
                assert isinstance(status, str)
                assert direction is None or isinstance(direction, str)
                lengths[direction or "unclassified"].append(stop - start)
                status_counts[status] += stop - start
    selected = _select_samples(sorted(names), min(sample_count, len(names)), experiment_id)
    identity = [
        {
            "source_namespace": source.source_namespace,
            "snapshot_id": source.export.source.snapshot_id,
            "export_sha256": source.export.content_sha256,
        }
        for source in checked
    ]
    versions = sorted({source.export.producer.version for source in checked})
    summary: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "source_content_sha256": content_hash(identity),
        "dataset_sha256": dataset.content_sha256 if dataset else None,
        "sources": identity,
        "model": model,
        "results": prediction_identities,
        "generator_version": ", ".join(versions),
        "experiment_count": len(names),
        "totals": dict(totals),
        "regime_counts": dict(regime_counts),
        "class_names": list(class_names),
        "status_counts": dict(status_counts),
        "configuration_coverage": {
            "producers": sorted({source.export.producer.name for source in checked}),
            "source_versions": versions,
            "configuration_ids": sorted(
                {source.export.source.configuration_id for source in checked}
            ),
            "quote_capabilities": dict(
                Counter(source.export.capabilities.historical_quotes.mode for source in checked)
            ),
        },
        "experiments": experiments,
        "return_percent": {name: _distribution(values) for name, values in returns.items()},
        "activity": {
            name: {field: _distribution(values) for field, values in fields.items()}
            for name, fields in activity.items()
        },
        "segment_lengths": {
            name: _distribution([float(value) for value in values])
            for name, values in lengths.items()
        },
        "missing": {
            field: {"count": missing[field], "fraction": missing[field] / totals["observations"]}
            for field in OBSERVABLE_FIELDS
        },
        "sample_experiment_ids": selected,
        "selection": "explicit identifier"
        if experiment_id
        else "evenly spaced sorted snapshot identifiers, selected before examining directions",
    }
    return Inspection(summary, tuple(series for name in selected for series in groups[name]))
