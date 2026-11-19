"""Current immutable snapshots, complete diagnostics, and blind offline review."""

import json
from importlib.metadata import distribution
from pathlib import Path

import pytest
from support.report_fixtures import prediction_for

from regimex.application.service import AnalysisResult
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    SourceSnapshot,
    SplitGroup,
    create_snapshot_member,
    seal_dataset,
)
from regimex.directions import AnalysisConfig, Direction, DirectionAnnotations
from regimex.exports import ExportPayloadV2, seal_export
from regimex.inspection import inspect_snapshots
from regimex.storage import digest


def source(
    index: int, *, instrument_count: int = 1, missing: bool = False, active: bool = False
) -> SourceSnapshot:
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    value.pop("content_sha256")
    value["source"].update(snapshot_id=f"snapshot-{index}", run_id=f"run-{index}")
    value["time_grid"].update(step_count=8, completed_horizon=104.0)
    template = value["observations"][0]
    initial = float(100 + index)
    value["instruments"] = [
        dict(
            value["instruments"][0],
            instrument_id=f"{name}/USD",
            base_asset=name,
            quote_asset="USD",
            initial_price=initial,
        )
        for name in ("BTC", "ETH")[:instrument_count]
    ]
    value["observations"] = [
        dict(
            template,
            instrument_id=instrument["instrument_id"],
            step=step,
            last_price=None if missing else initial,
            price_state="unknown" if missing else "observed" if active else "carried",
            trade_count=int(active),
            trade_volume=float(active),
        )
        for instrument in value["instruments"]
        for step in range(8)
    ]
    return SourceSnapshot(
        source_namespace="test-source",
        split_group=SplitGroup(namespace="test-seeds", group_id=f"seed-{index}", seed=index),
        export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
    )


def dataset() -> tuple[DatasetManifest, tuple[SourceSnapshot, ...]]:
    sources = tuple(source(index, instrument_count=1 + index % 2) for index in range(3))
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    return seal_dataset(
        "reports",
        tuple(
            create_snapshot_member(item, partition)
            for item, partition in zip(sources, partitions, strict=True)
        ),
    ), sources


def predictions(sources: tuple[SourceSnapshot, ...]) -> tuple[AnalysisResult, ...]:
    return tuple(prediction_for(item.export) for item in sources)


def report_data(output: Path) -> dict[str, object]:
    html = (output / "index.html").read_text()
    result = json.loads(
        html.split('<script type="application/json" id="report-data">')[1].split("</script>")[0]
    )
    assert isinstance(result, dict)
    return result


def test_inspection_covers_every_snapshot_and_all_instruments_in_selected_examples() -> None:
    manifest, sources = dataset()
    inspection = inspect_snapshots(
        sources, results=predictions(sources), dataset=manifest, sample_count=1
    )
    assert inspection.summary["experiment_count"] == 3
    assert inspection.summary["dataset_sha256"] == manifest.content_sha256
    assert inspection.summary["totals"] == {
        "experiments_with_1_instruments": 2,
        "experiments_with_2_instruments": 1,
        "instrument_sequences": 4,
        "observations": 32,
        "steps_without_executions": 32,
    }
    selected = inspect_snapshots(
        sources, results=predictions(sources), dataset=manifest, experiment_id="snapshot-1"
    )
    assert {sample.instrument_id for sample in selected.samples} == {"BTC/USD", "ETH/USD"}
    assert all("instrument_label" not in sample.payload() for sample in selected.samples)
    assert all("issues" not in sample.payload() for sample in selected.samples)
    assert "issues" not in inspection.summary
    assert all(sample.regimes == ("unclassified",) * 8 for sample in inspection.samples)
    assert inspection == inspect_snapshots(
        sources, results=predictions(sources), dataset=manifest, sample_count=1
    )


def test_report_rejects_changed_order_and_content_before_selecting_examples() -> None:
    manifest, sources = dataset()
    with pytest.raises(ValueError, match="ordered dataset membership"):
        inspect_snapshots(
            tuple(reversed(sources)), results=predictions(sources), dataset=manifest, sample_count=1
        )
    tampered = sources[-1].export.model_copy(update={"content_sha256": "0" * 64})
    changed = (*sources[:-1], sources[-1].model_copy(update={"export": tampered}))
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_snapshots(changed, results=predictions(sources), sample_count=1)
    with pytest.raises(ValueError, match="identifier"):
        inspect_snapshots(sources, results=predictions(sources), experiment_id="absent")


def test_unknown_values_remain_distinct_from_recorded_zero() -> None:
    sources = (source(0, missing=True),)
    inspection = inspect_snapshots(sources, results=predictions(sources))
    sample = inspection.samples[0]
    assert sample.returns == (None,) * 8
    payload = sample.payload()
    observations = payload["observations"]
    assert isinstance(observations, dict)
    assert observations["last_price"] == [None] * 8
    assert observations["trade_count"] == [0] * 8
    missing = inspection.summary["missing"]
    assert isinstance(missing, dict)
    assert missing["last_price"]["fraction"] == 1.0
    assert missing["trade_count"]["fraction"] == 0.0


def test_steps_without_executions_use_inactivity_evidence_when_counts_are_missing() -> None:
    snapshot = source(0, missing=True)
    value = snapshot.export.model_dump(mode="json", exclude={"content_sha256"})
    for row in value["observations"]:
        row.update(trade_count=None, trade_volume=0.0 if row["step"] < 4 else None)
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    inspection = inspect_snapshots(
        (snapshot.model_copy(update={"export": export}),), results=(prediction_for(export),)
    )
    totals = inspection.summary["totals"]
    assert isinstance(totals, dict)
    assert totals["steps_without_executions"] == 4
    assert inspection.summary["status_counts"] == {"no_trades": 4, "insufficient_data": 4}
    assert all(row.trade_count is None for row in inspection.samples[0].rows)


def test_same_external_snapshot_id_keeps_sources_and_results_separate() -> None:
    first, other = source(0), source(1, active=True)
    payload = other.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["source"].update(snapshot_id=first.export.source.snapshot_id, source_version="engine-3")
    payload["producer"]["version"] = "adapter-1"
    second = other.model_copy(
        update={
            "source_namespace": "second-installation",
            "export": seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload))),
        }
    )
    sources = (first, second)
    result = inspect_snapshots(sources, results=predictions(sources))
    digests = {item.export.content_sha256 for item in sources}
    assert {item.experiment_id for item in result.samples} == digests
    coverage = result.summary["configuration_coverage"]
    assert isinstance(coverage, dict)
    assert set(coverage["source_versions"]) == {first.export.source.source_version, "engine-3"}
    assert result.samples[0].source["snapshot_id"] == result.samples[1].source["snapshot_id"]
    selected = inspect_snapshots(
        sources, results=predictions(sources), experiment_id=second.export.content_sha256
    )
    assert selected.samples[0].experiment_id == second.export.content_sha256
    with pytest.raises(ValueError, match="Ambiguous snapshot identifier"):
        inspect_snapshots(
            sources, results=predictions(sources), experiment_id=first.export.source.snapshot_id
        )


def test_extreme_finite_observations_preserve_intervals_and_serializable_statistics() -> None:
    item = source(0, active=True)
    payload = item.export.model_dump(mode="json", exclude={"content_sha256"})
    for row in payload["observations"]:
        row.update(last_price=1.0 if row["step"] == 0 else 1e308, trade_volume=1e308)
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
    result = prediction_for(export)
    stats = result.sequences[0].intervals[0].statistics
    assert (stats.first_price_step, stats.last_price_step) == (0, 7)
    assert stats.price_coverage == stats.trade_volume_coverage == 1
    assert stats.trade_volume is None and stats.price_change_percent is None
    assert stats.return_std is not None and stats.trade_count == 8
    updated = item.model_copy(update={"export": export})
    report = inspect_snapshots((updated,), results=(result,))
    assert report.samples[0].direction_intervals[0]["statistics"] == stats.model_dump(mode="json")
    from regimex.application.report import _interval_table

    assert _interval_table(result).count("out of range") == 2


@pytest.mark.parametrize(("known", "expected_total"), [(0, None), (3, 0), (8, 0)])
def test_trade_totals_report_counter_coverage(known: int, expected_total: int | None) -> None:
    item = source(0)
    payload = item.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["capabilities"]["trade_aggregates"] = "partial"
    for row in payload["observations"][known:]:
        row.update(trade_count=None, trade_volume=None)
    updated = item.model_copy(
        update={"export": seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))}
    )
    report = inspect_snapshots((updated,), results=predictions((updated,)))
    experiments = report.summary["experiments"]
    assert isinstance(experiments, list)
    assert experiments[0]["observed_trade_total"] == expected_total
    assert experiments[0]["steps_with_trade_count"] == known


def test_report_preserves_saved_directions_instead_of_annotating_prices() -> None:
    sources = (source(0, active=True),)
    result = prediction_for(sources[0].export, direction=Direction.DECLINE)
    inspection = inspect_snapshots(sources, results=(result,))
    # These explicit model predictions describe constant prices differently from
    # the statistical annotator. Rendering must preserve the supplied decisions.
    sample = inspection.samples[0]
    assert sample.regimes == ("decline",) * 8
    assert sample.direction_intervals == tuple(
        interval.model_dump(mode="json") for interval in result.sequences[0].intervals
    )
    assert sample.payload()["prediction"] == {
        "analysis_id": str(result.analysis_id),
        "snapshot_id": str(result.snapshot_id),
        "result_sha256": sample.prediction["result_sha256"],
    }
    assert "reference_kind" not in sample.payload()
    assert "annotation_policy" not in sample.payload()
    model = inspection.summary["model"]
    assert isinstance(model, dict)
    assert model["bundle_sha256"] == result.model.bundle_sha256


@pytest.mark.parametrize("shape", ["absent", "missing", "duplicate", "extra"])
def test_report_requires_exact_result_membership(shape: str) -> None:
    _, sources = dataset()
    results = predictions(sources)
    if shape == "absent":
        results = ()
    elif shape == "missing":
        results = results[:-1]
    elif shape == "duplicate":
        results = (*results[:-1], results[0])
    else:
        results = (*results, prediction_for(source(10).export))
    with pytest.raises(ValueError, match="exactly one saved result"):
        inspect_snapshots(sources, results=results, sample_count=1)


def test_results_match_by_snapshot_hash_instead_of_file_order() -> None:
    _, sources = dataset()
    results = predictions(sources)
    assert inspect_snapshots(sources, results=results) == inspect_snapshots(
        sources, results=tuple(reversed(results))
    )


@pytest.mark.parametrize(
    "field",
    ["model", "analysis", "inference_runtime", "source", "time_grid", "units", "statistics"],
)
def test_unselected_results_cannot_hide_different_models_or_invalid_evidence(field: str) -> None:
    _, sources = dataset()
    results = predictions(sources)
    last = results[-1]
    if field == "model":
        last = last.model_copy(
            update={"model": last.model.model_copy(update={"bundle_sha256": "b" * 64})}
        )
    elif field == "analysis":
        last = last.model_copy(
            update={"analysis": last.analysis.model_copy(update={"min_duration": 8})}
        )
    elif field == "inference_runtime":
        last = last.model_copy(
            update={"inference_runtime": last.inference_runtime.model_copy(update={"threads": 1})}
        )
    elif field == "source":
        last = last.model_copy(
            update={"source": last.source.model_copy(update={"run_id": "different"})}
        )
    elif field == "time_grid":
        last = last.model_copy(
            update={"time_grid": last.time_grid.model_copy(update={"unit": "other-time"})}
        )
    elif field == "units":
        instrument = last.instruments[0]
        units = instrument.units.model_copy(update={"price_decimal_places": 10})
        last = last.model_copy(
            update={"instruments": (instrument.model_copy(update={"units": units}),)}
        )
    else:
        sequence = last.sequences[0]
        interval = sequence.intervals[0]
        changed = interval.model_copy(
            update={"statistics": interval.statistics.model_copy(update={"trade_count": 10})}
        )
        last = last.model_copy(
            update={"sequences": (sequence.model_copy(update={"intervals": (changed,)}),)}
        )
    with pytest.raises(ValueError):
        inspect_snapshots(sources, results=(*results[:-1], last), sample_count=1)


@pytest.mark.parametrize("field", ["analysis_id", "snapshot_id"])
def test_saved_resource_identity_cannot_be_reused_for_another_snapshot(field: str) -> None:
    _, sources = dataset()
    results = predictions(sources)
    changed = results[-1].model_copy(update={field: getattr(results[0], field)})
    with pytest.raises(ValueError, match="distinct saved resource identities"):
        inspect_snapshots(sources, results=(*results[:-1], changed))


def test_result_runtime_must_match_its_recorded_policy_digest() -> None:
    sources = (source(0),)
    result = prediction_for(sources[0].export)
    result = result.model_copy(
        update={"model": result.model.model_copy(update={"inference_policy_sha256": "b" * 64})}
    )
    with pytest.raises(ValueError, match="provenance identities differ"):
        inspect_snapshots(sources, results=(result,))


def test_report_preserves_signed_zero_in_timing_and_model_identity() -> None:
    original = source(0)
    payload = original.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["time_grid"].update(origin=-0.0, completed_horizon=4.0)
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
    snapshot = original.model_copy(update={"export": export})
    result = prediction_for(export)
    changed = result.model_copy(
        update={"time_grid": result.time_grid.model_copy(update={"origin": 0.0})}
    )
    with pytest.raises(ValueError, match="provenance identities differ"):
        inspect_snapshots((snapshot,), results=(changed,))

    _, sources = dataset()
    results = tuple(
        result.model_copy(
            update={"analysis": result.analysis.model_copy(update={"decoder_switch_penalty": 0.0})}
        )
        for result in predictions(sources)
    )
    changed = results[-1].model_copy(
        update={
            "analysis": results[-1].analysis.model_copy(update={"decoder_switch_penalty": -0.0})
        }
    )
    with pytest.raises(ValueError, match="same immutable model"):
        inspect_snapshots(sources, results=(*results[:-1], changed))


def test_current_export_rejects_price_changes_without_executions() -> None:
    original = source(0)
    value = original.export.model_dump(mode="json", exclude={"content_sha256"})
    value["observations"][1]["last_price"] = 101.0
    with pytest.raises(ValueError, match="carried"):
        seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))


def test_offline_report_preserves_source_identity_and_canonical_symbols(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    from regimex.visualization import _script_json, render_snapshot_report

    manifest, sources = dataset()
    output = tmp_path / "report"
    inspection = render_snapshot_report(
        sources, output, results=predictions(sources), dataset=manifest, sample_count=2
    )
    data = report_data(output)
    assert data["kind"] == "prediction"
    assert inspection.summary["report_version"] == "6.2"
    assert all("events" not in sample.payload() for sample in inspection.samples)
    assert data["samples"] == json.loads(
        json.dumps([sample.payload() for sample in inspection.samples])
    )
    assert json.loads((output / "snapshots.json").read_text()) == [
        item.model_dump(mode="json") for item in sources
    ]
    record = json.loads((output / "report-manifest.json").read_text())
    assert record["version"] == "report-manifest/1.0"
    assert record["source_sha256"] == inspection.summary["source_content_sha256"]
    assert all(digest(output / name) == checksum for name, checksum in record["files"].items())
    notices = (output / "LICENSES.md").read_text()
    for package, filename in (("regimex", "licenses/LICENSE"), ("plotly", "licenses/LICENSE.txt")):
        license_text = distribution(package).read_text(filename)
        assert license_text and license_text in notices
    assert record["files"]["LICENSES.md"] == digest(output / "LICENSES.md")
    plotly_js = Path(str(distribution("plotly").locate_file("plotly/package_data/plotly.min.js")))
    assert (output / "plotly.min.js").read_bytes() == plotly_js.read_bytes()
    assert not tuple(output.glob("*.png"))
    assert "matplotlib_version" not in record
    assert json.loads((output / "results.json").read_text()) == [
        item.model_dump(mode="json") for item in predictions(sources)
    ]
    html = (output / "index.html").read_text()
    assert 'src="plotly.min.js"' in html and "https://" not in html
    assert "Dataset checks" not in html and 'id="diagnostics"' not in html
    assert "How to read" not in html and 'id="event-summary"' not in html
    assert '<footer class="provenance"' in html
    malicious = "</script><script>alert('x')</script>"
    assert "</script>" not in _script_json(malicious)
    assert json.loads(_script_json(malicious)) == malicious


def test_blind_review_exports_no_automatic_labels_or_seed_metadata(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    from regimex.visualization import review_reference

    export = source(0, instrument_count=2, missing=True).export
    output = tmp_path / "blind"
    review_reference(export, output, AnalysisConfig(min_duration=4))
    data = report_data(output)
    assert data["kind"] == "reference"
    samples = data["samples"]
    assert isinstance(samples, list)
    for sample in samples:
        assert sample["intervals"] == [] and "events" not in sample
        assert "simulation" not in sample and "seed" not in sample
        assert sample["observations"]["last_price"] == [None] * 8
        assert sample["observations"]["trade_count"] == [0] * 8
    assert json.loads((output / "observations.json").read_text()) == {
        "experiment_id": export.content_sha256,
        **export.model_dump(mode="json", include={"instruments", "time_grid", "observations"}),
    }
    assert all(sample["experiment_id"] == export.content_sha256 for sample in samples)
    assert not (output / "export.json").exists()
    with pytest.raises(ValueError):
        DirectionAnnotations.model_validate_json((output / "annotation-draft.json").read_bytes())
