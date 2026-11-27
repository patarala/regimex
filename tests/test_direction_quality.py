"""Direction evidence pools intervals by group and keeps evaluation references identifiable."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from regimex.direction_quality import (
    DirectionQualityReport,
    GroupCounts,
    MeasuredCohort,
    read_direction_quality,
)
from regimex.directions import DIRECTION_POLICY, PREVIOUS_DIRECTION_POLICY, AutomaticPolicy
from regimex.model_bundles import ReferenceIdentity, TrainingEvidence
from regimex.storage import json_bytes


def cohort() -> MeasuredCohort:
    return MeasuredCohort(
        reference="simulator",
        split="train",
        reference_sha256="1" * 64,
        groups=(
            GroupCounts(group_sha256="a" * 64, counts=((1, 0, 1), (0, 2, 0), (0, 0, 0))),
            GroupCounts(group_sha256="b" * 64, counts=((2, 0, 0), (2, 0, 0), (0, 0, 0))),
        ),
    )


def report(*rows: MeasuredCohort, bundle: str = "0" * 64) -> DirectionQualityReport:
    return DirectionQualityReport(
        metric="decoded_segment_macro_f1/iou-0.50",
        class_names=("sideways", "growth", "decline"),
        bundle_sha256=bundle,
        protocol_sha256="f" * 64,
        rows=rows,
    )


def test_pools_counts_before_macro_average_and_pins_the_source(tmp_path: Path) -> None:
    row = cohort()
    path = tmp_path / "scores.json"
    path.write_bytes(json_bytes(report(row).model_dump(mode="json")))
    result = read_direction_quality(path)
    (score,) = result.rows
    assert result.report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert score.group_count == 2
    assert score.interval_macro_f1 == pytest.approx((6 / 7 + 4 / 6) / 3)
    reversed_row = row.model_copy(update={"groups": row.groups[::-1]})
    assert report(reversed_row).summarize("3" * 64).rows == result.rows
    all_missed = row.groups[0].model_copy(update={"counts": ((0, 1, 1), (0, 0, 1), (0, 0, 1))})
    assert (
        report(row.model_copy(update={"groups": (all_missed,)}))
        .summarize("3" * 64)
        .rows[0]
        .interval_macro_f1
        == 0
    )
    with pytest.raises(ValueError, match="reference intervals"):
        report(
            row.model_copy(
                update={"groups": (all_missed.model_copy(update={"counts": ((0, 1, 0),) * 3}),)}
            )
        ).summarize("3" * 64)


def test_each_group_belongs_to_one_split_and_once_per_reference() -> None:
    row = cohort()
    other = row.model_copy(update={"reference": "known_direction", "reference_sha256": "4" * 64})
    assert len(report(row, other).summarize("3" * 64).rows) == 2
    for bad, reason in (
        ((row, row), "single direction score"),
        (
            (row, other.model_copy(update={"reference_sha256": row.reference_sha256})),
            "reference types",
        ),
        ((row, other.model_copy(update={"split": "independent"})), "overlapping groups"),
        ((row.model_copy(update={"groups": (row.groups[0], row.groups[0])}),), "contribute once"),
    ):
        with pytest.raises(ValueError, match=reason):
            report(*bad).summarize("3" * 64)


def test_cli_report_rejects_noninteger_and_out_of_range_counts(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    data = json.loads(report(cohort()).model_dump_json())
    for bad in (False, -1, 0.2, 2**32):
        data["rows"][0]["groups"][0]["counts"][0][0] = bad
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            read_direction_quality(path)


def test_evidence_binds_bundle_training_selection_and_reference_origin() -> None:
    labels = ReferenceIdentity(
        reference_set_id="train",
        kind="automatic",
        policy="original",
        content_sha256="1" * 64,
        direction_semantics_sha256="2" * 64,
    )
    training = TrainingEvidence(
        run_id="fitted",
        dataset_sha256="3" * 64,
        split_manifest_sha256="4" * 64,
        labels=labels,
        validation_references=labels.model_copy(update={"content_sha256": "5" * 64}),
        seeds=(1,),
        selected_seed=1,
        selection_metric="decoded_segment_macro_f1/iou-0.50",
        validation_segment_macro_f1=0.5,
    )
    row = cohort()
    evidence = report(row).summarize("3" * 64)
    evidence.validate_for("0" * 64, training)
    with pytest.raises(ValueError, match="another bundle"):
        evidence.validate_for("1" * 64, training)
    for change, reason in (
        ({"reference_sha256": "6" * 64}, "used to fit"),
        ({"split": "validation"}, "model selection"),
        ({"split": "validation", "reference_sha256": "5" * 64}, "selection score"),
        ({"split": "independent"}, "refer to training"),
        ({"reference": "known_direction"}, "reference type"),
    ):
        with pytest.raises(ValueError, match=reason):
            report(row.model_copy(update=change)).summarize("3" * 64).validate_for(
                "0" * 64, training
            )


@pytest.mark.parametrize("shared_source", [False, True], ids=["single-export", "multiple-exports"])
@pytest.mark.parametrize("policy", [PREVIOUS_DIRECTION_POLICY, DIRECTION_POLICY])
def test_offline_scorer_verifies_native_inputs_before_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_source: bool, policy: AutomaticPolicy
) -> None:
    import importlib
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from test_data_catalog import PROFILE, dataset, reference_set, snapshot

    from regimex.data_catalog import SnapshotGrid, create_snapshot_member, seal_dataset
    from regimex.segmentation import annotate

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    scorer = importlib.import_module("measure_direction_quality")
    measure = scorer.measure
    manifest, sources = dataset()
    source = sources[1]
    validation = [source]
    if shared_source:
        other = snapshot(3)
        manifest = seal_dataset(
            manifest.dataset_id, (*manifest.members, create_snapshot_member(other, "validation"))
        )
        validation.append(other)
    labels = tuple(
        annotate(SnapshotGrid.from_export(s.export), PROFILE, policy=policy) for s in validation
    )
    refs = reference_set(manifest, sources, "validation", documents=labels)
    bundle = SimpleNamespace(
        architecture_id="baseline",
        training=SimpleNamespace(
            labels=refs.identity(),
            validation_references=refs.identity(),
            dataset_sha256=manifest.content_sha256,
            validation_dataset_sha256=None,
        ),
    )
    predictor = SimpleNamespace(
        analysis=PROFILE,
        predict=lambda inputs, runtime: np.tile([1.0, 0.0, 0.0], (inputs.row_count, 1)),
    )
    dataset_path, refs_path, exports_path = (
        tmp_path / name for name in ("dataset", "refs", "locations")
    )
    dataset_path.write_text(manifest.model_dump_json())
    refs_path.write_text(refs.model_dump_json())
    locations = {}
    for item in validation:
        snapshot_id = item.export.source.snapshot_id
        path = tmp_path / f"{snapshot_id}.json"
        path.write_text(item.export.model_dump_json())
        locations[snapshot_id] = path.name
    export_path = tmp_path / locations[source.export.source.snapshot_id]
    paths = (dataset_path, refs_path, exports_path)
    exports_path.write_text(json.dumps({source.source_namespace: export_path.name}))
    if shared_source:
        with pytest.raises(ValueError, match="multiple snapshots"):
            measure(*paths, predictor, bundle, None, {})
        first = source.export.source.snapshot_id
        exports_path.write_text(json.dumps({source.source_namespace: {first: export_path.name}}))
        with pytest.raises(ValueError, match=validation[-1].export.source.snapshot_id):
            measure(*paths, predictor, bundle, None, {})
        exports_path.write_text(json.dumps({source.source_namespace: locations}))
    score = measure(*paths, predictor, bundle, None, {})
    member = manifest.members[1]
    altered = member.model_copy(
        update={"split_group": member.split_group.model_copy(update={"seed": 1000})}
    )
    exposures: dict[tuple[str, ...], tuple[str, str]] = {}
    scorer.record_group(member, exposures)
    with pytest.raises(ValueError, match="conflicting group or split"):
        scorer.record_group(altered, exposures)
    assert score.split == "validation"
    assert len(score.groups) == len(validation)
    assert sum(tp + fn for g in score.groups for tp, _, fn in g.counts) == sum(
        len(doc.intervals) for doc in labels
    )
    export_path.write_text(sources[0].export.model_dump_json())
    with pytest.raises(ValueError, match="differs from its registered dataset member"):
        measure(*paths, predictor, bundle, None, {})
    refs_path.write_text(reference_set(manifest, sources, "test").model_dump_json())
    with pytest.raises(ValueError, match="training or validation"):
        measure(*paths, predictor, bundle, None, {})
