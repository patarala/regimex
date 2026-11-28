"""Scientific administrative tasks executed only in an isolated child process."""

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pydantic import JsonValue, RootModel, TypeAdapter

from regimex.admin_contracts import (
    AdminSpec,
    CatalogBinding,
    DatasetRecord,
    EvaluationSpec,
    GenerationSpec,
    ReferenceGenerationSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.application.analysis_computation import restore_predictor
from regimex.application.training import (
    SharedSelectionEvaluator,
    TrainingRequest,
    observation_inputs,
    train_candidate,
)
from regimex.application.training_progress import record_training_progress
from regimex.benchmark_contracts import FinalEvaluationGrant
from regimex.benchmarks import DirectionSemantics
from regimex.contracts import StrictModel
from regimex.data_catalog import (
    Partition,
    ReferenceMember,
    ReferencePayload,
    ReferenceSet,
    SnapshotMember,
    SourceSnapshot,
    TrainingCatalog,
    _load_partition,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import DirectionAnnotations
from regimex.exports import ExportInstrument, ExportObservation, ExportV2
from regimex.market_export import simulate_export
from regimex.market_identity import market_seed_group
from regimex.market_sampling import sample_market
from regimex.model_bundles import ModelBundle
from regimex.model_contracts import LoadedPredictor
from regimex.model_registry import ArchitectureRegistry
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.segmentation import DECODER_VERSION, annotate
from regimex.service_contracts import ModelRecord
from regimex.storage import json_bytes

_SPEC = RootModel[AdminSpec]
_DICT = TypeAdapter(dict[str, JsonValue])


def _parse[T: StrictModel](value: JsonValue, model: type[T]) -> T:
    return model.model_validate_json(json_bytes(value))


def _object(value: JsonValue) -> dict[str, JsonValue]:
    return _DICT.validate_json(json_bytes(value))


def _items(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("expected a computation input list")
    return value


def _path(root: Path, value: JsonValue) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ValueError("computation paths must remain inside the private scratch directory")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise ValueError("computation inputs must be materialized regular files")
    return path


@dataclass(frozen=True)
class _Grid:
    export: ExportV2

    @property
    def experiment_id(self) -> str:
        return self.export.source.snapshot_id

    @property
    def step_count(self) -> int:
        return self.export.time_grid.step_count

    @property
    def instruments(self) -> tuple[ExportInstrument, ...]:
        return self.export.instruments

    @property
    def observations(self) -> tuple[ExportObservation, ...]:
        return self.export.observations


def _raw_sources(
    context: dict[str, JsonValue], root: Path
) -> dict[tuple[str, str], SourceSnapshot]:
    result = {}
    for raw in _items(context["snapshots"]):
        item = _object(raw)
        binding = _parse(item["binding"], CatalogBinding)
        export = ExportV2.model_validate_json(_path(root, item["path"]).read_bytes())
        source = SourceSnapshot(
            source_namespace=binding.source_namespace,
            split_group=binding.split_group,
            export=export,
        )
        result[(binding.source_namespace, export.source.snapshot_id)] = source
    return result


def _generation(spec: GenerationSpec, root: Path, owner: str) -> dict[str, JsonValue]:
    recipe: dict[str, JsonValue] = {
        "version": "service-exchange/2.0",
        "plan": spec.plan.model_dump(mode="json"),
    }
    # Jobs created before version pinning must reproduce their original snapshots on retry.
    version = spec.saved_generator_version
    if version != "4.0":
        recipe.update(version="service-exchange/2.1", generator_version=version)
    identity = content_hash(recipe)
    seeds = [
        int(np.random.SeedSequence([spec.plan.seed, index]).generate_state(1)[0])
        for index in range(spec.plan.experiment_count)
    ]
    # Assign seed groups before simulation. No annotation outcome can influence
    # membership, random seeds, market configuration or experiment retention.
    groups = sorted(set(seeds))
    ordering = np.random.Generator(np.random.PCG64(spec.splits.seed)).permutation(groups).tolist()
    if len(groups) < 3:
        raise ValueError("generation requires at least three independent seed groups")
    train_count = min(len(groups) - 2, max(1, int(len(groups) * spec.splits.train_fraction)))
    validation_count = min(
        len(groups) - train_count - 1, max(1, int(len(groups) * spec.splits.validation_fraction))
    )
    partitions = {
        seed: "train"
        if index < train_count
        else "validation"
        if index < train_count + validation_count
        else "test"
        for index, seed in enumerate(ordering)
    }
    sources = []
    records: list[JsonValue] = []
    provenance = []
    (root / "output").mkdir()
    for index, seed in enumerate(seeds):
        market = spec.plan.market or sample_market(seed, 1 + index % 2)
        external_id = str(uuid5(NAMESPACE_URL, f"regimex:{identity}:{index}"))
        snapshot_id = uuid5(NAMESPACE_URL, f"regimex-snapshot:{owner}:{external_id}")
        namespace = content_hash([owner, "regimex-exchange", external_id])
        export, audit = simulate_export(
            market,
            seed=seed,
            steps=spec.plan.step_count,
            configuration_id=identity,
            run_id=external_id,
            generator_version=version,
        )
        binding = CatalogBinding(
            snapshot_id=snapshot_id,
            source_namespace=namespace,
            split_group=market_seed_group(seed),
        )
        source = SourceSnapshot(
            source_namespace=namespace, split_group=binding.split_group, export=export
        )
        sources.append(create_snapshot_member(source, cast(Partition, partitions[seed])))
        name = f"output/snapshot-{snapshot_id}.json"
        (root / name).write_bytes(json_bytes(export.model_dump(mode="json")))
        records.append(
            {
                "snapshot_id": str(snapshot_id),
                "binding": binding.model_dump(mode="json"),
                "path": name,
            }
        )
        provenance.append(
            {
                "snapshot_id": str(snapshot_id),
                "seed": seed,
                "market": market.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
            }
        )
    manifest = seal_dataset(spec.dataset_id, tuple(sources))
    (root / "output/generation-provenance.json").write_bytes(
        json_bytes(
            {
                **recipe,
                "splits": spec.splits.model_dump(mode="json"),
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "members": provenance,
            }
        )
    )
    return {
        "manifest": manifest.model_dump(mode="json"),
        "snapshots": records,
        "provenance_path": "output/generation-provenance.json",
    }


def _predictor(
    context: dict[str, JsonValue], root: Path
) -> tuple[LoadedPredictor, ModelBundle, ModelRecord]:
    model = _parse(context["model"], ModelRecord)
    predictor, manifest = restore_predictor(
        root / "input/model",
        model,
        ArchitectureRegistry.discover(),
        worker_id="administrative-child",
    )
    return predictor, manifest, model


def compute(payload: dict[str, JsonValue], root: Path) -> dict[str, JsonValue]:
    spec = _SPEC.model_validate_json(json_bytes(payload["spec"])).root
    owner = payload["owner_id"]
    if not isinstance(owner, str):
        raise ValueError("computation requires an owner identity")
    if isinstance(spec, GenerationSpec):
        return _generation(spec, root, owner)
    context = _object(payload["inputs"])
    if isinstance(spec, ReferenceGenerationSpec):
        dataset = _parse(context["dataset"], DatasetRecord)
        sources = _raw_sources(context, root)
        members = []
        documents: list[JsonValue] = []
        (root / "output").mkdir()
        for member in dataset.manifest.members:
            if member.partition != spec.partition:
                continue
            source = sources[(member.source_namespace, member.snapshot_id)]
            # annotate's required view is observations and grid only. It does not
            # receive or reconstruct an experiment seed or generator state.
            annotations = annotate(_Grid(source.export), spec.analysis, policy=spec.saved_policy)
            members.append(create_reference_member(member, annotations))
            name = f"output/annotation-{len(documents)}.json"
            (root / name).write_bytes(json_bytes(annotations.model_dump(mode="json")))
            documents.append(
                {
                    "source_namespace": member.source_namespace,
                    "snapshot_id": member.snapshot_id,
                    "path": name,
                }
            )
        references = seal_reference_set(
            ReferencePayload(
                reference_set_id=spec.reference_set_id,
                dataset_sha256=dataset.manifest.content_sha256,
                partition=spec.partition,
                kind="automatic",
                policy=spec.saved_policy,
                semantics=DirectionSemantics(
                    profile_sha256=analysis_profile_hash(spec.analysis),
                    decoder_version=DECODER_VERSION,
                ),
                analysis_profile=spec.analysis,
                members=tuple(members),
            )
        )
        return {"references": references.model_dump(mode="json"), "annotations": documents}
    if isinstance(spec, (TrainingSpec, EvaluationSpec)):
        dataset = _parse(context["dataset"], DatasetRecord)
        sources = _raw_sources(context, root)
        sets = [_parse(item, ReferenceSetRecord) for item in _items(context["reference_sets"])]
        annotation_documents: dict[tuple[str, str], DirectionAnnotations] = {}
        for raw in _items(context["annotations"]):
            item = _object(raw)
            namespace, snapshot_id = item["source_namespace"], item["snapshot_id"]
            if not isinstance(namespace, str) or not isinstance(snapshot_id, str):
                raise ValueError("annotation input requires namespace and snapshot identity")
            annotation_documents[(namespace, snapshot_id)] = (
                DirectionAnnotations.model_validate_json(_path(root, item["path"]).read_bytes())
            )

        def snapshot_reader(member: SnapshotMember) -> SourceSnapshot:
            # The catalog loader verifies and copies each member once. Release the
            # source as it is consumed to avoid retaining a second set of exports.
            return sources.pop((member.source_namespace, member.snapshot_id))

        def reference_reader(
            references: ReferenceSet, member: ReferenceMember
        ) -> DirectionAnnotations:
            return annotation_documents[(member.source_namespace, member.snapshot_id)]

        if isinstance(spec, TrainingSpec):
            by_id = {item.references.reference_set_id: item.references for item in sets}
            catalog = TrainingCatalog(
                dataset.manifest,
                by_id[spec.labels_reference_set_id],
                by_id[spec.validation_reference_set_id],
                snapshot_reader=snapshot_reader,
                reference_reader=reference_reader,
            )
            request = TrainingRequest(
                run_id=cast(str, payload["run_id"]),
                model_version_id=spec.model_version_id,
                architecture_id=spec.architecture_id,
                parameters=spec.parameters,
                policy=spec.policy,
                training_runtime=spec.training_runtime,
                verification_runtime=spec.inference_runtime,
                analysis_profile_id=spec.analysis_profile_id,
                build_id=spec.build_id,
            )
            with record_training_progress(root) as record:
                result = train_candidate(
                    catalog,
                    request,
                    root / "output/bundle",
                    registry=ArchitectureRegistry.discover(),
                    worker_id="administrative-child",
                    on_history=record,
                )
            return {"candidate": result.model_dump(mode="json"), "bundle_path": "output/bundle"}
        reference_set = sets[0].references
        if reference_set.partition != spec.partition:
            raise ValueError("evaluation references differ from the requested partition")
        validate_reference_membership(dataset.manifest, reference_set)
        samples = _load_partition(
            dataset.manifest,
            reference_set,
            snapshot_reader=snapshot_reader,
            reference_reader=reference_reader,
        )
        predictor, manifest, model = _predictor(context, root)
        descriptor = ArchitectureRegistry.discover().get(model.architecture_id)
        for sample in samples:
            descriptor.validate_observation_fields(sample.export.capabilities.observation_fields())
        if (
            (
                spec.partition == "validation"
                and manifest.training.dataset_sha256 != dataset.manifest.content_sha256
            )
            or manifest.analysis_profile_sha256 != reference_set.semantics.profile_sha256
            or manifest.decoder_version != reference_set.semantics.decoder_version
            or predictor.analysis != reference_set.analysis_profile
        ):
            raise ValueError("evaluation dataset or reference scale differs from training evidence")
        if spec.partition == "test":
            grant = _parse(context["final_grant"], FinalEvaluationGrant)
            if (
                grant.owner_id,
                grant.benchmark_id,
                grant.dataset_id,
                grant.dataset_sha256,
                grant.reference_set_id,
                grant.reference_sha256,
                grant.model_version_id,
                grant.bundle_sha256,
                grant.inference_policy_sha256,
            ) != (
                owner,
                spec.benchmark_id,
                dataset.dataset_id,
                dataset.manifest.content_sha256,
                reference_set.reference_set_id,
                reference_set.content_sha256,
                model.model_version_id,
                model.bundle_sha256,
                model.inference_policy_sha256,
            ):
                raise ValueError("final evaluation differs from the frozen authorization grant")
        inputs = observation_inputs(samples)
        evaluation_references = tuple(
            tuple(
                interval
                for interval in sample.annotations.intervals
                if interval.instrument_id == instrument.instrument_id
            )
            for sample in samples
            for instrument in sample.export.instruments
        )
        evaluator = SharedSelectionEvaluator(
            inputs, evaluation_references, predictor.analysis, reference_set.kind
        )
        metrics = evaluator.evaluate(
            predictor.predict(inputs, runtime=model.inference_runtime),
            bootstrap_repetitions=1000,
            include_validation_diagnostics=spec.partition == "validation",
        )
        metrics.update(
            {
                "scope": "fixed_model_validation"
                if spec.partition == "validation"
                else "controlled_final_benchmark",
                "benchmark_id": spec.benchmark_id,
                "model_version_id": model.model_version_id,
                "bundle_sha256": model.bundle_sha256,
                "dataset_sha256": dataset.manifest.content_sha256,
                "reference_sha256": reference_set.content_sha256,
                "reference_policy": reference_set.policy,
                "training_reference_kind": manifest.training.labels.kind,
                "training_reference_policy": manifest.training.labels.policy,
                "training_reference_sha256": manifest.training.labels.content_sha256,
                "validation_selection_reference_sha256": (
                    manifest.training.validation_references.content_sha256
                ),
                "analysis_profile_sha256": manifest.analysis_profile_sha256,
                "decoder_version": manifest.decoder_version,
                "inference_runtime": model.inference_runtime.model_dump(mode="json"),
                "quality_status": "validation_only",
                "independent": reference_set.independent,
                "blind_review": reference_set.blind_review,
            }
        )
        (root / "output").mkdir()
        encoded = json_bytes(metrics)
        if len(encoded) > MAX_REPORT_BYTES and "validation_examples" in metrics:
            # Reserve the report's size budget for the complete scientific evidence.
            del metrics["validation_examples"]
            encoded = json_bytes(metrics)
        (root / "output/evaluation.json").write_bytes(encoded)
        return {"metrics": metrics, "report_path": "output/evaluation.json"}
    _predictor(context, root)
    return {"verified": True}
