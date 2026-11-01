"""Scientific administrative tasks executed only in an isolated child process."""

import platform
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pydantic import JsonValue, TypeAdapter

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
from regimex.application.training import (
    SharedSelectionEvaluator,
    TrainingRequest,
    _available_fields,
    observation_inputs,
    train_candidate,
)
from regimex.benchmark_contracts import FinalEvaluationGrant
from regimex.benchmarks import DirectionSemantics
from regimex.contracts import Producer, StrictModel
from regimex.data_catalog import (
    Partition,
    ReferenceMember,
    ReferencePayload,
    ReferenceSet,
    SnapshotMember,
    SourceSnapshot,
    SplitGroup,
    _load_partition,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    load_training_data,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import DIRECTION_POLICY, DirectionAnnotations
from regimex.exports import (
    ExportInstrument,
    ExportObservation,
    ExportPayloadV2,
    ExportV2,
    HistoricalQuoteCapability,
    InstrumentUnits,
    ModelTimeGrid,
    SourceCapabilities,
    SourceIdentity,
    seal_export,
)
from regimex.market import simulate
from regimex.market_sampling import GENERATOR_VERSION, sample_market
from regimex.model_bundles import ModelBundle, verify_model_bundle
from regimex.model_contracts import LoadedPredictor, ModelBackend
from regimex.model_registry import ArchitectureRegistry, RegistryError
from regimex.segmentation import DECODER_VERSION, annotate
from regimex.service_contracts import ModelRecord
from regimex.storage import json_bytes

_SPEC: TypeAdapter[AdminSpec] = TypeAdapter(AdminSpec)
_DICT = TypeAdapter(dict[str, JsonValue])


def _parse[T: StrictModel](value: JsonValue, model: type[T]) -> T:
    return model.model_validate_json(json_bytes(value))


def _object(value: JsonValue) -> dict[str, JsonValue]:
    return _DICT.validate_json(json_bytes(value))


def _items(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("expected a computation input list")
    return value


def _precision(number: float) -> int:
    return max(0, -int(Decimal(str(number)).normalize().as_tuple().exponent))


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
    identity = content_hash(
        {"version": "service-exchange/2.0", "plan": spec.plan.model_dump(mode="json")}
    )
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
        market_run = simulate(market, seed, spec.plan.step_count)
        external_id = str(uuid5(NAMESPACE_URL, f"regimex:{identity}:{index}"))
        snapshot_id = uuid5(NAMESPACE_URL, f"regimex-snapshot:{owner}:{external_id}")
        namespace = content_hash([owner, "regimex-exchange", external_id])
        export = seal_export(
            ExportPayloadV2(
                schema_version="2.1",
                source=SourceIdentity(
                    source_id="regimex-exchange",
                    source_version=GENERATOR_VERSION,
                    configuration_id=identity,
                    run_id=external_id,
                    snapshot_id=external_id,
                    state_revision="completed-1",
                    run_status="completed",
                    consistency="atomic_completed_snapshot",
                ),
                producer=Producer(name="regimex-exchange", version=GENERATOR_VERSION),
                time_grid=ModelTimeGrid(
                    origin=0.0,
                    unit="simulation_step",
                    bin_width=1.0,
                    completed_horizon=float(spec.plan.step_count),
                    step_count=spec.plan.step_count,
                    interval_bounds="[start,stop)",
                    terminal_bin="full_bins_only",
                ),
                capabilities=SourceCapabilities(
                    execution_prices="complete",
                    trade_aggregates="complete",
                    accepted_order_aggregates="complete",
                    historical_quotes=HistoricalQuoteCapability(
                        mode="retained_snapshots",
                        coverage="complete",
                        provenance="Closing resting book after all events in each simulation step.",
                        complete_event_history_sha256=None,
                        reconstruction_policy=None,
                    ),
                    price_policy="initial_then_carry",
                    trade_count_policy="one_per_partial_execution",
                    order_count_policy="accepted_submissions_including_immediate_fills",
                    rejected_order_policy="excluded",
                ),
                instruments=tuple(
                    ExportInstrument(
                        **instrument.model_dump(),
                        units=InstrumentUnits(
                            price_unit="quote_asset_per_base_asset",
                            volume_unit="base_asset",
                            price_decimal_places=_precision(market.tick_size),
                            volume_decimal_places=_precision(market.lot_size),
                        ),
                    )
                    for instrument in market_run.instruments
                ),
                observations=tuple(
                    ExportObservation(
                        **row.model_dump(),
                        price_state="observed" if row.trade_count else "carried",
                        bid_state="observed" if row.best_bid is not None else "empty",
                        ask_state="observed" if row.best_ask is not None else "empty",
                    )
                    for row in market_run.observations
                ),
            )
        )
        # Native asset names change content identity, not the stochastic market.
        # Keep earlier same-seed data in this group to prevent split leakage.
        binding = CatalogBinding(
            snapshot_id=snapshot_id,
            source_namespace=namespace,
            split_group=SplitGroup(
                namespace="regimex-exchange-seed/3.0", group_id=f"seed:{seed}", seed=seed
            ),
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
                "audit": market_run.audit.model_dump(mode="json"),
            }
        )
    manifest = seal_dataset(spec.dataset_id, tuple(sources))
    (root / "output/generation-provenance.json").write_bytes(
        json_bytes(
            {
                "version": "service-exchange/2.0",
                "plan": spec.plan.model_dump(mode="json"),
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
    path = root / "input/model"
    registry = ArchitectureRegistry.discover()
    descriptor = registry.get(model.architecture_id)
    manifest = verify_model_bundle(
        path,
        manifest_sha256=model.bundle_sha256,
        architecture_id=descriptor.architecture_id,
        implementation_version=descriptor.implementation_version,
    )
    factory = registry.load_backend_factory_on_worker(
        model.architecture_id, model.inference_runtime, worker_id="administrative-child"
    )
    backend = factory()
    if not isinstance(backend, ModelBackend) or (
        backend.architecture_id,
        backend.implementation_version,
    ) != (descriptor.architecture_id, descriptor.implementation_version):
        raise RegistryError("installed backend identity differs from its descriptor")
    predictor = backend.load_bundle(
        path, manifest_sha256=model.bundle_sha256, runtime=model.inference_runtime
    )
    if (
        manifest.model_version_id != model.model_version_id
        or predictor.architecture_id != model.architecture_id
        or predictor.implementation_version != manifest.implementation_version
        or predictor.class_names != manifest.class_names
        or content_hash(predictor.analysis.model_dump(mode="json"))
        != manifest.analysis_profile_sha256
        or manifest.decoder_version != DECODER_VERSION
    ):
        raise ValueError("model identity, decoder or restored analysis profile is incompatible")
    return predictor, manifest, model


def compute(payload: dict[str, JsonValue], root: Path) -> dict[str, JsonValue]:
    spec = _SPEC.validate_json(json_bytes(payload["spec"]))
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
            annotations = annotate(_Grid(source.export), spec.analysis)
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
                policy=DIRECTION_POLICY,
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
            return sources[(member.source_namespace, member.snapshot_id)]

        def reference_reader(
            references: ReferenceSet, member: ReferenceMember
        ) -> DirectionAnnotations:
            return annotation_documents[(member.source_namespace, member.snapshot_id)]

        if isinstance(spec, TrainingSpec):
            by_id = {item.references.reference_set_id: item.references for item in sets}
            data = load_training_data(
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
            result = train_candidate(
                data,
                request,
                root / "output/bundle",
                registry=ArchitectureRegistry.discover(),
                worker_id="administrative-child",
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
            descriptor.validate_observation_fields(_available_fields(sample))
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
        (root / "output/evaluation.json").write_bytes(json_bytes(metrics))
        return {"metrics": metrics, "report_path": "output/evaluation.json"}
    _predictor(context, root)
    return {"verified": True}
