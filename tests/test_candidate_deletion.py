"""Exercise the training-run, version registry and object store across weight removal."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("psycopg")

from psycopg.types.json import Json
from test_admin_repository import (
    candidate,
    claim,
    evaluated,
    principal,
    ready_training,
    training_metrics,
)
from test_admin_repository import repository as repository

from regimex.admin_contracts import DatasetRecord, EvaluationSpec, PublicationSpec, TrainingSpec
from regimex.application.deletion import DeletionService
from regimex.deletion_contracts import DeletionPreview
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.deletion_repository import DeletionRepository
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.service_contracts import Conflict, ModelRecord, NotFound, Principal
from regimex.storage import digest


@dataclass
class Candidate:
    run_id: UUID
    dataset: DatasetRecord
    spec: TrainingSpec
    model: ModelRecord
    weight: ObjectInfo
    deletion: DeletionService

    def preview(self, identity: Principal) -> DeletionPreview:
        return self.deletion.repository.preview_weights(
            identity, self.run_id, self.weight, store_scope_id=self.deletion.store.scope_id
        )


@pytest.fixture
def trained_candidate(repository: AdminRepository, tmp_path: Path) -> Candidate:
    dataset, spec = ready_training(repository)
    run = repository.submit(principal(), spec, idempotency_key="training")
    store = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    source = tmp_path / "weights"
    source.write_bytes(b"candidate weights")
    weight = store.put("fixture/weights", source, expected_sha256=digest(source))
    model = candidate(spec).model_copy(update={"members": {"nested/weights": weight}})
    repository.complete_training(
        run.run_id, claim(repository, run.run_id), model, metrics=training_metrics(spec)
    )
    return Candidate(
        run.run_id,
        dataset,
        spec,
        model,
        weight,
        DeletionService(DeletionRepository(repository.database), store),
    )


def test_removal_retains_history_and_evaluation_and_can_resume_after_storage_failure(
    repository: AdminRepository, trained_candidate: Candidate, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = trained_candidate
    owner = principal()
    store = trained.deletion.store
    result = evaluated(repository, trained.dataset, trained.model)
    original = repository.get_run(owner, trained.run_id)
    preview = trained.preview(owner)
    assert not preview.blockers
    assert (preview.size_bytes, preview.object_count, preview.target_id) == (
        trained.weight.size_bytes,
        1,
        trained.run_id,
    )
    with pytest.raises(NotFound):
        trained.preview(principal("another-admin"))
    for other in [principal("another-admin"), principal(admin=False)]:
        with pytest.raises(NotFound):
            trained.deletion.apply(other, preview.plan_id, expected_sha256=preview.sha256)

    def fail(_info: ObjectInfo) -> None:
        raise OSError("storage unavailable")

    with monkeypatch.context() as context:
        context.setattr(store, "delete_version", fail)
        pending = trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert pending.state == "pending" and pending.pending_bytes == trained.weight.size_bytes
    assert trained.deletion.list_pending(principal(admin=False)) == ()
    assert trained.deletion.list_pending(owner) == (pending,)
    assert store.stat(trained.weight.key) == trained.weight
    with pytest.raises(NotFound):
        repository.database.get_model(trained.spec.model_version_id)
    with pytest.raises(Conflict, match="reserved"):
        repository.database.existing_registration(owner, trained.spec.model_version_id)
    assert repository.database.list_models(owner) == ()
    assert repository.get_run(owner, trained.run_id).output == original.output
    assert repository.get_run(owner, trained.run_id).weights_deletion_id == preview.plan_id
    assert repository.get_training_metrics(owner, trained.run_id) == training_metrics(trained.spec)
    assert repository.get_evaluation(owner, result.run_id) == result
    with pytest.raises(NotFound):
        repository.submit(
            owner,
            PublicationSpec(
                model_version_id=trained.spec.model_version_id, evaluation_run_id=result.run_id
            ),
            idempotency_key="publish-discarded",
        )
    receipt = trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "completed" and receipt.reclaimed_bytes == trained.weight.size_bytes
    assert trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256) == receipt
    with pytest.raises(FileNotFoundError):
        store.stat(trained.weight.key)


@pytest.mark.parametrize("discard_first", [False, True])
def test_corrupt_training_score_is_detected_with_or_without_weights(
    repository: AdminRepository, trained_candidate: Candidate, discard_first: bool
) -> None:
    trained = trained_candidate
    owner = principal()
    preview = trained.preview(owner)
    if discard_first:
        trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    with repository.database.connection() as connection:
        row = connection.execute(
            "SELECT document FROM regimex_training_metrics WHERE operation_id=%s", (trained.run_id,)
        ).fetchone()
        assert row is not None and isinstance(row[0], dict)
        document = row[0]
        document["metrics"]["validation_segment_macro_f1"] = 0.5
        connection.execute(
            "UPDATE regimex_training_metrics SET document=%s WHERE operation_id=%s",
            (Json(document), trained.run_id),
        )
    with pytest.raises(Conflict, match="checksum"):
        repository.get_training_metrics(owner, trained.run_id)
    with pytest.raises(Conflict, match="checksum"):
        repository.save_training_metrics(
            owner,
            trained.run_id,
            bundle_sha256=trained.model.bundle_sha256,
            metrics=training_metrics(trained.spec),
        )
    if not discard_first:
        with pytest.raises(Conflict, match="checksum"):
            trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
        assert trained.deletion.store.stat(trained.weight.key) == trained.weight


def test_new_model_references_invalidate_a_preview(
    repository: AdminRepository, trained_candidate: Candidate
) -> None:
    trained = trained_candidate
    owner = principal()
    preview = trained.preview(owner)
    evaluation = repository.submit(
        owner,
        EvaluationSpec(
            model_version_id=trained.model.model_version_id,
            dataset_id=trained.dataset.dataset_id,
            reference_set_id=trained.spec.validation_reference_set_id,
        ),
        idempotency_key="validation",
    )
    assert [blocker.code for blocker in trained.preview(owner).blockers] == ["active_model_jobs"]
    with pytest.raises(Conflict, match="changed"):
        trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert trained.deletion.store.stat(trained.weight.key) == trained.weight
    repository.database.cancel_operation(owner, evaluation.run_id)
    result = evaluated(repository, trained.dataset, trained.model)
    publication = repository.submit(
        owner,
        PublicationSpec(
            model_version_id=trained.model.model_version_id, evaluation_run_id=result.run_id
        ),
        idempotency_key="publish",
    )
    assert [blocker.code for blocker in trained.preview(owner).blockers] == ["active_model_jobs"]
    repository.complete_publication(publication.run_id, claim(repository, publication.run_id))
    blocked = trained.preview(owner)
    assert [blocker.code for blocker in blocked.blockers] == ["published_model"]
    with pytest.raises(Conflict, match="blocked"):
        trained.deletion.apply(owner, blocked.plan_id, expected_sha256=blocked.sha256)
    assert repository.database.get_active_model() == trained.model


def test_frozen_choice_and_shared_weight_remain_protected(
    repository: AdminRepository, trained_candidate: Candidate
) -> None:
    trained = trained_candidate
    owner = principal()
    database = repository.database
    another_model = trained.model.model_copy(update={"model_version_id": "other-version"})
    database.register_model(owner, another_model)
    with database.connection() as connection:
        # The benchmark service owns its registry document. The deletion check
        # uses the relational choice that pins the model for final evaluation.
        connection.execute(
            "INSERT INTO regimex_benchmark_bindings "
            "(benchmark_id,owner_id,dataset_id,reference_set_id,document) "
            "VALUES ('reserved',%s,%s,%s,'{}')",
            (owner.owner_id, trained.dataset.dataset_id, trained.spec.validation_reference_set_id),
        )
        connection.execute(
            "INSERT INTO regimex_benchmark_choices (benchmark_id,model_version_id,document) "
            "VALUES ('reserved',%s,'{}')",
            (trained.spec.model_version_id,),
        )
    preview = trained.preview(owner)
    assert {blocker.code for blocker in preview.blockers} == {"benchmark_choice", "shared_objects"}
    with pytest.raises(Conflict, match="blocked"):
        trained.deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert trained.deletion.store.stat(trained.weight.key) == trained.weight
