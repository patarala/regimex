from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def mixed(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("train_mixed_direction")


def test_latent_path_replays_observations_and_separate_drift_targets(mixed: ModuleType) -> None:
    import numpy as np

    from regimex.directions import observation_blocks

    profile = mixed.PROFILE
    seed = mixed.FIRST_SEED
    prices, counts, rows, targets = mixed.generate(seed, 1024, profile)
    again = mixed.generate(seed, 1024, profile)
    for first, second in zip(
        (prices, counts, targets), (again[0], again[1], again[3]), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert mixed.observations(prices, counts) == rows
    assert (counts < 0).any()
    assert set(np.unique(targets)).issubset({-1, 0, 1, 2})
    for a, b, status in observation_blocks(rows, profile):
        if status != "observed":
            assert (targets[a:b] == -1).all()
    observation_hash, target_hash = mixed.hashes(prices, counts, targets)
    swapped = np.select([targets == 1, targets == 2], [2, 1], default=targets)
    other_observations, other_targets = mixed.hashes(prices, counts, swapped)
    assert other_observations == observation_hash
    assert other_targets != target_hash


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.45])
def test_each_source_supplies_its_registered_weight_in_every_class(
    mixed: ModuleType, fraction: float
) -> None:
    import numpy as np

    from regimex.backends.adaptive_forest import FEATURE_NAMES, ForestTrainingRows, training_rows

    def source(lengths: tuple[int, int, int]) -> list[ForestTrainingRows]:
        labels = np.repeat(np.arange(3, dtype=np.int64), lengths)
        x = np.ones((len(labels), len(FEATURE_NAMES)))
        return [training_rows(x, labels)]

    native = source((150, 80, 36))
    latent = source((36, 270, 128))
    total = mixed.class_totals(native, "counts") + mixed.class_totals(latent, "counts")
    balanced = mixed.mixture(native, latent, fraction)
    np.testing.assert_allclose(mixed.class_totals(balanced, "interval_weights"), total)
    np.testing.assert_allclose(
        mixed.class_totals(balanced[1:], "interval_weights"), fraction * total
    )
    np.testing.assert_array_equal(balanced[0].counts, native[0].counts)


@pytest.mark.parametrize("historical_run_id", [False, True], ids=["sample", "uuid"])
def test_native_training_export_must_match_the_declared_seed(
    mixed: ModuleType, tmp_path: Path, historical_run_id: bool
) -> None:
    from regimex.data_catalog import (
        SnapshotGrid,
        SourceSnapshot,
        create_reference_member,
        create_snapshot_member,
    )
    from regimex.exports import ExportPayloadV2, seal_export
    from regimex.market_export import sample_snapshot
    from regimex.market_identity import market_seed_group
    from regimex.segmentation import annotate
    from regimex.storage import json_bytes, write_json

    export = sample_snapshot(seed=7, steps=96, instruments=1)
    if historical_run_id:
        payload = export.model_dump(mode="json", exclude={"content_sha256"})
        identifier = export.source.run_id.split(":")[-1]
        payload["source"].update(snapshot_id=identifier, run_id=identifier)
        export = seal_export(ExportPayloadV2.model_validate_json(json_bytes(payload)))
    original = create_snapshot_member(
        SourceSnapshot(source_namespace="local", split_group=market_seed_group(7), export=export),
        "train",
    )
    labels = annotate(SnapshotGrid.from_export(export), mixed.PROFILE)
    reference = create_reference_member(original, labels)
    write_json(tmp_path / "export.json", export.model_dump(mode="json"))
    (tmp_path / "annotations").mkdir()
    for seed in (7, 8):
        write_json(tmp_path / "annotations" / f"{seed}.json", labels.model_dump(mode="json"))

    assert set(mixed.extract((original, reference, "export.json"), folder=tmp_path)) == {
        "greedy",
        "multiscale",
    }
    other = original.model_copy(update={"split_group": market_seed_group(8)})
    with pytest.raises(ValueError, match=r"declared split group|sampled market configuration"):
        mixed.extract((other, reference, "export.json"), folder=tmp_path)


@pytest.mark.parametrize(
    ("wrong_seed", "message"),
    [(True, "declared split group"), (False, "Invalid native sampled run identity")],
)
def test_local_fit_checks_unopened_partitions_before_reading_training_inputs(
    mixed: ModuleType, tmp_path: Path, wrong_seed: bool, message: str
) -> None:
    from regimex.data_catalog import (
        DatasetPayload,
        Partition,
        SourceSnapshot,
        create_snapshot_member,
    )
    from regimex.market_export import sample_snapshot
    from regimex.market_identity import market_seed_group
    from regimex.storage import content_hash, write_json

    parts: tuple[Partition, ...] = ("train", "validation", "test")
    members = []
    for seed, part in enumerate(parts, start=7):
        source = SourceSnapshot(
            source_namespace=part,
            split_group=market_seed_group(seed),
            export=sample_snapshot(seed=seed, steps=32, instruments=1),
        )
        members.append(create_snapshot_member(source, part))
    update = {"split_group": market_seed_group(12)} if wrong_seed else {"run_id": "sample:9:"}
    members[-1] = members[-1].model_copy(update=update)
    old = DatasetPayload(dataset_id="historical", members=tuple(members)).model_dump(mode="json")
    (tmp_path / "train").mkdir()
    write_json(tmp_path / "train" / "dataset.json", {**old, "content_sha256": content_hash(old)})
    large = import_module("train_large_direction")
    for fit in (mixed.fit, large.train):
        with pytest.raises(ValueError, match=message):
            fit(tmp_path, 1)
