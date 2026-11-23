"""A probability mixture of two independently verified, already fitted direction forests."""

import shutil
import struct
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from pydantic import JsonValue, field_validator

from regimex.backends.adaptive import (
    ARCHITECTURE_ID,
    FEATURE_VERSION,
    IMPLEMENTATION_VERSION,
    PREPROCESSING_VERSION,
    AdaptiveBackend,
    AdaptivePredictor,
    _read_json,
)
from regimex.contracts import StrictModel
from regimex.data_catalog import content_hash
from regimex.directions import DECODER_VERSION, DIRECTION_NAMES, AnalysisConfig
from regimex.evaluation import probabilities_valid
from regimex.features import DoubleArray
from regimex.model_bundles import (
    BundleMember,
    BundleRole,
    ModelBundle,
    ParentModelEvidence,
    TrainingEvidence,
    verify_model_bundle,
)
from regimex.model_contracts import BundleProvenance, ObservationInputs, require_observation_inputs
from regimex.model_registry import InferenceRuntime
from regimex.operations import Sha256
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import digest, json_bytes, output_directory, write_json

BLEND_VERSION = "adaptive-probability-blend/1.0"
FRACTIONS = (0.125, 0.25, 0.375, 0.5)
_SOURCES = ("greedy", "multiscale")
_FEATURES = (FEATURE_VERSION, "adaptive-multiscale/1.0")
_FILES: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "preprocessing": "preprocessing.json",
    "weights": "forests.zip",
    "validation": "validation.json",
    "history": "history.json",
}
_PARENT_FILES = (
    "manifest.json",
    "configuration.json",
    "preprocessing.json",
    "weights.npz",
    "validation.json",
    "history.json",
)
_END = struct.Struct("<4s4H2IH")


class BlendConfiguration(StrictModel):
    codec_version: Literal["adaptive-probability-blend/1.0"] = "adaptive-probability-blend/1.0"
    greedy_sha256: Sha256
    multiscale_sha256: Sha256
    fraction: float

    @field_validator("fraction")
    @classmethod
    def supported_fraction(cls, value: float) -> float:
        if value not in FRACTIONS:
            raise ValueError("unsupported multiscale probability fraction")
        return value


def blend_environment(
    greedy_sha256: str, multiscale_sha256: str, fraction: float
) -> dict[str, JsonValue]:
    """Identify the two original environments and the operation used to combine them."""
    config = BlendConfiguration(
        greedy_sha256=greedy_sha256, multiscale_sha256=multiscale_sha256, fraction=fraction
    )
    return config.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class BlendedAdaptivePredictor:
    greedy: AdaptivePredictor
    multiscale: AdaptivePredictor
    fraction: float
    architecture_id: str = ARCHITECTURE_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    @property
    def analysis(self) -> AnalysisConfig:
        return self.greedy.analysis

    @property
    def numeric_bytes(self) -> int:
        return self.greedy.numeric_bytes + self.multiscale.numeric_bytes

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        require_observation_inputs(inputs)
        InferenceRuntime.model_validate(runtime.model_dump())
        if self.numeric_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("both forests must fit in the inference model cache")
        result = np.empty((inputs.row_count, len(DIRECTION_NAMES)), dtype=np.float64)
        first = 0
        for rows in inputs.observations:
            stop = first + len(rows)
            # Keep only one instrument's component predictions alongside the final output.
            result[first:stop] = self.greedy.predict_sequence(rows) * (1.0 - self.fraction)
            result[first:stop] += self.multiscale.predict_sequence(rows) * self.fraction
            first = stop
        probabilities_valid(result, inputs.row_count)
        return result


@contextmanager
def _open_forests(archive: Path) -> Iterator[zipfile.ZipFile]:
    expected = {f"{source}/{name}" for source in _SOURCES for name in _PARENT_FILES}
    with archive.open("rb") as stream:
        size = stream.seek(0, 2)
        if size < _END.size:
            raise ValueError("incomplete adaptive blend archive")
        # zipfile reads the entire central directory when it opens the archive. A fixed
        # member count alone would not bound that first allocation on an untrusted file.
        stream.seek(size - _END.size)
        signature, disk, directory_disk, entries, total, length, offset, comment = _END.unpack(
            stream.read(_END.size)
        )
        if (
            signature != b"PK\x05\x06"
            or disk
            or directory_disk
            or comment
            or total != len(expected)
            or entries != total
            or length > 4096
            or offset + length + _END.size != size
        ):
            raise ValueError("invalid directory of adaptive blend archive")
        # ZIP64 records are unnecessary for 12 members and could override the bounds above.
        if size >= _END.size + 20:
            stream.seek(size - _END.size - 20)
            if stream.read(4) == b"PK\x06\x07":
                raise ValueError("adaptive blend archives do not use ZIP64")
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream) as packed:
                entries_info = packed.infolist()
                if (
                    len(entries_info) != len(expected)
                    or {e.filename for e in entries_info} != expected
                ):
                    raise ValueError("unexpected adaptive blend member names")
                for entry in entries_info:
                    maximum = 128 * 1024**2 if entry.filename.endswith("/weights.npz") else 1024**2
                    if (
                        entry.compress_type != zipfile.ZIP_STORED
                        or entry.flag_bits
                        or not 0 < entry.file_size <= maximum
                        or entry.compress_size != entry.file_size
                    ):
                        raise ValueError("unexpected adaptive blend member encoding or size")
                yield packed
        except (zipfile.BadZipFile, EOFError) as exc:
            raise ValueError("invalid adaptive blend archive") from exc


def _records(
    parents: tuple[ModelBundle, ModelBundle], training: TrainingEvidence
) -> dict[str, object]:
    return {
        "preprocessing.json": {
            "feature_versions": [p.feature_version for p in parents],
            "preprocessing_versions": [p.preprocessing_version for p in parents],
        },
        "history.json": {
            "operation": "combine_fitted_forests",
            "source_runs": [p.training.run_id for p in parents],
        },
        "validation.json": {
            "metric": training.selection_metric,
            "score": training.validation_segment_macro_f1,
            "references_sha256": training.validation_references.content_sha256,
        },
    }


def _parent(root: Path, manifest_sha: str, feature: str) -> ModelBundle:
    parent = verify_model_bundle(
        root,
        manifest_sha256=manifest_sha,
        architecture_id=ARCHITECTURE_ID,
        implementation_version=IMPLEMENTATION_VERSION,
    )
    if parent.feature_version != feature or parent.training.parent_models:
        raise ValueError("adaptive blend requires one greedy and one multiscale base forest")
    return parent


def _restore_component(
    packed: zipfile.ZipFile, source: str, sha: str, feature: str, runtime: InferenceRuntime
) -> tuple[ModelBundle, AdaptivePredictor]:
    # Stage only this parent's original files. Its temporary weights are removed
    # before the second forest is restored; the outer archive stays in the model cache.
    with TemporaryDirectory(prefix="regimex-forest-") as temporary:
        folder = Path(temporary)
        for name in _PARENT_FILES:
            with packed.open(f"{source}/{name}") as entry, (folder / name).open("xb") as target:
                shutil.copyfileobj(entry, target, length=128 * 1024)
        manifest = _parent(folder, sha, feature)
        return manifest, AdaptiveBackend()._restore_forest(folder, manifest, runtime)


def _lineage(
    parents: tuple[ModelBundle, ModelBundle],
    config: BlendConfiguration,
    training: TrainingEvidence,
    profile_sha: str,
) -> None:
    greedy, multi = parents
    evidence = (
        ParentModelEvidence.from_bundle(greedy, config.greedy_sha256),
        ParentModelEvidence.from_bundle(multi, config.multiscale_sha256),
    )
    if (
        training.parent_models != evidence
        or training.additional_training_sources
        or training.dataset_sha256 != multi.training.dataset_sha256
        or training.split_manifest_sha256 != multi.training.split_manifest_sha256
        or training.labels != multi.training.labels
        or training.seeds != multi.training.seeds
        or training.selected_seed != multi.training.selected_seed
        or training.validation_dataset_sha256 is None
        or any(p.analysis_profile_sha256 != profile_sha for p in parents)
    ):
        raise ValueError("adaptive blend provenance differs from the two fitted forests")


def pack_bundle(
    destination: Path,
    *,
    greedy_bundle: Path,
    greedy_sha256: str,
    multiscale_bundle: Path,
    multiscale_sha256: str,
    fraction: float,
    provenance: BundleProvenance,
) -> str:
    """Publish a standalone forest mixture and verify it can restore its original parents."""
    config = BlendConfiguration(
        greedy_sha256=greedy_sha256, multiscale_sha256=multiscale_sha256, fraction=fraction
    )
    parents = (
        _parent(greedy_bundle, greedy_sha256, _FEATURES[0]),
        _parent(multiscale_bundle, multiscale_sha256, _FEATURES[1]),
    )
    _lineage(parents, config, provenance.training, provenance.analysis_profile_sha256)
    if provenance.environment_sha256 != content_hash(config.model_dump(mode="json")):
        raise ValueError("adaptive blend environment differs from its sources or mixture")
    with output_directory(destination) as stage:
        write_json(stage / "configuration.json", config.model_dump(mode="json"))
        for name, record in _records(parents, provenance.training).items():
            write_json(stage / name, record)
        with zipfile.ZipFile(stage / "forests.zip", "x", allowZip64=False) as packed:
            for key, folder in zip(_SOURCES, (greedy_bundle, multiscale_bundle), strict=True):
                for name in _PARENT_FILES:
                    member = zipfile.ZipInfo(f"{key}/{name}")
                    with (folder / name).open("rb") as source, packed.open(member, "w") as target:
                        shutil.copyfileobj(source, target, length=128 * 1024)
        members = tuple(
            BundleMember(
                role=role,
                path=name,
                size_bytes=(stage / name).stat().st_size,
                sha256=digest(stage / name),
            )
            for role, name in _FILES.items()
        )
        manifest = ModelBundle(
            bundle_version="1.0",
            architecture_id=ARCHITECTURE_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            interface_version="1.0",
            observation_version="2.1",
            feature_version=BLEND_VERSION,
            preprocessing_version=PREPROCESSING_VERSION,
            decoder_version=DECODER_VERSION,
            class_names=("sideways", "growth", "decline"),
            members=members,
            **provenance.model_dump(),
        )
        write_json(stage / "manifest.json", manifest.model_dump(mode="json"))
        sha = digest(stage / "manifest.json")
        # Reopen the bytes actually copied, including both fitted weight codecs.
        AdaptiveBackend().load_bundle(stage, manifest_sha256=sha, runtime=InferenceRuntime())
        return sha


def load_blend(
    root: Path, bundle: ModelBundle, runtime: InferenceRuntime
) -> BlendedAdaptivePredictor:
    if (
        {m.role: m.path for m in bundle.members} != _FILES
        or bundle.decoder_version != DECODER_VERSION
        or bundle.preprocessing_version != PREPROCESSING_VERSION
    ):
        raise ValueError("adaptive blend codec or scientific contract differs")
    config = _read_json(root / "configuration.json", BlendConfiguration)
    if bundle.environment_sha256 != content_hash(config.model_dump(mode="json")):
        raise ValueError("adaptive blend environment differs from its sources or mixture")
    with _open_forests(root / "forests.zip") as packed:
        greedy_manifest, greedy = _restore_component(
            packed, "greedy", config.greedy_sha256, _FEATURES[0], runtime
        )
        remaining = runtime.model_cache_budget_bytes - greedy.numeric_bytes
        if remaining < 1:
            raise ResourceBudgetExceeded("both forests must fit in the inference model cache")
        multi_manifest, multi = _restore_component(
            packed,
            "multiscale",
            config.multiscale_sha256,
            _FEATURES[1],
            runtime.model_copy(update={"model_cache_budget_bytes": remaining}),
        )
    parents = (greedy_manifest, multi_manifest)
    _lineage(parents, config, bundle.training, bundle.analysis_profile_sha256)
    for name, record in _records(parents, bundle.training).items():
        expected = json_bytes(record)
        with (root / name).open("rb") as stream:
            if stream.read(len(expected) + 1) != expected:
                raise ValueError("adaptive blend selection record differs from its provenance")
    return BlendedAdaptivePredictor(greedy, multi, config.fraction)
