"""Architecture discovery, strict settings, and worker-only availability checks."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from pydantic import ConfigDict, JsonValue, ValidationError

from regimex.contracts import StrictModel
from regimex.model_registry import (
    ArchitectureDescriptor,
    ArchitectureInfo,
    ArchitectureRegistry,
    InferenceRuntime,
    LogisticParameters,
    RegistryError,
    TCNParameters,
    TrainingPolicy,
    TrainingRuntime,
    WorkerCapability,
    baseline_descriptor,
    tcn_descriptor,
)

PLUGIN_SOURCE = '''"""Small installed test plugin with a real stdlib runtime probe."""
import json

from pydantic import Field

from regimex.contracts import StrictModel
from regimex.model_registry import ArchitectureDescriptor, TrainingPolicy

probe_calls = []


class PluginParameters(StrictModel):
    scale: float = Field(default=1.0, gt=0)


def probe(device):
    probe_calls.append(device)
    if device != "cpu":
        raise ValueError("this test plugin requires CPU")
    assert json.JSONDecoder().decode("[1, 2]") == [1, 2]


def descriptor():
    return ArchitectureDescriptor(
        architecture_id="test-json",
        implementation_version="1.0.0",
        parameters_model=PluginParameters,
        default_training_policy=TrainingPolicy(optimizer="lbfgs"),
        supported_optimizers=("lbfgs",),
        required_observation_fields=("last_price",),
        dependency_modules=("json",),
        backend_factory="json:JSONDecoder",
        worker_probe="regimex_test_plugin:probe",
    )
'''


@pytest.fixture
def installed_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "regimex_test_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    distribution = tmp_path / "regimex_test_plugin-1.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: regimex-test-plugin\nVersion: 1.0.0\n", encoding="utf-8"
    )
    (distribution / "entry_points.txt").write_text(
        "[regimex.models]\ntest-json = regimex_test_plugin:descriptor\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Each test receives a newly imported module and therefore a fresh probe log.
    monkeypatch.delitem(sys.modules, "regimex_test_plugin", raising=False)
    return tmp_path


def changed_descriptor(**changes: Any) -> ArchitectureDescriptor:
    values = baseline_descriptor().model_dump()
    values["parameters_model"] = LogisticParameters
    values.update(backend_factory=None, worker_probe=None, unavailable_reason="fixture unavailable")
    values.update(changes)
    return ArchitectureDescriptor.model_validate(values)


def test_real_entry_point_discovery_is_lazy(installed_plugin: Path) -> None:
    registry = ArchitectureRegistry.discover()
    descriptor = registry.get("test-json")
    plugin = import_module("regimex_test_plugin")
    assert plugin.probe_calls == []
    assert descriptor.describe().executable is None
    assert descriptor.validate_parameters({"scale": 2.0}).model_dump() == {"scale": 2.0}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        descriptor.validate_parameters({"dropout": 0.3})


def test_worker_probe_and_lazy_factory(installed_plugin: Path) -> None:
    registry = ArchitectureRegistry.discover()
    runtime = InferenceRuntime(threads=1)
    capability = registry.check_capability_on_worker("test-json", runtime, worker_id="analysis-1")
    assert capability.executable
    assert capability.reasons == ()
    assert capability.checked_dependency_modules == ("json",)
    assert capability.worker_id == "analysis-1"
    assert capability.checked_at.tzinfo is not None
    assert WorkerCapability.model_validate_json(capability.model_dump_json()) == capability
    factory = registry.load_backend_factory_on_worker("test-json", runtime, worker_id="analysis-1")
    decoder = factory()
    assert isinstance(decoder, json.JSONDecoder)
    assert decoder.decode("[0, null]") == [0, None]
    assert import_module("regimex_test_plugin").probe_calls == ["cpu", "cpu"]


def test_startup_imports_no_scientific_libraries(installed_plugin: Path) -> None:
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

blocked = {"numpy", "sklearn", "torch", "matplotlib", "plotly", "threadpoolctl"}


class RejectScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("startup attempted scientific import: " + fullname)


sys.meta_path.insert(0, RejectScientificImports())
from regimex.model_registry import ArchitectureRegistry, baseline_descriptor, tcn_descriptor
assert baseline_descriptor().describe().executable is None
assert tcn_descriptor().describe().executable is None
registry = ArchitectureRegistry.discover()
assert registry.get("test-json").describe().executable is None
assert not blocked.intersection(sys.modules)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(installed_plugin), source_root))},
    )


def test_duplicate_architecture_ids_are_rejected(installed_plugin: Path) -> None:
    distribution = installed_plugin / "duplicate_plugin-1.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: duplicate-plugin\nVersion: 1.0.0\n", encoding="utf-8"
    )
    (distribution / "entry_points.txt").write_text(
        "[regimex.models]\nduplicate = regimex_test_plugin:descriptor\n", encoding="utf-8"
    )
    with pytest.raises(RegistryError, match="duplicate architecture ID 'test-json'"):
        ArchitectureRegistry.discover()


@pytest.mark.parametrize("replacement", ["42", "lambda: {}", "lambda: 3"])
def test_invalid_installed_factories_fail_discovery(
    installed_plugin: Path, replacement: str
) -> None:
    module_path = installed_plugin / "regimex_test_plugin.py"
    module_path.write_text(f"descriptor = {replacement}\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="invalid architecture entry point 'test-json'"):
        ArchitectureRegistry.discover()


def test_import_failure_is_an_explicit_discovery_error(installed_plugin: Path) -> None:
    (installed_plugin / "regimex_test_plugin.py").write_text(
        "raise ImportError('missing plugin dependency')\n", encoding="utf-8"
    )
    with pytest.raises(RegistryError, match="missing plugin dependency"):
        ArchitectureRegistry.discover()


@pytest.mark.parametrize(
    "changes",
    [
        {"descriptor_version": "2.0"},
        {"interface_version": "2.0"},
        {"implementation_version": ""},
        {"architecture_id": "upload/python.py"},
        {"required_observation_fields": ("reference_boundaries",)},
        {"required_observation_fields": ("last_price", "last_price")},
        {"supported_devices": ("cuda",)},
        {"supported_optimizers": ("adamw",)},
        {"backend_factory": "uploaded_code()"},
        {"backend_factory": "json:JSONDecoder", "unavailable_reason": None},
        {"worker_probe": "json:JSONDecoder"},
        {"unavailable_reason": None},
    ],
)
def test_invalid_descriptor_contracts(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        changed_descriptor(**changes)


def test_parameter_schemas_cannot_relax_strict_validation() -> None:
    class LenientParameters(StrictModel):
        model_config = ConfigDict(extra="ignore")

    with pytest.raises(ValidationError, match="parameter schemas must forbid extras"):
        changed_descriptor(parameters_model=LenientParameters)


def test_registry_revalidates_constructed_descriptors() -> None:
    descriptor = baseline_descriptor().model_copy(update={"interface_version": "2.0"})
    with pytest.raises(RegistryError, match="invalid architecture descriptor"):
        ArchitectureRegistry((descriptor,))


def test_registry_rejects_unknown_architectures_and_duplicates() -> None:
    with pytest.raises(RegistryError, match="duplicate architecture ID 'baseline'"):
        ArchitectureRegistry((baseline_descriptor(), baseline_descriptor()))
    with pytest.raises(RegistryError, match="unknown installed architecture 'other'"):
        ArchitectureRegistry((baseline_descriptor(),)).get("other")


def test_builtin_descriptors_distinguish_installed_from_worker_verified() -> None:
    registry = ArchitectureRegistry((baseline_descriptor(), tcn_descriptor()))
    assert [descriptor.architecture_id for descriptor in registry.descriptors] == [
        "baseline",
        "tcn",
    ]
    baseline = registry.get("baseline")
    assert baseline.describe().executable is None
    assert baseline.backend_factory == "regimex.backends.baseline:create_backend"
    assert baseline.worker_probe == "regimex.backends.baseline:probe_runtime"
    assert baseline.default_training_policy.seeds == (11,)
    for descriptor in (registry.get("tcn"),):
        info = descriptor.describe()
        assert info.executable is None
        assert info.availability_reason == "execution must be checked on the worker"
        assert descriptor.backend_factory == "regimex.backends.tcn:create_backend"
        assert descriptor.worker_probe == "regimex.backends.tcn:probe_runtime"
        assert ArchitectureInfo.model_validate_json(info.model_dump_json()) == info


def test_baseline_factory_is_executable_only_after_worker_probe() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("threadpoolctl")
    from regimex.model_contracts import ModelBackend

    registry = ArchitectureRegistry((baseline_descriptor(),))
    runtime = InferenceRuntime(threads=1)
    assert registry.check_capability_on_worker("baseline", runtime, worker_id="test").executable
    backend = registry.load_backend_factory_on_worker("baseline", runtime, worker_id="test")()
    assert isinstance(backend, ModelBackend)
    assert backend.architecture_id == "baseline"


def test_tcn_factory_is_executable_only_after_worker_probe() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from regimex.model_contracts import ModelBackend

    registry = ArchitectureRegistry((tcn_descriptor(),))
    runtime = InferenceRuntime(threads=1)
    assert registry.check_capability_on_worker("tcn", runtime, worker_id="test").executable
    backend = registry.load_backend_factory_on_worker("tcn", runtime, worker_id="test")()
    assert isinstance(backend, ModelBackend)
    assert backend.architecture_id == "tcn"
    assert backend.implementation_version == registry.get("tcn").implementation_version == "0.2.0"


@pytest.mark.parametrize(
    "model,parameters",
    [
        (LogisticParameters, {"channels": 32}),
        (LogisticParameters, {"regularization": [0.0]}),
        (LogisticParameters, {"regularization": [1.0, 1.0]}),
        (LogisticParameters, {"regularization": ["1.0"]}),
        (LogisticParameters, {"context_sizes": [1]}),
        (LogisticParameters, {"context_sizes": [4, 4]}),
        (LogisticParameters, {"context_sizes": [4097]}),
        (TCNParameters, {"regularization": [1.0]}),
        (TCNParameters, {"dilations": [4097]}),
        (TCNParameters, {"dilations": list(range(1, 34))}),
        (TCNParameters, {"dropout": 1.0}),
        (TCNParameters, {"channels": True}),
        (TCNParameters, {"dilations": [1, 1]}),
        (TCNParameters, {"dilations": [0]}),
        (TCNParameters, {"smoothing": float("inf")}),
    ],
)
def test_architecture_parameter_isolation(
    model: type[StrictModel], parameters: dict[str, JsonValue]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(parameters))


def test_architecture_parameters_accept_json_arrays_without_scalar_coercion() -> None:
    descriptor = baseline_descriptor()
    assert descriptor.validate_parameters({"regularization": [0.5, 2.0]}) == LogisticParameters(
        regularization=(0.5, 2.0)
    )
    assert descriptor.describe().parameter_schema["additionalProperties"] is False
    with pytest.raises(RegistryError, match="does not support optimizer 'adamw'"):
        descriptor.validate_training_policy(TrainingPolicy(optimizer="adamw"))


@pytest.mark.parametrize("runtime_model", [TrainingRuntime, InferenceRuntime])
def test_runtime_profiles_reject_unsupported_device_and_invalid_threads(
    runtime_model: type[TrainingRuntime] | type[InferenceRuntime],
) -> None:
    for values in ({"device": "cuda"}, {"threads": 0}, {"threads": True}):
        with pytest.raises(ValidationError):
            runtime_model.model_validate(values)
    assert TrainingRuntime().threads == 4
    assert InferenceRuntime().threads == 2


def test_training_policy_keeps_runtime_and_architecture_settings_separate() -> None:
    for values in (
        {"optimizer": "lbfgs", "threads": 4},
        {"optimizer": "adamw", "channels": 32},
        {"optimizer": "adamw", "seeds": [11, 11]},
        {"optimizer": "adamw", "selection_metric": "test_segment_macro_f1"},
    ):
        with pytest.raises(ValidationError):
            TrainingPolicy.model_validate_json(json.dumps(values))


def test_observation_capabilities_are_checked_without_treating_null_as_zero() -> None:
    descriptor = tcn_descriptor()
    descriptor.validate_observation_fields(("last_price", "trade_count"))
    with pytest.raises(RegistryError, match="requires observation fields: last_price"):
        descriptor.validate_observation_fields(("trade_count",))


def test_missing_worker_dependency_is_explicit(installed_plugin: Path) -> None:
    descriptor = ArchitectureRegistry.discover().get("test-json")
    values = descriptor.model_dump()
    values["parameters_model"] = descriptor.parameters_model
    values["dependency_modules"] = ("regimex_missing_optional_dependency_for_test",)
    registry = ArchitectureRegistry((ArchitectureDescriptor.model_validate(values),))
    result = registry.check_capability_on_worker(
        "test-json", InferenceRuntime(), worker_id="analysis-2"
    )
    assert not result.executable
    assert "cannot be imported" in result.reasons[0]
    assert "regimex_missing_optional_dependency_for_test" in result.reasons[0]
    assert import_module("regimex_test_plugin").probe_calls == []


@pytest.mark.parametrize(
    "probe_source,expected",
    [
        ("raise RuntimeError('CPU runtime unavailable')", "CPU runtime unavailable"),
        ("return True", "must return None on success"),
    ],
)
def test_worker_runtime_probe_must_actually_succeed(
    installed_plugin: Path, monkeypatch: pytest.MonkeyPatch, probe_source: str, expected: str
) -> None:
    plugin = import_module("regimex_test_plugin")
    namespace: dict[str, Any] = {}
    exec(f"def failing_probe(device):\n    {probe_source}\n", namespace)
    monkeypatch.setattr(plugin, "probe", namespace["failing_probe"])
    result = ArchitectureRegistry.discover().check_capability_on_worker(
        "test-json", TrainingRuntime(), worker_id="training-2"
    )
    assert not result.executable
    assert expected in result.reasons[0]


def test_worker_rejects_noncallable_factory(installed_plugin: Path) -> None:
    descriptor = ArchitectureRegistry.discover().get("test-json")
    values = descriptor.model_dump()
    values["parameters_model"] = descriptor.parameters_model
    values["backend_factory"] = "json:__version__"
    registry = ArchitectureRegistry((ArchitectureDescriptor.model_validate(values),))
    result = registry.check_capability_on_worker(
        "test-json", TrainingRuntime(), worker_id="training-3"
    )
    assert not result.executable
    assert "is not callable" in result.reasons[0]
