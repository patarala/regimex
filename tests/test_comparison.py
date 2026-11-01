"""A comparison cannot mix data, reference conventions or model identities."""

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from regimex.comparison import render_model_comparison
from regimex.directions import Direction
from regimex.storage import digest


def evaluation(model: str) -> dict[str, JsonValue]:
    pytest.importorskip("numpy")
    from test_evaluation import intervals, rows, sequences

    from regimex.evaluation import report

    reference = intervals(
        rows(96),
        [(0, 32, Direction.SIDEWAYS), (32, 64, Direction.GROWTH), (64, 96, Direction.DECLINE)],
    )
    value = report(sequences((reference,)), (reference,), bootstrap_repetitions=20)
    value.update(
        scope="fixed_model_validation",
        model_version_id=model,
        bundle_sha256="1" * 64 if model == "baseline" else "2" * 64,
        dataset_sha256="3" * 64,
        reference_sha256="4" * 64,
        reference_policy="fixture-directions/1.0",
        training_reference_kind="automatic",
        training_reference_policy="fixture-directions/1.0",
        training_reference_sha256="7" * 64,
        validation_selection_reference_sha256="4" * 64,
        analysis_profile_sha256="5" * 64,
        decoder_version="duration-viterbi/1.0",
        inference_runtime={"device": "cpu", "threads": 1},
    )
    return value


def test_comparison_preserves_downloads_and_escapes_embedded_script(tmp_path: Path) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    for value in (first, second):
        details = cast(list[dict[str, JsonValue]], value["experiment_statistics"])
        details[0]["experiment_id"] = "</script><script>alert(1)</script>"
    target = tmp_path / "comparison"
    render_model_comparison((first, second), target)
    assert json.loads((target / "evaluations.json").read_text()) == [first, second]
    html = (target / "index.html").read_text()
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html
    manifest = json.loads((target / "report-manifest.json").read_text())
    assert manifest["files"] == {
        name: digest(target / name)
        for name in (
            "index.html",
            "comparison.js",
            "comparison.css",
            "evaluations.json",
            "paired-comparisons.json",
        )
    }
    assert manifest["version"] == "model-comparison/2.0"
    paired = json.loads((target / "paired-comparisons.json").read_text())
    assert paired[0]["reference_model_version_id"] == "baseline"
    assert paired[0]["candidate_model_version_id"] == "tcn"
    assert paired[0]["metrics"]["segment_macro_f1"]["percentile_interval"] == [0, 0]
    assert not list(target.glob("*.png"))


@pytest.mark.parametrize("scope", ["fixed_model_validation", "controlled_final_benchmark"])
def test_displayed_selection_caveat_matches_evaluation_scope(tmp_path: Path, scope: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped report controller")
    first, second = evaluation("baseline"), evaluation("tcn")
    first["scope"] = second["scope"] = scope
    target = tmp_path / "comparison"
    render_model_comparison((first, second), target)
    script = r"""
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const directory = process.argv[1];
const html = fs.readFileSync(path.join(directory, "index.html"), "utf8");
const expression = /<script id="comparison-data" type="application\/json">([\s\S]*?)<\/script>/;
const payload = html.match(expression)[1];
class Element {
  constructor() { this.textContent = ""; this.value = ""; }
  append() {}
  replaceChildren() {}
  addEventListener() {}
  createTHead() { return new Element(); }
  createTBody() { return new Element(); }
  insertRow() { return new Element(); }
  insertCell() { return new Element(); }
}
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
get("comparison-data").textContent = payload;
get("paired-metric").value = "segment_macro_f1";
get("metric-kind").value = "segment_metrics";
get("experiment").value = JSON.parse(payload).evaluations[0].experiment_statistics[0].experiment_id;
vm.runInNewContext(fs.readFileSync(path.join(directory, "comparison.js"), "utf8"), {
  document: {getElementById: get, createElement: () => new Element()}, Node: Element,
});
console.log(JSON.stringify({
  population: get("population").textContent, note: get("paired-note").textContent,
}));
"""
    executed = subprocess.run(
        [node, "-e", script, str(target)], capture_output=True, text=True, check=True, timeout=10
    )
    displayed = json.loads(executed.stdout)
    assert "do not include training-seed variation" in displayed["note"]
    if scope == "fixed_model_validation":
        assert "validation" in displayed["population"]
        assert "Reused selection data" in displayed["note"]
        assert "not used for selection" not in displayed["note"]
    else:
        assert "final benchmark" in displayed["population"]
        assert "fixed before opening" in displayed["note"]
        assert "not used for selection" in displayed["note"]
        assert "Reused selection data" not in displayed["note"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dataset_sha256", "6" * 64),
        ("reference_sha256", "6" * 64),
        ("analysis_profile_sha256", "6" * 64),
        ("reference_policy", "fixture-directions/2.0"),
        ("scope", "controlled_final_benchmark"),
        ("decoder_version", "duration-viterbi/2.0"),
    ],
)
def test_comparison_rejects_incomparable_evaluations(tmp_path: Path, key: str, value: str) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    second[key] = value
    target = tmp_path / "comparison"
    with pytest.raises(ValueError, match="same dataset, references and analysis policy"):
        render_model_comparison((first, second), target)
    assert not target.exists()


def test_comparison_rejects_missing_uncertainty_and_different_experiments(tmp_path: Path) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    no_uncertainty = {**second, "confidence_intervals": None}
    with pytest.raises(ValueError, match="confidence_intervals"):
        render_model_comparison((first, no_uncertainty), tmp_path / "missing")
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details[0]["experiment_id"] = "different-experiment"
    with pytest.raises(ValueError, match="populations"):
        render_model_comparison((first, second), tmp_path / "different")


def test_comparison_rejects_duplicate_model_and_invalid_score(tmp_path: Path) -> None:
    first = evaluation("baseline")
    with pytest.raises(ValueError, match="unique"):
        render_model_comparison((first, copy.deepcopy(first)), tmp_path / "duplicate")
    second = evaluation("tcn")
    second["segment_macro_f1"] = float("nan")
    with pytest.raises(ValueError):
        render_model_comparison((first, second), tmp_path / "invalid")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("segment_metrics", "iou_threshold"), 0.75),
        (("segment_metrics", "per_class", "growth", "precision"), 0.0),
        (("boundary_metrics", "matched"), 999),
        (("boundary_metrics", "defined"), False),
        (("coverage", "predicted_classified_step_count"), 100000),
        (("coverage", "predicted_classified_fraction"), 0.2),
        (("fragmentation", "predicted_to_reference_interval_ratio"), 5.0),
    ],
)
def test_comparison_rejects_inconsistent_metrics(
    tmp_path: Path, path: tuple[str, ...], value: JsonValue
) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    current = second
    for key in path[:-1]:
        current = cast(dict[str, JsonValue], current[key])
    current[path[-1]] = value
    with pytest.raises(ValueError):
        render_model_comparison((first, second), tmp_path / "invalid")


def test_experiment_scores_cannot_disagree_with_saved_counts(tmp_path: Path) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details[0]["segment_macro_f1"] = 0.0
    with pytest.raises(ValueError, match="experiment F1 disagrees"):
        render_model_comparison((first, second), tmp_path / "invalid")


def test_reference_support_cannot_be_swapped_between_experiment_ids(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from test_evaluation import intervals, rows, sequences

    from regimex.evaluation import report

    mixed = intervals(
        rows(96),
        [(0, 32, Direction.SIDEWAYS), (32, 64, Direction.GROWTH), (64, 96, Direction.DECLINE)],
    )
    growth = intervals(rows(96), [(0, 96, Direction.GROWTH)])
    metrics = report(sequences((mixed, growth)), (mixed, growth), bootstrap_repetitions=20)
    first, second = evaluation("baseline"), evaluation("tcn")
    first.update(copy.deepcopy(metrics))
    second.update(copy.deepcopy(metrics))
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details[0]["experiment_id"], details[1]["experiment_id"] = (
        details[1]["experiment_id"],
        details[0]["experiment_id"],
    )
    with pytest.raises(ValueError, match="populations differ within an experiment"):
        render_model_comparison((first, second), tmp_path / "invalid")


def test_comparison_rejects_old_metrics_and_different_reference_grids(tmp_path: Path) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    second["evaluation_version"] = "2.1"
    with pytest.raises(ValueError, match="evaluation_version"):
        render_model_comparison((first, second), tmp_path / "old")
    second = evaluation("tcn")
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details[0]["reference_grid_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="populations differ within an experiment"):
        render_model_comparison((first, second), tmp_path / "different-grid")


def test_comparison_checks_typed_alignment_against_experiment_lags(tmp_path: Path) -> None:
    first, second = evaluation("baseline"), evaluation("tcn")
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details[0]["typed_boundary_lags"] = [3, 3]
    with pytest.raises(ValueError, match="lag summary"):
        render_model_comparison((first, second), tmp_path / "invalid")


def test_paired_download_is_invariant_to_experiment_order_and_negates_on_swap(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    from test_evaluation import intervals, rows, sequences

    from regimex.evaluation import report

    observations = rows(96)
    reference = intervals(
        observations,
        [
            (0, 32, Direction.SIDEWAYS),
            (32, 64, Direction.GROWTH),
            (64, 96, Direction.DECLINE),
        ],
    )
    wrong = intervals(observations, [(0, 96, Direction.GROWTH)])
    first, second = evaluation("baseline"), evaluation("tcn")
    data = sequences((reference, reference, reference))
    first.update(report(data, (wrong, reference, reference), bootstrap_repetitions=101))
    second.update(report(data, (reference, reference, reference), bootstrap_repetitions=101))
    details = cast(list[dict[str, JsonValue]], second["experiment_statistics"])
    details.reverse()
    render_model_comparison((first, second), tmp_path / "forward")
    render_model_comparison((second, first), tmp_path / "backward")
    forward = json.loads((tmp_path / "forward/paired-comparisons.json").read_text())[0]
    backward = json.loads((tmp_path / "backward/paired-comparisons.json").read_text())[0]
    for name, item in forward["metrics"].items():
        swapped = backward["metrics"][name]
        assert swapped["delta"] == -item["delta"]
        assert swapped["percentile_interval"] == [
            -item["percentile_interval"][1],
            -item["percentile_interval"][0],
        ]
        assert item["improved_experiments"] == 1 and item["tied_experiments"] == 2
