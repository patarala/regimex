"""Compare fresh inference processes with one process that restores a bundle once."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from regimex.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
TCN_MANIFEST = "fa222148503cb4b5e3626834dfee7e04baa8939f792b9708109ce34866716506"
SEEDS = (42, 43, 44, 45)


def elapsed_ms(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1_000_000


def predict(bundle: Path, digest: str, exports: list[Path]) -> None:
    from regimex.application.analysis_computation import restore_bundle, snapshot_inputs
    from regimex.evaluation import probabilities_valid
    from regimex.exports import parse_export_v2
    from regimex.model_registry import ArchitectureRegistry, InferenceRuntime

    start = time.perf_counter_ns()
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    runtime = InferenceRuntime(threads=1)
    predictor, _ = restore_bundle(
        bundle,
        digest,
        manifest["model_version_id"],
        manifest["architecture_id"],
        runtime,
        ArchitectureRegistry.discover(),
        worker_id="offline-benchmark",
    )
    restoration = elapsed_ms(start)
    samples = []
    for path in exports:
        start = time.perf_counter_ns()
        snapshot = parse_export_v2(path.read_bytes())
        inputs = snapshot_inputs(snapshot)
        prepared = elapsed_ms(start)
        start = time.perf_counter_ns()
        values = predictor.predict(inputs, runtime=runtime)
        prediction = elapsed_ms(start)
        probabilities_valid(values, inputs.row_count)
        samples.append(
            {
                "snapshot_sha256": snapshot.content_sha256,
                "probabilities_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
                "rows": inputs.row_count,
                "prepare_ms": prepared,
                "predict_ms": prediction,
            }
        )
    result = {"restore_ms": restoration, "jobs": samples}
    if sys.platform == "linux":
        import resource

        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(json.dumps(result))


def measure(bundle: Path, manifest_sha256: str, exports: list[Path]) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = time.perf_counter_ns()
    output = subprocess.check_output(
        [sys.executable, __file__, "--predict", bundle, manifest_sha256, *exports],
        env=env,
    )
    return {"wall_ms": elapsed_ms(start), **json.loads(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "release/models/tcn")
    parser.add_argument("--manifest-sha256", default=TCN_MANIFEST)
    args = parser.parse_args()
    manifest_bytes = (args.bundle / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        parser.error("supply the expected SHA-256 of the bundle manifest")

    from regimex.market_export import sample_snapshot

    with tempfile.TemporaryDirectory() as directory:
        exports = []
        for seed in SEEDS:
            snapshot = sample_snapshot(seed=seed, steps=512, instruments=2)
            path = Path(directory) / f"{seed}.json"
            write_json(path, snapshot.model_dump(mode="json"))
            exports.append(path)
        fresh = [measure(args.bundle, args.manifest_sha256, [path]) for path in exports]
        reused = measure(args.bundle, args.manifest_sha256, exports)
    for one, same in zip(fresh, reused["jobs"], strict=True):
        for field in ("snapshot_sha256", "rows", "probabilities_sha256"):
            if one["jobs"][0][field] != same[field]:
                raise ValueError(f"Different {field} for {same['snapshot_sha256']}")
    result = {
        "model_version": json.loads(manifest_bytes)["model_version_id"],
        "manifest_sha256": args.manifest_sha256,
        "threads": 1,
        "seeds": SEEDS,
        "steps": 512,
        "instruments": 2,
        "fresh_processes": fresh,
        "reused_process": reused,
        "fresh_total_wall_ms": sum(p["wall_ms"] for p in fresh),
        "fresh_median_wall_ms": statistics.median(p["wall_ms"] for p in fresh),
        "probabilities_bitwise_equal": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--predict"]:
        predict(Path(sys.argv[2]), sys.argv[3], [Path(p) for p in sys.argv[4:]])
    else:
        main()
