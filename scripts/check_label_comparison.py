"""Compare interval scores on the outputs of check_execution_label_quality.py."""

import argparse
import json
from pathlib import Path

import numpy as np
from check_execution_label_quality import LENGTHS, SHAPES
from check_execution_labels import METHODS
from confirm_trend_pauses import STRESS

from regimex.directions import DIRECTION_NAMES
from regimex.storage import json_bytes


def macro_f1(counts):
    matched, extra, missed = np.moveaxis(counts, -1, 0)
    total = 2 * matched + extra + missed
    return np.divide(
        2 * matched, total, out=np.zeros_like(total, dtype=float), where=total > 0
    ).mean(-1)


def compare(files, method, reference):
    panels = {}
    for path in files:
        panel = json.loads(path.read_bytes())
        size = panel["steps"]
        if size in panels or panel["class_order"] != list(DIRECTION_NAMES):
            raise ValueError(f"Repeated panel size or incompatible direction order in {path}")
        panels[size] = panel
    sizes = (*map(str, LENGTHS), "stress")
    if set(panels) != set(sizes):
        raise ValueError("Supply one input for each of 256, 512, 1024 and stress")
    random = np.random.default_rng(253071)
    seen = set()
    cells = {}
    for size in sizes:
        panel = panels[size]
        groups = {}
        for row in panel["records"]:
            key = row["shape"], row["view"]
            group = groups.setdefault(row["seed"], {})
            if key in group:
                raise ValueError(f"Repeated shape and view in seed {row['seed']}")
            group[key] = row["counts"]
        if seen.intersection(groups) or len(groups) != panel["groups"] or not groups:
            raise ValueError(f"Repeated or missing seed groups in {size}")
        seen.update(groups)
        views = ["stress"] if size == "stress" else ["complete", "masked"]
        if size == "stress" and any(
            "dense" in [view for _, view in group] for group in groups.values()
        ):
            views += ["dense", "sparse"]
        shapes = STRESS if size == "stress" else SHAPES
        expected = {(shape, view) for shape in shapes for view in views}
        if any(group.keys() != expected for group in groups.values()):
            raise ValueError(f"Incomplete seed group in {size}")
        # The same bootstrap draw applies to every shape, view and method within a seed.
        n = len(groups)
        weights = random.multinomial(n, np.full(n, 1 / n), size=5000)
        for view in views:
            estimates = []
            for name in (method, reference):
                counts = np.array(
                    [
                        sum(np.array(groups[s][shape, view][name]) for shape in shapes)
                        for s in sorted(groups)
                    ]
                )
                resampled = (weights @ counts.reshape(n, 9)).reshape(-1, 3, 3)
                estimates.append(np.r_[macro_f1(counts.sum(0)), macro_f1(resampled)])
            cells[f"{size}/{view}"] = np.stack(estimates)
    aggregates = {
        size: np.mean([cells[f"{size}/{view}"] for view in ("complete", "masked")], axis=0)
        for size in sizes[:-1]
    }
    aggregates["known_drift"] = np.mean(list(aggregates.values()), axis=0)
    results = {}
    for key, (selected, previous) in (cells | aggregates).items():
        delta = selected - previous
        results[key] = {
            method: float(selected[0]),
            reference: float(previous[0]),
            "difference": float(delta[0]),
            "paired_95_percent": np.quantile(delta[1:], [0.025, 0.975]).tolist(),
        }
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panels", nargs=4, type=Path)
    cli.add_argument("--method", choices=METHODS, default="combined")
    cli.add_argument("--reference", choices=METHODS, default="current")
    args = cli.parse_args()
    if args.method == args.reference:
        cli.error("Select two distinct methods")
    print(json_bytes(compare(args.panels, args.method, args.reference)).decode())


if __name__ == "__main__":
    main()
