"""Explain each reference sideways interval on the frozen profile-2.1 validation split."""

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from compare_profile_21 import PROFILE, RUNTIME, Corpus
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from regimex.application.training import _references, observation_inputs
from regimex.backends.tcn import TCNBackend
from regimex.data_catalog import _load_partition
from regimex.interval_errors import Outcome, sideways_intervals
from regimex.model_bundles import ModelBundle
from regimex.report_licenses import license_notices
from regimex.segmentation import decode
from regimex.storage import digest, output_directory, write_json

COLORS = {
    "sideways": "#687789",
    "growth": "#158066",
    "decline": "#b84c34",
    "no_trades": "#ad9c72",
    "insufficient_data": "#b2b4be",
}
DURATION_BINS = ((1, 64), (64, 128), (128, 256), (256, None))


def state(interval):
    return interval.direction.value if interval.direction else interval.status


def label(outcome):
    return outcome.replace("_", " ").capitalize()


def chart(path, rows, references, predictions, selected, title, snapshot_id):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(
            x=[row.step + 0.5 for row in rows],
            y=[row.last_price for row in rows],
            name="Price",
            mode="lines",
            line_color="#364b61",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    for label, color in COLORS.items():
        bars = [
            (lane, item)
            for lane, items in (("Reference", references), ("TCN", predictions))
            for item in items
            if state(item) == label
        ]
        if bars:
            fig.add_trace(
                go.Bar(
                    name=label,
                    orientation="h",
                    y=[lane for lane, _ in bars],
                    base=[item.start for _, item in bars],
                    x=[item.stop - item.start for _, item in bars],
                    customdata=[[item.start, item.stop] for _, item in bars],
                    hovertemplate=(
                        "%{y} · [%{customdata[0]}, %{customdata[1]})<extra>%{fullData.name}</extra>"
                    ),
                    marker_color=color,
                    width=0.6,
                ),
                row=2,
                col=1,
            )
    margin = selected.stop - selected.start
    fig.update_xaxes(
        range=[max(0, selected.start - margin), min(len(rows), selected.stop + margin)]
    )
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=["TCN", "Reference"], row=2, col=1)
    for step in (selected.start, selected.stop):
        fig.add_vline(x=step, line_dash="dot", line_width=1, line_color="#333")
    fig.update_layout(
        title=escape(title),
        template="plotly_white",
        barmode="overlay",
        height=540,
        margin=dict(l=80, r=30, t=85, b=55),
        legend=dict(orientation="h", y=-0.18),
    )
    plot = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=path.stem,
        config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>{escape(title)}</title>'
        '<script src="plotly.min.js"></script><body style="font:1rem system-ui">'
        f'<a href="index.html">Sideways intervals</a> · {escape(snapshot_id)}'
        + plot
        + "</body></html>",
        encoding="utf-8",
    )


def counts(records):
    outcomes = Counter(item["outcome"] for item in records)
    total = len(records)
    return {
        "reference": total,
        "missed": total - outcomes[Outcome.MATCHED],
        "recall": outcomes[Outcome.MATCHED] / total if total else None,
        "outcomes": {kind.value: outcomes[kind] for kind in Outcome},
    }


def write_index(path, summary):
    by_outcome = {kind: [] for kind in Outcome}
    for item in summary["intervals"]:
        if item["chart"]:
            interval = f"{item['instrument_id']} · [{item['start']}, {item['stop']})"
            link = f'<a href="{item["chart"]}">{escape(interval)}</a>'
            by_outcome[item["outcome"]].append(f"<li>{link} · {escape(item['snapshot_id'])}</li>")
    sections = [
        f"<h2>{label(kind)} ({len(links)})</h2><ul>{''.join(links)}</ul>"
        for kind, links in by_outcome.items()
        if links
    ]
    durations = []
    for group in summary["by_reference_duration"]:
        start, stop = group["min_steps"], group["stop_before"]
        span = f"{start} to {stop - 1}" if stop else f"{start}+"
        durations.append(
            f"<tr><td>{span}</td><td>{group['reference']}</td>"
            f"<td>{group['outcomes']['matched']}</td><td>{group['missed']}</td></tr>"
        )
    totals = summary["total"]
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Sideways intervals</title>'
        "<style>body{font:1rem/1.6 system-ui;max-width:64rem;margin:2rem auto;padding:0 1rem}"
        "li{overflow-wrap:anywhere}table{border-collapse:collapse}"
        "td,th{text-align:right;padding:.15em .3em}</style>"
        "<h1>Missed sideways intervals</h1>"
        f"<p>{totals['missed']} of {totals['reference']} validation reference intervals. "
        '<a href="summary.json">Download counts and interval records</a></p>'
        "<table><thead><tr><th>Steps</th><th>Reference</th><th>Matched</th>"
        "<th>Missed</th></tr></thead><tbody>"
        + "".join(durations)
        + "</tbody></table>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )


def diagnose(corpus_path, output):
    corpus = Corpus(corpus_path)
    freeze = json.loads((corpus_path / "freeze.json").read_bytes())
    if freeze["reservation_sha256"] != digest(corpus_path / "reservation.json"):
        raise ValueError("The population reservation differs from the model freeze")
    bundle_digest = freeze["bundles"]["tcn"]
    root = corpus_path / "tcn"
    model = TCNBackend().load_bundle(root, manifest_sha256=bundle_digest, runtime=RUNTIME)
    manifest = ModelBundle.model_validate_json((root / "manifest.json").read_bytes())
    references, annotations = corpus.references("validation")
    if (
        model.analysis != PROFILE
        or corpus.dataset.content_sha256 != manifest.training.dataset_sha256
        or references.content_sha256 != manifest.training.validation_references.content_sha256
    ):
        raise ValueError("The model profile, dataset or validation reference digest differs")
    samples = _load_partition(
        corpus.dataset,
        references,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, member: annotations[member.source_namespace],
    )
    inputs = observation_inputs(samples)
    probabilities = model.predict(inputs, runtime=RUNTIME)
    records = []
    offset = 0
    sources = (sample for sample in samples for _ in sample.export.instruments)
    with output_directory(output) as folder:
        for sample, rows, expected in zip(
            sources, inputs.observations, _references(samples), strict=True
        ):
            actual = decode(probabilities[offset : offset + len(rows)], rows, model.analysis)
            offset += len(rows)
            for item in sideways_intervals(expected, actual):
                ref = item.reference
                filename = None
                if item.outcome != Outcome.MATCHED:
                    filename = f"interval-{len(records) + 1}.html"
                    chart(
                        folder / filename,
                        rows,
                        expected,
                        actual,
                        ref,
                        f"{ref.instrument_id} · [{ref.start}, {ref.stop}) · {label(item.outcome)}",
                        sample.snapshot_id,
                    )
                records.append(
                    {
                        "source_namespace": sample.source_namespace,
                        "run_id": sample.export.source.run_id,
                        "snapshot_id": sample.snapshot_id,
                        "export_sha256": sample.export.content_sha256,
                        "instrument_id": ref.instrument_id,
                        "step_count": len(rows),
                        "start": ref.start,
                        "stop": ref.stop,
                        "outcome": item.outcome.value,
                        "best_sideways_iou": item.best_sideways_iou,
                        "predictions": [
                            {"start": p.start, "stop": p.stop, "state": state(p)}
                            for p in item.overlapping
                        ],
                        "chart": filename,
                    }
                )
        summary = {
            "partition": "validation",
            "reference_kind": "automatic",
            "dataset_sha256": corpus.dataset.content_sha256,
            "validation_reference_sha256": references.content_sha256,
            "model_version_id": manifest.model_version_id,
            "model_manifest_sha256": bundle_digest,
            "experiment_count": len(samples),
            "sequence_count": len(inputs.info),
            "total": counts(records),
            "intervals": records,
            "by_reference_duration": [
                {
                    "min_steps": lo,
                    "stop_before": hi,
                    **counts(
                        [
                            r
                            for r in records
                            if lo <= r["stop"] - r["start"]
                            and (hi is None or r["stop"] - r["start"] < hi)
                        ]
                    ),
                }
                for lo, hi in DURATION_BINS
            ],
        }
        write_json(folder / "summary.json", summary)
        write_index(folder / "index.html", summary)
        if summary["total"]["missed"]:
            (folder / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
        (folder / "LICENSES.md").write_text(license_notices(include_plotly=True), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus", type=Path, help="Prepared and fitted profile-2.1 comparison directory"
    )
    parser.add_argument("output", type=Path, help="New or empty report directory")
    args = parser.parse_args()
    diagnose(args.corpus, args.output)
