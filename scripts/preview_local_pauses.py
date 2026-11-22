"""Compare three frozen pause classifiers on registered examples or a simulation export."""

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path

from check_local_pause_transfer import CONFIG, RUNTIME, examples
from fit_local_pause_features import load_predictors
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from refine_model_intervals import refine_decoded

from regimex.application.analysis_computation import snapshot_inputs
from regimex.exports import load_export_v2
from regimex.model_contracts import ObservationInputs
from regimex.segmentation import decode
from regimex.storage import digest, write_new_file

COLORS = {
    "sideways": "#716b9e",
    "growth": "#198367",
    "decline": "#bc5961",
    "no_trades": "#969aa1",
    "insufficient_data": "#ceced1",
}
LABELS = {
    "truth": "Known direction",
    "control": "TCN",
    "candidate": "TCN + slopes",
    "hybrid": "TCN + price check",
}


def scored_examples(comparison, training, noise):
    result = json.loads((comparison / "results.json").read_bytes())
    plan = comparison / "plan.json"
    if result["plan_sha256"] != digest(plan) or result["training_freeze_sha256"] != digest(
        training / "fitted/freeze.json"
    ):
        raise ValueError("The preview requires the scored plan and frozen model pair")
    cases = json.loads(plan.read_bytes())["cases"]
    for shape, title in (
        ("plateau-32", "32-step pause inside a rising trend"),
        ("slowdown", "Rising trend with a slowdown"),
    ):
        # Take the first rising path at this noise level in the first registered seed group.
        idx = 8 if noise == "low" else 24
        data = examples(shape, cases[shape][0])
        inputs = data.inputs
        sample = ObservationInputs(
            (inputs.info[idx],),
            (inputs.observations[idx],),
            (inputs.time_grids[idx],),
            (inputs.instruments[idx],),
        )
        yield title, sample, 0, data.references[idx]


def export_examples(path):
    inputs = snapshot_inputs(load_export_v2(path))
    for idx, instrument in enumerate(inputs.instruments):
        yield escape(instrument.instrument_id), inputs, idx, None


def chart(samples, models):
    fig = make_subplots(
        rows=2,
        cols=len(samples),
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        horizontal_spacing=0.14,
        vertical_spacing=0.08,
        subplot_titles=[title for title, _, _, _ in samples],
    )
    legend = set()
    previous_inputs = None
    for col, (_, inputs, idx, truth) in enumerate(samples, start=1):
        if inputs is not previous_inputs:
            predictions = {
                name: model.predict(inputs, runtime=RUNTIME) for name, model in models.items()
            }
            previous_inputs = inputs
        rows = inputs.observations[idx]
        offset = sum(info.step_count for info in inputs.info[:idx])
        segments = {"truth": truth} if truth is not None else {}
        for name, probs in predictions.items():
            segments[name] = decode(probs[offset : offset + len(rows)], rows, CONFIG)
        segments["hybrid"] = refine_decoded(rows, segments["control"], CONFIG)
        fig.add_trace(
            go.Scatter(
                x=[r.step for r in rows],
                y=[r.last_price for r in rows],
                mode="lines",
                line={"color": "#34445a", "width": 1.5},
                showlegend=False,
                hovertemplate="Step %{x}<br>Close %{y}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        for label, intervals in segments.items():
            by_status = defaultdict(list)
            for interval in intervals:
                key = interval.direction.value if interval.direction else interval.status
                by_status[key].append(interval)
            for name, spans in by_status.items():
                times, lanes, tips = [], [], []
                for interval in spans:
                    times.extend((interval.start, interval.stop, None))
                    lanes.extend((LABELS[label], LABELS[label], None))
                    tip = (
                        f"{LABELS[label]} · {name.replace('_', ' ')}<br>"
                        f"Steps [{interval.start}, {interval.stop})"
                    )
                    tips.extend((tip, tip, None))
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=lanes,
                        text=tips,
                        mode="lines",
                        line={"color": COLORS[name], "width": 12},
                        name=name.replace("_", " ").capitalize(),
                        legendgroup=name,
                        showlegend=name not in legend,
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=2,
                    col=col,
                )
                legend.add(name)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=[LABELS[k] for k in reversed(segments)],
            ticklabelstandoff=6,
            row=2,
            col=col,
        )
        fig.update_xaxes(title_text="Step", row=2, col=col)
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=540,
        legend={"orientation": "h", "y": -0.18},
        margin={"t": 75, "b": 85, "l": 125, "r": 24},
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--comparison", type=Path, help="Completed synthetic comparison")
    source.add_argument("--export", type=Path, help="Validated experiment-v2 JSON file")
    parser.add_argument(
        "--noise", choices=("low", "high"), help="Synthetic noise level (default: high)"
    )
    parser.add_argument("output", type=Path, help="New self-contained HTML file")
    for field in ("training", "populations", "previous"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    if args.export and args.noise:
        parser.error("--noise applies to the synthetic comparison")
    models = load_predictors(args.training, args.populations, args.previous)
    samples = (
        export_examples(args.export)
        if args.export
        else scored_examples(args.comparison, args.training, args.noise or "high")
    )
    figure = chart(list(samples), models)
    write_new_file(
        args.output, figure.to_html(include_plotlyjs=True, div_id="pause-comparison").encode()
    )
