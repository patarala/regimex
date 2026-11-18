"use strict";

const directionSymbols = { growth: "↗", decline: "↘", sideways: "→", no_trades: "—", insufficient_data: "—" };

function plotLabel(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function instrumentLanes(sequences, selected, colors, boundary, timeUnit) {
  const states = new Map();
  const labels = [];
  sequences.forEach((sequence, lane) => {
    const id = sequence.instrument_id;
    const characters = Array.from(id);
    const short = plotLabel(characters.length > 10 ? characters.slice(0, 9).join("") + "…" : id);
    labels.push(sequences.length === 1 ? "Prediction" : id === selected ? "<b>" + short + "</b>" : short);
    for (const interval of sequence.intervals) {
      const state = interval.direction || interval.status;
      if (!states.has(state)) states.set(state, []);
      states.get(state).push({ lane, interval });
    }
  });
  const traces = [];
  for (const [state, symbol] of Object.entries(directionSymbols)) {
    const pieces = states.get(state);
    if (!pieces) continue;
    const name = symbol + " " + state.replaceAll("_", " ");
    traces.push({
      type: "bar", orientation: "h", name, meta: "instrument-lane",
      y: pieces.map(({ lane }) => lane),
      base: pieces.map(({ interval }) => boundary(interval.start)),
      x: pieces.map(({ interval }) => boundary(interval.stop) - boundary(interval.start)),
      width: 0.65, marker: { color: colors[state], line: { color: "white", width: 1 } },
      text: pieces.map(() => symbol), textposition: "inside", insidetextanchor: "middle",
      hovertext: pieces.map(({ lane, interval }) => {
        const time = timeUnit ? "<br>" + plotLabel(timeUnit) + " [" +
          boundary(interval.start) + ", " + boundary(interval.stop) + ")" : "";
        return plotLabel(sequences[lane].instrument_id) + " · " + name +
          "<br>Steps [" + interval.start + ", " + interval.stop + ")" + time;
      }),
      hovertemplate: "%{hovertext}<extra></extra>",
    });
  }
  return {
    traces,
    axis: {
      type: "linear", fixedrange: true, showgrid: false, zeroline: false,
      tickmode: "array", tickvals: labels.map((_, index) => index), ticktext: labels,
      range: [labels.length - 0.5, -0.5],
    },
  };
}

function instrumentLaneClick(event) {
  const point = event.points.find((point) => point.data.meta === "instrument-lane");
  return point?.y;
}
