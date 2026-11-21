"use strict";

// Plotly writes concrete colors into SVG attributes and does not resolve CSS light-dark().
function reportPlotColors() {
  return reportIsDark() ? {
    paper: "#1a1a1c", plot: "#212124", ink: "#ebebed", grid: "#414146",
    price: "#dedee3", growth: "#76d8af", decline: "#ffa590", sideways: "#95bff1",
    no_trades: "#a9b1bd", insufficient_data: "#d9c99d", unknown: "#c6a7ff",
  } : {
    paper: "#ffffff", plot: "#fbfcfe", ink: "#20252c", grid: "#e1e5e8",
    price: "#263b55", growth: "#147d64", decline: "#b84c34", sideways: "#416ea7",
    no_trades: "#687380", insufficient_data: "#786d4e", unknown: "#763ac3",
  };
}
