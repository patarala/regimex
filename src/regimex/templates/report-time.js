"use strict";

function reportTimeAxis(grid, lastBinFraction) {
  function midpoint(left, right) {
    const sum = left + right;
    return Number.isFinite(sum) ? sum / 2 : left / 2 + right / 2;
  }
  const boundaries = Array.from({length: grid.step_count + 1}, (_, step) =>
    step === grid.step_count ? grid.completed_horizon : grid.origin + step * grid.bin_width);
  // Adjacent bins at large absolute times may round to the same coordinate.
  const absoluteTime = Number.isFinite(grid.completed_horizon - grid.origin)
    && boundaries.every((edge, step) => {
      if (step === 0) return true;
      const previous = boundaries[step - 1];
      const center = midpoint(previous, edge);
      return previous < center && center < edge;
    });
  let unit = grid.unit;
  if (!absoluteTime) {
    for (let step = 0; step < grid.step_count; step++) boundaries[step] = step;
    const lastStart = grid.step_count - 1;
    const lastStop = lastStart + lastBinFraction;
    const lastCenter = midpoint(lastStart, lastStop);
    // Even relative coordinates may round a tiny terminal bin to zero width.
    if (lastStart < lastCenter && lastCenter < lastStop) {
      boundaries[grid.step_count] = lastStop;
      unit = "Bins from start";
    } else {
      boundaries[grid.step_count] = grid.step_count;
      unit = "Bin index";
      lastBinFraction = 1;
    }
  }
  return {
    unit,
    relative: !absoluteTime,
    lastBinFraction,
    boundary: (step) => boundaries[step],
    center: (step) => midpoint(boundaries[step], boundaries[step + 1]),
  };
}
