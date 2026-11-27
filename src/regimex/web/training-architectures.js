// Runs and bundles created before the registry rename retain their original ID.
export function canonicalArchitectureId(id) {
  return id === "persistence" ? "segment-moe" : id;
}

export function architectureName(id, compact = false) {
  switch (canonicalArchitectureId(id)) {
    case "tcn": return compact ? "TCN" : "Temporal convolutional network (TCN)";
    case "segment-moe": return "Segment MoE";
    case "adaptive": return compact ? "Adaptive forest" : "Adaptive direction forest";
    case "baseline": return compact ? "Baseline" : "Logistic baseline";
    default: return id;
  }
}

export function availableArchitectures(descriptors) {
  const available = new Map();
  for (const descriptor of descriptors) {
    if (descriptor.executable === false) continue;
    const id = canonicalArchitectureId(descriptor.architecture_id);
    if (descriptor.architecture_id === id || !available.has(id)) available.set(id, descriptor);
  }
  return available;
}
