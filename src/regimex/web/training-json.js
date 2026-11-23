const identifiers = [
  "dataset_id", "architecture_id", "labels_reference_set_id",
  "validation_reference_set_id", "benchmark_selection_id",
];
const requestFields = new Set([
  ...identifiers, "policy", "parameters", "display_name", "analysis_profile_id",
  "training_runtime", "inference_runtime",
]);
const stringsAndNumbers = /"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?/g;

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function readTrainingRequest(text, base, {fromRun = false} = {}) {
  let input;
  try { input = JSON.parse(text); }
  catch { throw new Error("Enter valid JSON for the run configuration or training policy."); }
  if (!isObject(input)) throw new Error("Enter a JSON object for the run configuration or training policy.");

  // The API accepts fixed-update policies that omit the default budget and optimizer.
  const standalone = !Object.keys(input).some(key => requestFields.has(key)) &&
    ["optimizer", "budget", "updates", "validation_updates"].some(key => Object.hasOwn(input, key));
  const request = standalone ? {...JSON.parse(base), policy: input} : input;
  for (const key of Object.keys(request))
    if (!requestFields.has(key)) throw new Error("Unknown run configuration field: " + key + ".");
  for (const key of identifiers)
    if (!(fromRun && key === "benchmark_selection_id" && request[key] == null) &&
        (typeof request[key] !== "string" || !request[key].length))
      throw new Error("Provide a nonempty string for " + key + ".");
  for (const key of ["parameters", "policy", "training_runtime", "inference_runtime"]) {
    if (key.endsWith("_runtime") && request[key] === undefined) continue;
    if (!isObject(request[key])) throw new Error("Provide a JSON object for " + key + ".");
  }
  if (request.analysis_profile_id !== undefined &&
      (typeof request.analysis_profile_id !== "string" || !request.analysis_profile_id.length))
    throw new Error("Provide a nonempty string for analysis_profile_id.");
  if (request.display_name != null && typeof request.display_name !== "string")
    throw new Error("Provide a string or null for display_name.");
  if (!standalone) return {request, body: text};
  const tagged = JSON.parse(taggedTokens(base));
  tagged.spolicy = JSON.parse(taggedTokens(text));
  return {request, body: formatTagged(tagged)};
}

function decimalValue(token) {
  const [mantissa, exponent = "0"] = token.toLowerCase().split("e");
  const [whole, fraction = ""] = mantissa.split(".");
  const digits = (whole + fraction).replace(/^(-?)0+/, "$1");
  const significant = digits.replace(/0+$/, "");
  if (!significant || significant === "-") return significant + "0";
  return significant + "e" + (Number(exponent) - fraction.length + digits.length - significant.length);
}

function taggedTokens(text) {
  // Keep original number literals and distinguish them from user-supplied strings.
  return text.replace(stringsAndNumbers, token =>
    token.startsWith('"') ? '"s' + token.slice(1) : '"n' + token + '"');
}

function formatTagged(value) {
  return JSON.stringify(value, null, 2).replace(stringsAndNumbers, token =>
    token.startsWith('"n') ? token.slice(2, -1) : '"' + token.slice(2));
}

// Keep the persisted request's number literals when a stored run becomes a new draft.
function savedConfiguration(row) {
  if (row.straining.sorigin === "sexternal") {
    const configuration = row.straining.sconfiguration;
    return configuration == null ? null : formatTagged(configuration);
  }
  const fields = Object.fromEntries([...requestFields]
    .filter(key => Object.hasOwn(row.straining, "s" + key))
    .map(key => ["s" + key, row.straining["s" + key]]));
  if (row.sdisplay_name) fields.sdisplay_name = row.sdisplay_name;
  return formatTagged(fields);
}

export function readTrainingRuns(text) {
  const data = JSON.parse(text);
  const tagged = JSON.parse(taggedTokens(text));
  if (Array.isArray(data.items)) {
    data.items.forEach((row, index) => { row.configuration = savedConfiguration(tagged.sitems[index]); });
  } else {
    data.configuration = savedConfiguration(tagged);
  }
  return data;
}

export function renamedConfiguration(text, name) {
  const fields = JSON.parse(taggedTokens(text));
  if (name) fields.sdisplay_name = "s" + name;
  else delete fields.sdisplay_name;
  return formatTagged(fields);
}

export function differentSettings(left, right) {
  const differences = [];
  function compare(a, b, path) {
    if (isObject(a) && isObject(b)) {
      const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
      for (const key of [...keys].sort()) {
        if (!path.length && key === "sdisplay_name") continue;
        compare(a[key], b[key], [...path, key.slice(1)]);
      }
      return;
    }
    const first = a === undefined ? "—" : formatTagged(a);
    const second = b === undefined ? "—" : formatTagged(b);
    if (first !== second) differences.push([path.join("."), first, second]);
  }
  compare(JSON.parse(taggedTokens(left)), JSON.parse(taggedTokens(right)), []);
  return differences;
}

export function trainingFieldsBody(fields, settings) {
  const tagged = JSON.parse(taggedTokens(JSON.stringify(fields)));
  for (const [key, text] of Object.entries(settings))
    if (text !== undefined) tagged["s" + key] = JSON.parse(taggedTokens(text));
  return formatTagged(tagged);
}

export function setParameterOverrides(text, overrides) {
  // Editing a preset must preserve the numeric literals of unrelated model settings.
  const parameters = JSON.parse(taggedTokens(text.trim() || "{}"));
  if (!isObject(parameters)) throw new Error("Model configuration must be a JSON object.");
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) delete parameters["s" + key];
    else parameters["s" + key] = JSON.parse(taggedTokens(JSON.stringify(value)));
  }
  return formatTagged(parameters);
}

function checkPolicyNumbers(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) checkPolicyNumbers(child);
    return;
  }
  if (typeof value !== "string" || !value.startsWith("n")) return;
  const literal = value.slice(1), number = Number(literal), formatted = String(number);
  if (!Number.isFinite(number) || decimalValue(literal) !== decimalValue(formatted))
    throw new Error("A number would change precision in the form. Use Start training from the JSON editor to keep it exact.");
}

export function parseTrainingJSON(text, base, options) {
  const {request, body} = readTrainingRequest(text, base, options);
  const tagged = JSON.parse(taggedTokens(body));
  checkPolicyNumbers(tagged.spolicy);
  const settings = Object.fromEntries(["parameters", "training_runtime", "inference_runtime", "policy"]
    .filter(key => Object.hasOwn(tagged, "s" + key))
    .map(key => [key, formatTagged(tagged["s" + key])]));
  return {request, settings, body};
}

export function trainingRequestKey(body) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return JSON.stringify(JSON.parse(taggedTokens(text), (_, value) => {
    if (!isObject(value)) return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]]));
  }));
}
