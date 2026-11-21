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

function readRequest(text, base) {
  let input;
  try { input = JSON.parse(text); }
  catch { throw new Error("Enter valid JSON for the run configuration or training policy."); }
  if (!isObject(input)) throw new Error("Enter a JSON object for the run configuration or training policy.");

  // The API accepts fixed-update policies that omit the default budget and optimizer.
  const standalone = !Object.keys(input).some(key => requestFields.has(key)) &&
    ["optimizer", "budget", "updates", "validation_updates"].some(key => Object.hasOwn(input, key));
  const request = standalone ? {...base, policy: input} : input;
  for (const key of Object.keys(request))
    if (!requestFields.has(key)) throw new Error("Unknown run configuration field: " + key + ".");
  for (const key of identifiers)
    if (typeof request[key] !== "string" || !request[key].length)
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
  return {request, standalone};
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

function checkFormNumbers(value, preserveNumberType) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) checkFormNumbers(child, preserveNumberType);
    return;
  }
  if (typeof value !== "string" || !value.startsWith("n")) return;
  const literal = value.slice(1), number = Number(literal), formatted = String(number);
  if (!Number.isFinite(number) || decimalValue(literal) !== decimalValue(formatted))
    throw new Error("A number would change precision in the form. Use Start training from the JSON editor to keep it exact.");
  if (preserveNumberType && /[.Ee]/.test(literal) !== /[.Ee]/.test(formatted))
    throw new Error("A number in parameters would change between an integer and a float in the form. Use Start training from the JSON editor to keep its type.");
}

export function parseTrainingJSON(text, base) {
  const {request} = readRequest(text, base);
  // Plugin parameters preserve the JSON number type. Runtime and policy fields have a declared type.
  for (const [key, value] of Object.entries(JSON.parse(taggedTokens(text))))
    checkFormNumbers(value, key === "sparameters");
  return request;
}

export function trainingJSONBody(text, base) {
  const {request, standalone} = readRequest(text, base);
  if (!standalone) return text;
  // The browser may round plugin parameters or policy numbers when serializing parsed JSON.
  const fields = Object.entries(request)
    .filter(([key, value]) => key !== "policy" && value !== undefined)
    .map(([key, value]) => JSON.stringify(key) + ":" + JSON.stringify(value));
  fields.push('"policy":' + text);
  return "{" + fields.join(",") + "}";
}

export function trainingRequestKey(body) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return JSON.stringify(JSON.parse(taggedTokens(text), (_, value) => {
    if (!isObject(value)) return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]]));
  }));
}
