import {addFieldHelp, element, option} from "./ui.js";

const maxSeed = 2 ** 32 - 1;
const maxUpdates = 32000;
const budgetNames = {epochs: "Epochs", updates: "Optimizer updates"};
const optimizerNames = {lbfgs: "L-BFGS", adamw: "AdamW"};
// A pasted API policy may omit fields. Their schema defaults can differ from the
// architecture's suggested values for a new run.
const numbers = {
  epochs: {title: "Maximum epochs", min: 1, max: 1000, fallback: 30},
  validation_interval: {title: "Epochs between validation checks", min: 1, fallback: 2},
  patience: {title: "Checks without improvement", min: 1, fallback: 5},
  updates: {title: "Updates per seed", min: 1, max: maxUpdates},
  batch_size: {title: "Batch size", min: 1, max: 128, fallback: 16},
  learning_rate: {title: "Learning rate", min: 0, step: "any", fallback: 0.001},
  weight_decay: {title: "Weight decay", min: 0, step: "any", fallback: 0.0001},
};
const commonNumbers = ["batch_size", "learning_rate", "weight_decay"];
const epochNumbers = ["epochs", "validation_interval", "patience"];
const selectionMetric = "validation_segment_macro_f1";
const help = {
  budget: "An epoch is one pass over the training data. A fixed update budget gives each repeat the same number of weight updates.",
  epochs: "Maximum number of passes over the training data per repeat. Early stopping can finish a repeat sooner.",
  validation_interval: "How often to score direction intervals on the validation labels, in epochs.",
  patience: "Stop this repeat after this many consecutive checks without a new best validation interval macro-F1. An improvement resets the count.",
  updates: "Number of optimizer steps in each repeat. Only batches that update the weights count.",
  validation_updates: "Leave blank for up to 10 evenly spaced checks. To choose your own, enter update numbers separated by commas and end with the final update.",
  batch_size: "Maximum number of instrument sequences grouped into one training batch. Larger batches use more memory.",
  learning_rate: "Controls the size of each weight update. Lower values change the weights more gradually.",
  weight_decay: "Shrinks weights at each optimizer step. Larger values shrink them more. Set to zero to turn it off.",
  repeats: "Each repeat trains the model with a different seed. Increasing the count adds seeds to the list.",
  seeds: "Generate a set or enter one integer per repeat, separated by commas. Reuse the same seeds to compare settings.",
};

class PolicyError extends Error {
  constructor(field, message) {
    super(message);
    this.field = field;
  }
}

function repeatLimit(architecture) {
  if (architecture.architecture_id === "baseline") return 1;
  const workspaceLimit = architecture.max_training_seeds ?? 64;
  return architecture.architecture_id === "tcn" ? Math.min(32, workspaceLimit) : workspaceLimit;
}

function isSeed(value) {
  return Number.isInteger(value) && value >= 0 && value <= maxSeed;
}

function extendSeeds(seeds, count, excluded = []) {
  const result = [...new Set(seeds.filter(isSeed))].slice(0, count);
  const used = new Set([...excluded, ...result]);
  const word = new Uint32Array(1);
  while (result.length < count) {
    const seed = crypto.getRandomValues(word)[0];
    if (!used.has(seed)) {
      result.push(seed);
      used.add(seed);
    }
  }
  return result;
}

function validate(architecture, raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw))
    throw new PolicyError("budget", "Training policy must be a JSON object.");
  const fixed = "updates" in raw || "validation_updates" in raw || raw.budget === "updates";
  const budget = fixed ? "updates" : "epochs";
  const numeric = [...(fixed ? ["updates"] : epochNumbers), ...commonNumbers];
  const allowed = ["optimizer", "selection_metric", "seeds", ...numeric];
  if (fixed) allowed.push("budget", "validation_updates");
  for (const key of Object.keys(raw))
    if (!allowed.includes(key))
      throw new PolicyError("budget", `Unknown field “${key}” for the ${budget} budget.`);
  if (!(architecture.supported_training_budgets || ["epochs"]).includes(budget))
    throw new PolicyError("budget", "This architecture does not support " + budget + " budgets.");

  const budgetDefaults = fixed ? {budget: "updates", optimizer: "adamw"} : {};
  const policy = {seeds: [11, 23, 47], selection_metric: selectionMetric, ...budgetDefaults, ...raw};
  if (fixed && (policy.budget !== "updates" || policy.optimizer !== "adamw"))
    throw new PolicyError("budget", "A fixed-update policy requires the updates budget and the AdamW optimizer.");
  const optimizers = architecture.supported_optimizers || [architecture.default_training_policy.optimizer];
  if (!optimizers.includes(policy.optimizer))
    throw new PolicyError("optimizer", "Choose a supported optimizer: " + optimizers.join(", ") + ".");
  if (policy.selection_metric !== selectionMetric)
    throw new PolicyError("selection_metric", "The supported selection metric is " + selectionMetric + ".");
  if (!Array.isArray(policy.seeds) || !policy.seeds.length ||
      !policy.seeds.every(isSeed))
    throw new PolicyError("seeds", "Seeds must be whole numbers between 0 and 4294967295.");
  if (new Set(policy.seeds).size !== policy.seeds.length)
    throw new PolicyError("seeds", "Enter each seed only once.");
  if (architecture.architecture_id === "baseline" && policy.seeds.length !== 1)
    throw new PolicyError("seeds", "The logistic baseline requires exactly one seed.");
  if (policy.seeds.length > repeatLimit(architecture))
    throw new PolicyError("seeds", `Choose at most ${repeatLimit(architecture)} training repeats.`);
  for (const key of numeric) {
    const {min, max, step, title, fallback} = numbers[key];
    const value = key in raw ? raw[key] : fallback;
    if (!Number.isFinite(value) || (step !== "any" && !Number.isSafeInteger(value)) ||
        value < min || value > (max ?? Infinity))
      throw new PolicyError(key, `${title} must be ${step === "any" ? "a number" : "a whole number"} ` +
        (max === undefined ? `of at least ${min}.` : `between ${min} and ${max}.`));
    if (architecture.architecture_id === "baseline" && value !== fallback)
      throw new PolicyError(key, `The logistic baseline does not use ${key}. Leave it at its default.`);
    policy[key] = value;
  }
  if (policy.learning_rate <= 0)
    throw new PolicyError("learning_rate", "Learning rate must be greater than zero.");
  if (fixed) {
    if (policy.updates * policy.seeds.length > maxUpdates)
      throw new PolicyError("updates", "All seeds together can use at most 32,000 updates.");
    const points = policy.validation_updates;
    if (!Array.isArray(points) || !points.length || points.length > 1000 ||
        points.some((n, i) => !Number.isInteger(n) || n < 1 || (i > 0 && n <= points[i - 1])) ||
        points.at(-1) !== policy.updates)
      throw new PolicyError("validation_updates",
        "Use up to 1,000 increasing checkpoints and end with update " + policy.updates + ".");
  }
  return policy;
}

function automaticCheckpoints(updates) {
  const count = Math.min(10, updates);
  return Array.from({length: count}, (_, i) => Math.ceil((i + 1) * updates / count));
}

export function policyDraft(architecture, rawPolicy, previousDraft = null) {
  const policy = validate(architecture, rawPolicy);
  const draft = {...previousDraft, ...policy, budget: policy.budget || "epochs",
    seeds: policy.seeds.join(", "), repeats: policy.seeds.length};
  if (policy.budget === "updates") {
    draft.optimizer = previousDraft?.optimizer ?? architecture.default_training_policy.optimizer;
    const automatic = automaticCheckpoints(policy.updates);
    const keepAutomatic = previousDraft?.budget === "updates" && !previousDraft.validation_updates &&
      automatic.length === policy.validation_updates.length &&
      automatic.every((step, i) => step === policy.validation_updates[i]);
    draft.validation_updates = keepAutomatic ? "" : policy.validation_updates.join(", ");
  }
  return draft;
}

function reject(field, message) {
  const settings = field.closest("details");
  if (settings) settings.open = true;
  field.setAttribute("aria-invalid", "true");
  field.focus();
  throw new Error(message);
}

function integerList(value) {
  return value.split(",").map(token => /^\d+$/.test(token.trim()) ? Number(token) : NaN);
}

export function createTrainingPolicy(container) {
  let architecture = null, values = {};
  const fields = new Map();

  function snapshot() {
    const saved = {...values};
    for (const [key, field] of fields) saved[key] = field.value;
    return saved;
  }

  function add(key, title, field) {
    field.id = "training-policy-" + key;
    field.value = values[key] ?? "";
    fields.set(key, field);
    const wrapper = element("div"), label = element("label", title);
    label.htmlFor = field.id;
    wrapper.append(label, field);
    container.append(wrapper);
    let hint = help[key];
    if (key === "seeds" && architecture.architecture_id === "baseline")
      hint = "The logistic baseline records one seed and fits once for each regularization setting.";
    if (key === "validation_interval" && architecture.architecture_id === "tcn")
      hint = "With a value of 2, the TCN scores validation intervals at epochs 1, 2, 4, 6 and so on. It also scores the final epoch if reached.";
    if (hint) addFieldHelp(label, field, hint);
    return field;
  }

  function number(key, {title, min, max, step = "1"} = numbers[key]) {
    const input = element("input");
    input.type = "number";
    input.required = true;
    input.min = min;
    if (max !== undefined) input.max = max;
    input.step = step;
    return add(key, title, input);
  }

  function choice(key, title, choices, labels) {
    const select = element("select");
    for (const value of choices) select.append(option(value, labels[value] || value));
    return add(key, title, select);
  }

  function seedControls(single) {
    const repeats = single ? null : number("repeats", {
      title: "Training repeats", min: 1, max: repeatLimit(architecture),
    });
    const seeds = single ? number("seeds", {title: "Random seed", min: 0, max: maxSeed})
      : add("seeds", "Random seeds", element("input"));
    seeds.required = true;
    const entry = element("div", "", "training-seed-entry");
    if (!single) seeds.parentElement.classList.add("training-wide");
    seeds.before(entry);
    const generate = element("button", single ? "Generate seed" : "Generate seeds", "quiet");
    generate.type = "button";
    entry.append(seeds, generate);

    const current = () => integerList(seeds.value);
    if (repeats) {
      let previous;
      repeats.addEventListener("input", () => {
        if (!repeats.validity.valid) return;
        // Keep the original prefix while a multi-digit count is still being typed.
        previous ??= current();
        const resized = extendSeeds(previous, repeats.valueAsNumber);
        if (resized.length > previous.length) previous = resized;
        seeds.value = resized.join(", ");
      });
      repeats.addEventListener("blur", () => { previous = null; });
      seeds.addEventListener("input", () => {
        const list = current();
        if (list.every(isSeed)) repeats.value = list.length;
      });
    }
    generate.addEventListener("click", () => {
      if (repeats && !repeats.reportValidity()) {
        repeats.focus();
        return;
      }
      seeds.value = extendSeeds([], repeats?.valueAsNumber ?? 1, current()).join(", ");
      seeds.dispatchEvent(new Event("input", {bubbles: true}));
    });
  }

  function render() {
    container.replaceChildren();
    fields.clear();
    if (!architecture) return;
    if (architecture.architecture_id === "baseline") {
      seedControls(true);
      return;
    }
    const optimizers = architecture.supported_optimizers || [values.optimizer];
    const budgets = (architecture.supported_training_budgets || ["epochs"])
      .filter(budget => budget !== "updates" || optimizers.includes("adamw"));
    if (budgets.length > 1) {
      const budget = choice("budget", "Training budget", budgets, budgetNames);
      budget.addEventListener("change", () => {
        values = snapshot();
        render();
        fields.get("budget").focus();
      });
    }
    if (values.budget === "updates") {
      number("updates");
      const input = element("input");
      input.placeholder = "Automatic";
      add("validation_updates", "Validation checkpoints", input);
    } else {
      for (const key of epochNumbers) number(key);
      if (optimizers.length > 1) choice("optimizer", "Optimizer", optimizers, optimizerNames);
    }
    for (const key of commonNumbers) number(key);
    seedControls(false);
  }

  function select(next, draft = null) {
    architecture = next;
    const defaults = architecture?.default_training_policy;
    const seeds = defaults?.seeds.slice(0, repeatLimit(next));
    values = defaults ? {...defaults, seeds: seeds.join(", "), repeats: seeds.length, budget: "epochs", ...draft} : {};
    render();
  }

  function payload() {
    if (!architecture) throw new Error("Choose an architecture.");
    for (const field of fields.values())
      if (!field.validity.valid) reject(field, field.validationMessage);
    const defaults = architecture.default_training_policy;
    const seedField = fields.get("seeds");
    if (architecture.architecture_id === "baseline")
      return {...defaults, seeds: [seedField.valueAsNumber]};
    const seeds = integerList(seedField.value);
    const fixed = values.budget === "updates";
    const policy = fixed
      ? {budget: "updates", optimizer: "adamw", selection_metric: defaults.selection_metric, seeds}
      : {...defaults, optimizer: fields.get("optimizer")?.value ?? defaults.optimizer, seeds};
    for (const key of [...(fixed ? ["updates"] : epochNumbers), ...commonNumbers])
      policy[key] = fields.get(key).valueAsNumber;
    if (fixed) {
      const schedule = fields.get("validation_updates").value.trim();
      policy.validation_updates = schedule ? integerList(schedule) : automaticCheckpoints(policy.updates);
    }
    try {
      return validate(architecture, policy);
    } catch (error) {
      const field = fields.get(error.field);
      if (field) reject(field, error.message);
      throw error;
    }
  }

  container.addEventListener("input", () => {
    for (const field of fields.values()) field.removeAttribute("aria-invalid");
  });
  container.addEventListener("invalid", event => {
    event.target.setAttribute("aria-invalid", "true");
  }, true);
  return {select, snapshot, payload};
}
