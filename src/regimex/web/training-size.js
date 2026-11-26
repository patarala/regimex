import {$, addFieldHelp, element, option} from "./ui.js";
import {integerParameterKeys, setParameterOverrides} from "./training-json.js";

export function parameterCountLabel(count) {
  return count.toLocaleString("en-US") + " parameters";
}

function editableFields(architecture) {
  if (architecture?.default_training_policy?.budget !== "fit") return [];
  return Object.entries(architecture.parameter_schema?.properties ?? {}).filter(([, field]) =>
    field.type === "integer" || (field.type === "string" && Array.isArray(field.enum)));
}

export function validateModelFields(architecture, text) {
  const fields = editableFields(architecture);
  if (!fields.length) return;
  const parameters = JSON.parse(text), integers = integerParameterKeys(text);
  for (const [key, field] of fields) {
    const present = Object.hasOwn(parameters, key);
    const value = present ? parameters[key] : field.default;
    if (field.type === "string") {
      if (!field.enum.includes(value))
        throw new Error(`${field.title} must be one of ${field.enum.join(", ")}.`);
    } else if ((present && !integers.has(key)) || !Number.isSafeInteger(value) ||
        value < field.minimum || value > field.maximum) {
      let range = "";
      if (field.minimum != null && field.maximum != null) range = ` from ${field.minimum} to ${field.maximum}`;
      else if (field.minimum != null) range = ` of at least ${field.minimum}`;
      else if (field.maximum != null) range = ` of at most ${field.maximum}`;
      throw new Error(`${field.title} must be a whole number${range}.`);
    }
  }
}

export function createSizePicker() {
  const container = $("training-size-field"), select = $("training-size"), parametersField = $("training-parameters");
  const modelFields = $("training-model-fields"), controls = new Map();
  let sizes = [], properties = {}, layoutKeys = [];
  function readParameters() {
    const value = JSON.parse(parametersField.value.trim() || "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
    return value;
  }
  function setting(parameters, key) {
    return Object.hasOwn(parameters, key) ? parameters[key] : properties[key]?.default;
  }
  function matches(preset, parameters) {
    return layoutKeys.every(key =>
      JSON.stringify(setting(parameters, key)) === JSON.stringify(setting(preset.parameters, key)));
  }
  function refresh() {
    if (controls.size) {
      let parameters;
      try { parameters = readParameters(); } catch { /* JSON editing can temporarily be incomplete. */ }
      for (const [key, control] of controls) {
        control.disabled = !parameters;
        if (parameters) control.value = setting(parameters, key);
      }
    }
    if (!sizes.length) return;
    let index = -1;
    try {
      const parameters = readParameters();
      index = sizes.findIndex(preset => matches(preset, parameters));
    } catch { /* Keep the draft editable in the architecture settings. */ }
    select.value = String(index);
  }
  select.addEventListener("change", () => {
    const preset = sizes[Number(select.value)];
    if (!preset) {
      refresh();
      parametersField.closest("details").open = true;
      parametersField.focus();
      return;
    }
    const overrides = Object.fromEntries(layoutKeys.map(key => {
      const value = setting(preset.parameters, key);
      return [key, JSON.stringify(value) === JSON.stringify(properties[key]?.default) ? undefined : value];
    }));
    try { parametersField.value = setParameterOverrides(parametersField.value, overrides); }
    catch {
      refresh();
      parametersField.closest("details").open = true;
      parametersField.focus();
      parametersField.setCustomValidity("Enter a JSON object before changing model size.");
      parametersField.reportValidity();
      parametersField.setCustomValidity("");
      return;
    }
    parametersField.dispatchEvent(new Event("input", {bubbles: true}));
  });
  parametersField.addEventListener("input", refresh);
  return {
    validate() {
      for (const field of controls.values()) {
        if (field.disabled || field.validity.valid) continue;
        field.reportValidity();
        field.focus();
        throw new Error(field.validationMessage);
      }
    },
    selectArchitecture(architecture) {
      sizes = architecture?.sizes || [];
      properties = architecture?.parameter_schema?.properties || {};
      layoutKeys = [...new Set(sizes.flatMap(preset => Object.keys(preset.parameters)))];
      controls.clear();
      modelFields.replaceChildren();
      for (const [key, field] of editableFields(architecture)) {
        const choices = field.type === "string";
        const input = element(choices ? "select" : "input"), label = element("label", field.title), wrapper = element("div");
        if (choices) {
          for (const value of field.enum) {
            const name = value.replaceAll("_", " ");
            input.append(option(value, name[0].toUpperCase() + name.slice(1)));
          }
        } else {
          input.type = "number";
          input.min = field.minimum;
          input.max = field.maximum;
        }
        input.id = "training-parameter-" + key;
        input.required = true;
        label.htmlFor = input.id;
        wrapper.append(label, input);
        if (field.description) addFieldHelp(label, input, field.description);
        controls.set(key, input);
        modelFields.append(wrapper);
        input.addEventListener("input", () => {
          if (!input.validity.valid) return;
          const value = choices ? input.value : input.valueAsNumber;
          parametersField.value = setParameterOverrides(parametersField.value, {
            [key]: value === field.default ? undefined : value,
          });
          parametersField.dispatchEvent(new Event("input", {bubbles: true}));
        });
      }
      modelFields.hidden = !controls.size;
      refresh();
      container.hidden = !sizes.length;
      select.disabled = !sizes.length;
      if (!sizes.length) {
        select.replaceChildren();
        return;
      }
      const custom = option("-1", "Custom size (JSON)");
      select.replaceChildren(...sizes.map((preset, index) =>
        option(String(index), preset.name + " · " + parameterCountLabel(preset.trainable_parameters))), custom);
      refresh();
    },
  };
}
