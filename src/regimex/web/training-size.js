import {$, option} from "./ui.js";
import {setParameterOverrides} from "./training-json.js";

export function parameterCountLabel(count) {
  return count.toLocaleString("en-US") + " parameters";
}

export function createSizePicker() {
  const container = $("training-size-field"), select = $("training-size"), parametersField = $("training-parameters");
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
    selectArchitecture(architecture) {
      sizes = architecture?.sizes || [];
      properties = architecture?.parameter_schema?.properties || {};
      layoutKeys = [...new Set(sizes.flatMap(preset => Object.keys(preset.parameters)))];
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
