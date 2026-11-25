import {$} from "./ui.js";
import {showTrainingActions} from "./training-evaluation.js";

const api = "/api/v1/benchmarks";

export function createFinalBenchmark({session, action, refresh}) {
  const section = $("training-final"), status = $("training-final-status"), button = $("training-final-next");
  let reservation = null;

  function reset() {
    reservation = null;
    section.hidden = button.hidden = true;
    button.onclick = null;
    $("training-final-actions").replaceChildren();
  }

  function setStep(label, work) {
    button.textContent = label;
    button.onclick = () => action(work);
    button.hidden = false;
  }

  async function show(run, attempt, valid) {
    const id = run.training.benchmark_selection_id;
    if (!id) { reset(); return; }
    const query = "?" + new URLSearchParams({benchmark_id: id});
    const model = run.training.model_version_id;
    section.hidden = false;
    button.hidden = true;
    button.onclick = null;
    showTrainingActions($("training-final-actions"), attempt ? [attempt] : [], session);

    function evaluate() {
      // A lost response, reload or second tab must attach to the same attempt.
      // Only a recorded failure or cancellation starts a different retry.
      const key = "final-" + run.run_id + "-" + (attempt?.run_id || "first");
      button.hidden = true;
      return session.json(api + "/evaluations" + query, {
        method: "POST", body: {model_version_id: model}, headers: {"Idempotency-Key": key},
      });
    }

    if (attempt) {
      status.textContent = "Benchmark: " + id;
      if (!run.weights_deletion_id && ["failed", "cancelled"].includes(attempt.job.state))
        setStep("Retry final evaluation", evaluate);
      return;
    }
    if (run.weights_deletion_id) { reset(); return; }

    if (!reservation) {
      status.textContent = "Loading benchmark…";
      try {
        const view = await session.json(api + query);
        if (!valid()) return;
        reservation = view;
      } catch (error) {
        if (valid()) {
          status.textContent = error.message;
          button.textContent = "Reload benchmark";
          button.onclick = refresh;
          button.hidden = false;
        }
        return;
      }
    }
    const frozen = reservation.frozen_choice?.model_version_id;
    const opened = reservation.reservation.opened_at;
    if (frozen && frozen !== model) {
      status.textContent = "Benchmark " + id + " is reserved for the frozen model " + frozen + ".";
    } else if (reservation.reservation.freeze && !frozen) {
      status.textContent = "This benchmark already has a frozen choice.";
    } else if (opened && frozen === model) {
      status.textContent = "The held-out test is open for this model.";
      setStep("Evaluate frozen model", evaluate);
    } else if (reservation.reservation.selection_started_at) {
      const endpoint = frozen ? "/open" : "/freeze";
      status.textContent = frozen ? "This model is frozen. Its test references are sealed." :
        "Test references are sealed. Freeze the model when selection is complete.";
      setStep(frozen ? "Open held-out test" : "Freeze this model", async () => {
        const confirmation = frozen
          ? "Open the held-out test for " + id + "? Its references cannot be sealed again."
          : "Freeze " + model + " for " + id + "? This closes model selection for all runs using this benchmark.";
        if (!window.confirm(confirmation)) return;
        reservation = null;
        button.hidden = true;
        await session.json(api + endpoint + query, {
          method: "POST", body: frozen ? undefined : {model_version_id: model},
        });
      });
    } else {
      status.textContent = "Model selection has not started for benchmark " + id + ".";
    }
  }

  return {show, reset, invalidate() { reservation = null; }};
}
