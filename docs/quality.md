# Model quality

## Find the score for a model

- **A training run.** Open **Training**, select the run and read **Model evaluation**.
  This section shows scores for its saved weights. If a successful workspace run
  has weights but no recorded evaluation, choose **Evaluate model** to score it
  on the run’s validation data.
  [Evaluate a training run](training.md#evaluate-and-publish)
- **A published model.** Read `active_model_version_id` from
  `GET /api/v1/models?published_only=true`
  to find the installation’s default. For a completed analysis, use
  `model.model_version_id` in its downloaded result. Request
  `GET /api/v1/metrics?model_version_id=ID` for that version’s quality status
  and recorded publication evaluation. If `evaluation_available` is `false`,
  the installation has no evaluation attached to that publication.
  [Model and metrics API](api.md#choose-a-model-or-compare-results)
- **A reference release or offline experiment.** Use the report for the named
  weights and population. The [reference benchmarks](evaluations/reference-releases.md)
  cover specific baseline and TCN fits. The [offline neural comparison](evaluations/neural-direction.md#segment-moe-on-exchange-prices-and-known-drift)
  includes a Segment MoE with a pretrained drift expert. A new
  [workspace fit](models.md#segment-moe) trains both experts and the gate from scratch.

## Read the metrics

**Interval macro-F1** measures whether a model finds whole periods of growth,
decline and sideways movement. Within one instrument and experiment, a prediction
matches a reference interval of the same direction when their intersection covers
at least half their union (IoU ≥ 0.50). The evaluator takes the largest set
of one-to-one matches. A missed reference lowers recall, and an extra predicted
interval lowers precision.

New evaluation reports also score the same predictions at **IoU ≥ 0.75**.
Compare this value (`strict_segment_metrics.macro_f1`) with the primary score
to see how often a direction was found with close agreement on its extent.
A two-step shift gives two ten-step intervals an IoU of 8/12, about 0.67. They
match at 0.50 and miss at 0.75. The same shift on a hundred-step interval gives
an IoU of 98/102, about 0.96. For timing in steps, read the boundary F1 with
its four-step tolerance and the reported early or late transition times.
Training uses the 0.50 score for selection.

**Point macro-F1** scores each step with a known reference direction. Leaving such
a step unclassified counts as a miss. A model can score well by step and still miss
a short interval. Both metrics give equal weight to all three directions. A class
with neither reference cases nor predictions contributes zero. A pooled score uses
counts across the whole population. Some offline reports also give equal weight to
each experiment length or observation condition and label this average separately.

Confidence intervals resample whole experiments, keeping their instruments
together. If a draw contains no labeled steps, per-step F1 is undefined.
Evaluation reports from version 2.5 omit the per-step confidence interval
in that case and record how many draws lacked labels. Predictions that miss
every labeled step still score zero.

Training run comparisons and offline comparison reports (version 2.3) use the
same seeded draws to measure a difference between two fixed models. Each draw
pools the direction counts before computing macro-F1. The paired interval
describes variation across validation experiments with these weights and labels.

## Compare models

Compare candidates on the same experiments, reference labels, analysis profile
and aggregation. Exchange benchmarks use automatic direction labels. Controlled
price paths can use their independently specified drift. Neither reference measures
agreement with human annotators. [Observations and labels](models.md#observations-and-intervals)

Keep all native exchange runs that share a generator seed in one partition,
including runs with different lengths. Both local fits and service training
check the registered groups before using the data. For older exports with
UUID-only run IDs, keep the original seed with the export.
[Native seed groups and dataset import](api.md#1-create-observations-and-references)

Validation scores guide checkpoint and settings selection. Re-evaluating the saved
weights on those observations checks the selected fit on the same data. A held-out
score measures performance on a separately reserved population after selection
has finished. Freeze the candidate before opening that benchmark.
[Final evaluation procedure](api.md#4-freeze-and-evaluate-final-data) · [Evaluation reports](evaluations/README.md)
