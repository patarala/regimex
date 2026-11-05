# Model quality

The `tcn-0.2.0` reference model reaches **0.7129 interval macro-F1** and
**0.8932 point macro-F1** on 384 held-out synthetic experiments. These results
measure agreement with automatic reference policy `retrospective-directions/3.0`.
Independent human accuracy and transfer to another market simulator have not
been measured. The logistic baseline is also available; the measurements below
apply to this TCN bundle.

## Evaluation protocol

Training uses 512 experiments with 512 steps, totaling 769 instrument sequences.
Preprocessing is fitted on training data only. Validation contains 192 experiments,
with 64 each at lengths 256, 512 and 1,024. Whole experiments and related seed groups
stay in one partition, including both instruments of a shared market.

Each training run performs 1,200 successful optimizer updates and is scored at
updates 300, 600, 900 and 1,200. Batches without known targets do not consume an
update. Decoded validation interval F1 selects the checkpoint; the reference
bundle uses seed 29 and update 900. Its history records the full completed budget.

The final population was reserved before fitting, independently of prior exposed
data. The model, inference policy and evaluation code were frozen before final
observations or labels were opened. All 384 experiments were processed: 128 per
length, with 64 one-instrument and 64 two-instrument experiments in each group.
This gives 576 instrument sequences and 344,064 original-grid rows. No case was
discarded or resampled to balance directions. Final scores did not choose a
different model, seed or checkpoint.

Intervals of the same direction match one-to-one at intersection-over-union of at
least 0.50, maximizing the number of matches. Macro-F1 averages all three
directions, including zero for an undefined class F1. Point macro-F1 evaluates
individual eligible steps separately. Pooled scores combine counts rather than
averaging length-specific scores.

## Results

| Sequence length | Interval macro-F1 | Point macro-F1 |
| --- | ---: | ---: |
| 256 steps | 0.8334 | 0.9130 |
| 512 steps | 0.7124 | 0.9026 |
| 1,024 steps | 0.6288 | 0.8808 |
| All lengths, pooled counts | 0.7129 | 0.8932 |

| Direction | Interval precision | Interval recall | Interval F1 |
| --- | ---: | ---: | ---: |
| Sideways | 0.6469 | 0.6547 | 0.6508 |
| Growth | 0.8121 | 0.7883 | 0.8000 |
| Decline | 0.7454 | 0.6389 | 0.6880 |

Every step eligible for direction classification is covered. On 1,024-step
sequences, the model produces 372 classified intervals against 412 references;
their median durations are 285.5 and 193.5 steps respectively. This indicates
missed structure despite strong pointwise agreement. Full coverage alone does
not establish correct directions or boundaries.

## Boundary sensitivity

Typed boundary F1 requires the correct direction on both sides of an internal
transition. Matching is one-to-one within a common available block; availability
edges are excluded. The model predicts 414 transitions against 454 references.

| Allowed displacement, steps | Typed boundary F1 |
| --- | ---: |
| 0 | 0.0392 |
| 4 | 0.1613 |
| 8 | 0.2327 |
| 16 | 0.3203 |
| 32 | 0.4124 |
| 64 | 0.4724 |

Precise placement remains weak. These tolerances describe agreement with the
automatic reference convention; they are not confidence intervals for the true
boundaries. Larger tolerances can pair distinct missed and extra transitions.
Boundary sensitivity is descriptive and does not select a preferred tolerance.

## Verification and limits

The native bundle preserves the selected weights and preprocessing without
retraining. After serialization, all probabilities match the frozen model exactly
on the complete 192-experiment validation and 384-experiment final populations.
The decoder, feature representation and availability rules are shared throughout.
Final metric reports were also recomputed from the saved probabilities.

Verification used Python 3.14.4, NumPy 2.5.4 and PyTorch 2.14.1 CPU, with four
inference threads and batches of at most 16 complete instrument sequences.
Different library versions, devices or batching can change floating-point results.
No service-latency claim follows from this scientific evaluation: queueing,
loading, rendering and HTTP are separate work.

Automatic agreement does not establish human annotation accuracy, financial
forecasting quality or transfer to another simulator. Model status therefore
is `validation_only`; publication is an operational decision. Further model
selection requires a fresh reserved benchmark, since an opened final population
cannot be reused as independent test data.
