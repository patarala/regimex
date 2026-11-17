# Model quality

On 384 held-out synthetic experiments, the reference model `tcn-0.2.0` scored
**0.7129 interval macro-F1** and **0.8932 point macro-F1**. Both scores measure
agreement with automatic labels under `retrospective-directions/3.0` and the
model's `direction-32/2.0` profile. Human agreement, performance on another
simulator and results for the current training profile are unmeasured. The
baseline has no score on this benchmark. [Interval definitions](models.md)

## Results

| Experiment length | Interval macro-F1 | Point macro-F1 |
| --- | ---: | ---: |
| 256 steps | 0.8334 | 0.9130 |
| 512 steps | 0.7124 | 0.9026 |
| 1,024 steps | 0.6288 | 0.8808 |
| All lengths, pooled | **0.7129** | **0.8932** |

An interval matches a reference when the directions agree and their overlap
covers at least half their union. Each interval can match once, and the evaluator
finds the largest set of valid matches. Point F1 instead scores individual steps
with a reference direction. For both metrics, macro-F1 gives equal weight to all
three directions and assigns zero to an undefined class F1. The pooled scores
use counts across all experiments.

| Direction | Interval precision | Interval recall | Interval F1 |
| --- | ---: | ---: | ---: |
| Sideways | 0.6469 | 0.6547 | 0.6508 |
| Growth | 0.8121 | 0.7883 | 0.8000 |
| Decline | 0.7454 | 0.6389 | 0.6880 |

The model classified all 283,946 steps that had a reference direction. On the
longest experiments it produced 372 intervals against 412 references. Its
median interval lasted 285.5 steps, compared with 193.5 for the references.
This helps explain why the model can agree on many individual steps while
missing shorter intervals.

## Boundary sensitivity

A typed boundary records both directions of a transition, such as growth to
decline. The evaluator pairs predicted and reference transitions of the same
type one-to-one within a continuous classified block. Intervals marked
`no_trades` or `insufficient_data` separate blocks. The model produced 414
transitions against 454 references.

| Allowed displacement, steps | Typed boundary F1 |
| --- | ---: |
| 0 | 0.0392 |
| 4 | 0.1613 |
| 8 | 0.2327 |
| 16 | 0.3203 |
| 32 | 0.4124 |
| 64 | 0.4724 |

Few boundaries fall on the exact reference step. Larger tolerances admit
nearby transitions and can pair distinct changes. These scores describe how
sensitive agreement is to placement under the automatic labeling rule.

## Evaluation protocol

Training used 512 experiments of 512 steps (769 instrument sequences).
Validation used 192 experiments, with 64 each at 256, 512 and 1,024 steps.
Whole experiments and related seed groups stayed in one partition.
Preprocessing was fitted on training data.

Each training seed ran for 1,200 optimizer updates. Validation scored updates
300, 600, 900 and 1,200. The selected checkpoint was update 900 for seed 29.
The model, inference settings and evaluation code were frozen before opening
the reserved final data.

The final population included 128 experiments per length, half with one
instrument and half with two. All 576 instrument sequences and 344,064 grid
rows were retained. Further model selection requires a fresh final benchmark.

## Reproducing the scores

The evaluation assets for `tcn-0.2.0` contain the references, per-step
probabilities and metric reports. After unpacking the assets under `release/`
in the checkout, recompute the scores with this command.

```sh
uv run --locked --extra ml python release/evaluation/reproduce.py score --output var/reproduced-scores
```

Choose a new output directory. Add `--predict` to run the saved model as well.
The recorded verification reproduced its validation and final probabilities
exactly with Python 3.14.4, NumPy 2.5.4 and PyTorch 2.14.1 on CPU.
