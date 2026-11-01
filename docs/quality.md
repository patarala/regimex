# Model quality

These results measure agreement with automatic reference policy
`retrospective-directions/3.0`. Independent human accuracy and transfer to another
market simulator have not been measured. The current models are an aggregate
logistic baseline and a temporal convolutional network (TCN).

## Evaluation protocol

Both models use 128 training experiments and 32 validation experiments, with
whole experiment/seed groups kept together and preprocessing fitted on training
only. A separate 64-experiment final dataset was reserved before fitting. Model
settings, runtime and evaluation rules were fixed before it was opened. All
64 experiments were scored: 32 with one instrument and 32 with two, totaling
96 instrument series and 49,152 original-grid rows. No cases were discarded or
resampled to balance classes. The selected TCN is validation epoch 1, seed 11.

The primary metric is interval macro-F1. Intervals of the same direction match
one-to-one at intersection-over-union of at least 0.50, maximizing the number of
matches. F1 is averaged across all three directions; undefined class F1 is zero.
Pointwise macro-F1 is separate. Boundary F1 matches internal transition times
within four steps; typed boundary F1 additionally requires the correct directions
on both sides. Availability edges are excluded.

## Results

| Model | Interval macro-F1 | Point macro-F1 | Boundary F1 | Typed boundary F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.470017 | 0.740093 | 0.055172 | 0.013793 |
| TCN | 0.639593 | 0.856032 | 0.030534 | 0.015267 |

The TCN's interval macro-F1 exceeds the baseline by **0.1695767011** on these
experiments.

Whole-experiment bootstrap 95% intervals are [0.376219, 0.565661] for baseline and
[0.562198, 0.731333] for TCN. The paired difference interval is
[0.076998, 0.265089], using 1,000 repetitions with seed 1729. Both models receive
the same sampled experiment identities; F1 is calculated from pooled counts.
These intervals measure experiment sampling variation conditional on these
models/references, not label correctness or training-seed variation. They do not
guarantee a population score.

Boundary placement is weak. TCN matches two of 84 transition times and only one
with the correct directions on both sides. It predicts 132 intervals against
169 references, with median duration 344 versus 207 steps. It covers all 40,870
eligible steps and produces no one- or two-step fragments; coverage does not
establish correct boundaries.

## Processing time

| Measurement | Baseline | TCN |
| --- | ---: | ---: |
| Maximum over all 64 final experiments, 512 steps, including first prediction | 0.317374 s | 1.938387 s |
| Maximum of separate warm repetitions, 512 steps | 0.258489 s | 0.140134 s |
| Maximum of 4,096-step, two-instrument stress repetitions | 4.383598 s | 2.969163 s |

Times include input construction, features, prediction, probability validation,
decoding and interval statistics **after model loading**. They exclude backend
import, loading, queue delay, HTML, HTTP and file transfer. Warm timings do not
replace the first-use measurement.

The measurements used Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 CPU and
scikit-learn 1.9.1, with four threads and a 4 GiB process address-space limit.
Execution time depends on the deployment environment; the recorded software
versions and limits do not guarantee reproducible timings. Saved-model
restoration and release-asset inference matched the recorded probabilities
exactly during verification.

The release's `evaluation/` directory holds current metrics, data identities and
reproduction instructions. Bundle training metadata remains `validation_only`;
final evaluation is a separate record. Publication does not establish independent
predictive accuracy. Additional model selection requires a newly reserved final
dataset; an opened test set cannot serve as fresh held-out evaluation data.
