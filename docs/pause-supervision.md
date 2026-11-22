# Training on refined pause labels

The [second-pass annotator](label-stability.md#looking-for-a-pause-inside-an-existing-trend)
can recover a short sideways interval when a single trend spans a genuine
pause. This experiment tests whether a TCN trained with those labels
changes its predictions. At inference both models see the same observations.
The annotators supply references during training and evaluation only.

The control is the [mixed 192-experiment TCN 48](training-populations.md).
Both fits use the same instruments, preprocessing, optimizer, batch size,
96 updates per seed and seeds 11, 23 and 47. The only intervention changes
the training targets. The original 30-experiment validation set and its
reference rule select the checkpoint in both fits. This keeps validation
comparable while potentially favoring the original target definition.
Replaying the control fit reproduced its preprocessing, all 36 weight
tensors and the complete history of 288 optimizer updates. The refined
targets changed 2,008 of 172,032 instrument steps in the training data.

## Results

Each model recovered **two of 27 newly inserted pauses** on 120 fresh
exchange simulations. The 27 pauses occurred in 19 independent experiments.
The paired 95% interval for the change in recall was −11.1 to +10.0
percentage points. At this training budget the additional labels did not
yield a measurable gain in finding the added intervals.

| Reference on exchange simulations | Model targets | Interval macro-F1 | Short sideways matched | Sideways precision | False sideways |
| --- | --- | ---: | ---: | ---: | ---: |
| Current rule | Original | 0.6398 | 4/34 | 58.6% | 46 |
| Current rule | Refined | 0.6384 | 4/34 | 55.5% | 49 |
| Second-pass rule | Original | 0.5802 | 6/61 | 60.4% | 44 |
| Second-pass rule | Refined | 0.5632 | 6/61 | 57.3% | 47 |

The interval F1 difference on the current references was −0.0015 (paired
95% interval −0.0308 to +0.0250). On the second-pass references it was
−0.0170 (−0.0564 to +0.0148). The 2,000 paired draws keep entire experiments
together within each length. Both fits selected seed 47 and update 96. The
mean best validation score over the three seeds was 0.7073 with the original
targets and 0.6853 with the refined targets.

On artificial paths with known directions, interval macro-F1 went from
0.6480 to 0.6531. The models found 27/96 and 30/96 short pauses respectively.
All of these matches came from pauses before a reversal. **Neither model
matched any of the 64 pauses after which the original trend resumed.**

A follow-up inspected the per-step probabilities on these same paths.
Before decoding, the original TCN assigned the highest probability to
sideways on zero steps in all 64 same-direction pauses. The refined-target
TCN had no sideways-winning step in 61/64 and no majority in any of them.
In contrast, sideways won on a majority of the steps in 31/32 pauses before
a reversal for both models. The missed same-direction pauses are already
visible in the raw class scores. This points to testing how the model
represents local flattening between two slopes. These artificial paths
have simpler execution and quote information than exchange data.

## Evaluation

The first check uses 120 new exchange-generated experiments, with 40 each
of 256, 512 and 1,024 steps. Each length has 20 one-instrument and 20
two-instrument experiments. The two frozen models are evaluated against
both automatic reference policies. An inserted pause is a refined sideways
interval shorter than 64 steps that has no one-to-one sideways match in the
original references at overlap IoU of at least 0.5. Recovering an insertion
measures whether the student learns what its teacher added.

The second check uses 256 newly drawn, noisy paths from the eight artificial
shapes in the [annotation confirmation](label-stability.md#looking-for-a-pause-inside-an-existing-trend).
Here the true directions and boundaries come from the construction. Both
models receive a cent-denominated price and one execution per step. Order
counts and quotes are unavailable. This tests transfer to a simpler
observation process than the exchange generator.

## Reproduce

Use the frozen training bundles and panel data from the [training
comparison](training-populations.md#choosing-a-model-on-different-experiment-lengths)
and fresh output directories. The first two commands register the exchange
holdout and bind the original bundle to the training protocol. The third
reserves the artificial paths before training. Scoring requires the frozen
model pair.

```sh
uv run --locked --extra ml python scripts/confirm_pause_targets.py prepare var/pause-targets --panels var/selection
uv run --locked --extra ml python scripts/confirm_pause_targets.py reserve var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/check_pause_target_transfer.py reserve var/pause-transfer
uv run --locked --extra ml python scripts/fit_pause_targets.py var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/confirm_pause_targets.py score var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/check_pause_target_transfer.py score var/pause-transfer --training var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/diagnose_pause_probabilities.py var/pause-transfer --training var/pause-targets --populations var/populations
```
