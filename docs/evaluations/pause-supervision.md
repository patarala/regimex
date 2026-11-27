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

Run the refined-target comparison in its original checkout, which includes
the per-step diagnostic used below. [Source at f65e4ce](https://github.com/patarala/regimex/tree/f65e4ce4c0a18d8e9306aee44551d38106c48c8b/scripts)

Use the frozen
training bundles and panel data from the [training
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

## Using slope contrasts to find a pause

We compared two ways of finding a brief interruption in a trend. Both use
the preceding experiment's TCN trained on refined targets as their control.

- **Additional TCN inputs.** A 32- or 48-step center window sits between
  two disjoint 32-step flanks. For each center width the model receives two
  numbers measuring directional agreement between the flanks and the
  flatness of the center relative to them.
  Each window needs at least 80% observed prices and uses their original
  timestamps. Four numeric features plus four availability indicators add
  384 trainable input weights. The added weights start at zero. Training
  reuses the control's data, target labels, validation references,
  initialization of shared weights and three-seed update budget.
- **Check the model's decoded trends.** The control TCN first classifies the
  entire sequence. For each predicted trend long enough to contain two flanks
  and a pause, the [frozen price-based rule](label-stability.md#looking-for-a-pause-inside-an-existing-trend)
  can insert one sideways interval. It uses prices within that predicted
  interval and preserves the model's other segments and gaps. This requires
  a single TCN forward pass.

We registered 144 new exchange simulations, evenly split across 256, 512
and 1,024 steps and one or two instruments. The secondary check has 768
independent artificial paths, 96 per shape. Of those, 192 contain a 32-
or 48-step pause followed by the original trend. Another 96 contain a
slowdown that remains directional throughout. All comparisons are paired.
Bootstrap intervals use 2,000 draws of whole experiments within length on
the exchange simulations and of whole paths within shape on the artificial
signals.

### What improved

The new input channels raised mean validation interval macro-F1 from 0.6853
to 0.6961. Both fits chose seed 47 at update 96. Almost all of the gain came
from that seed. The candidate had 77,475 parameters to the control's 77,091.
The apparent validation gain did not carry over to the independent exchange
simulations.

| Exchange simulations (144) | New short pauses matched | Interval F1, current references | Interval F1, refined references | False sideways, current references |
| --- | ---: | ---: | ---: | ---: |
| Control TCN | 2/35 | 0.6960 | 0.6414 | 58 |
| TCN with slope inputs | 3/35 | 0.6760 | 0.6322 | 70 |
| Control TCN with a price check | 25/35 | 0.6621 | 0.7333 | 88 |

The 35 newly inserted short pauses occurred in 25 of the 144 experiments.
The price check increased their recall by 65.7 percentage points (paired
95% interval +50.0 to +82.1). Its F1 difference against the control was
−0.0339 on the current references (−0.0631 to −0.0059) and +0.0918 on
the refined references (+0.0589 to +0.1265). The refined teacher uses the
same price check within its own trends, so agreement on exchange data
alone cannot establish that the additional pauses were genuine.

| Artificial paths (768) | Pauses followed by the original trend | Continuous slowdowns called sideways | Pooled interval F1 |
| --- | ---: | ---: | ---: |
| Control TCN | 3/192 | 0/96 | 0.6672 |
| TCN with slope inputs | 4/192 | 0/96 | 0.6738 |
| Control TCN with a price check | 175/192 | 7/96 | 0.8635 |

The price check recovered 82/96 true 32-step pauses and 93/96 true 48-step
pauses. Its pooled F1 gain was +0.1963 (paired 95% interval +0.1829 to
+0.2081). All seven false pauses on the slowdown cases occurred on different
paths. For the learned slope inputs, the raw sideways probability won on
a majority of steps in just 4/192 same-direction pauses, compared with
3/192 for the control. The price check is available in the offline preview
for inspecting interruptions inside a broad trend. The original decoder
scored better against the service's current reference definition.

On the earlier validation timing sample (12 experiments, 18 series,
10,752 steps), median CPU prediction and decoding time was 0.546 s for the
control, 0.555 s for the extra inputs and 0.593 s for the price check.
Each method had one warm-up and six timed runs covering all orderings of the
three methods, using two threads and batches of up to eight. The observed
ranges overlapped. These are local warm-model timings.

### Reproduce the slope comparison

At revision `f609e6d`, use the three completed datasets from the earlier
comparisons and fresh directories for both new checks. The reserve phases fix the cohort and the
rules before the new weights are fitted.

```sh
uv run --locked --extra ml python scripts/confirm_local_pause_features.py prepare var/local-pauses --previous var/pause-targets
uv run --locked --extra ml python scripts/confirm_local_pause_features.py reserve var/local-pauses --previous var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/check_local_pause_transfer.py reserve var/local-transfer
uv run --locked --extra ml python scripts/fit_local_pause_features.py var/local-pauses --previous var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/confirm_local_pause_features.py score var/local-pauses --previous var/pause-targets --populations var/populations
uv run --locked --extra ml python scripts/check_local_pause_transfer.py score var/local-transfer --training var/local-pauses --previous var/pause-targets --populations var/populations
```

Run the timing and preview commands from the checkout that produced the
figures, keeping the output directories from the comparison.
[Timing and preview scripts at 1b9d7c3](https://github.com/patarala/regimex/tree/1b9d7c3291bd643ce20f57e97ee03b0e3e7d6405/scripts)

```sh
uv run --locked --extra ml python scripts/time_local_pause_features.py var/local-pauses --panels var/selection --previous var/pause-targets --populations var/populations
```

To inspect individual steps, export the interactive comparison below and
open the HTML file in a browser. It shows the first rising path at the
higher noise level in the first group for two scenarios, a 32-step pause
and a continuous slowdown. Each chart includes the known direction and all
three sets of predicted intervals. The default high-noise example includes
a genuine pause that all three methods miss. Add `--noise low` to display
the quieter pair of paths.

```sh
uv run --locked --extra ml --extra viz python scripts/preview_local_pauses.py --comparison var/local-transfer var/local-transfer/examples.html --training var/local-pauses --previous var/pause-targets --populations var/populations
```

To compare predictions on your own simulation, replace `--comparison
var/local-transfer` with `--export PATH` pointing to an experiment-v2 JSON
file and choose a new output file. Each instrument gets its own price chart.
Missing prices leave gaps and unavailable intervals have their own legend entries.

The [report overlay comparison](trend-pause-detection.md) tests whether short
pauses can be highlighted within the TCN's original trend intervals, including
several pauses in one trend and periods with infrequent executions.
