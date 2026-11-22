# Training data for TCN 48

## Results

Doubling the number of short training experiments raised held-out interval
macro-F1 from 0.7086 to 0.7243. The paired 95% interval for this gain spans
zero. Adding 1,024-step experiments to training showed little benefit for
overall interval F1 at either cohort size. The mixed 192-experiment recipe
found the most short sideways intervals, 13 of 60, but also reported more
false sideways intervals than the original recipe.

| Training recipe | Mean validation interval F1 | Holdout interval F1 | Holdout point F1 | Interval F1 at 1,024 steps | Short sideways matched | Sideways precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short, 96 | **0.7380** | 0.7086 | 0.9138 | 0.6845 | 4/60 | 76.6% |
| Mixed, 96 | 0.6548 | 0.7066 | 0.9096 | 0.6632 | 7/60 | 68.2% |
| Short, 192 | 0.7042 | 0.7243 | 0.9159 | 0.6941 | 7/60 | 70.6% |
| Mixed, 192 | 0.7073 | 0.7147 | 0.9041 | 0.6890 | 13/60 | 70.0% |
| Mixed, 192, double updates | 0.7175 | 0.6761 | 0.8849 | 0.6365 | 8/60 | 65.6% |
| Mixed, 96, more short sideways | 0.6860 | 0.6909 | 0.9011 | 0.6686 | 9/60 | 65.9% |

The 60 short sideways references span 39 of 192 independent experiments.
Sideways precision counts predictions of every duration. Short sideways
recall counts only reference intervals shorter than 64 steps. The original
short recipe won on validation before the holdout was opened. The short
192-experiment recipe had the highest score on this holdout. Selecting it
for deployment on that basis would require another independent evaluation.

| Change at fixed architecture | Difference in pooled interval F1 | Paired 95% interval |
| --- | ---: | ---: |
| 96 → 192 short experiments | +0.0157 | −0.0119 to +0.0456 |
| 96 → 192 mixed experiments | +0.0081 | −0.0196 to +0.0322 |
| Short → mixed at 96 experiments | −0.0020 | −0.0361 to +0.0319 |
| Short → mixed at 192 experiments | −0.0096 | −0.0419 to +0.0235 |
| 96 → 192 updates on the same 192 mixed experiments | −0.0386 | −0.0703 to −0.0058 |
| More short sideways examples at 96 mixed experiments | −0.0157 | −0.0490 to +0.0192 |

Targeted sampling raised short matches from seven to nine at the cost of six
additional false sideways intervals (50 to 56). The paired interval for the
change in short recall was −4.0 to +11.5 percentage points. For the doubled
update budget, validation preferred a later checkpoint while the independent
interval score fell. These results support retaining the original recipe
and including all three target lengths in the next validation population.

Each interval uses 2,000 paired bootstrap draws of whole experiments within
length, keeping instruments from the same experiment together. Intervals are
pointwise for each comparison and conditional on the fitted models. Variation
from retraining and agreement with human labels are separate questions.

## What the comparison measures

The [architecture comparison](temporal-comparison.md) found that TCN 48
struggled with long experiments and short sideways intervals. This comparison
keeps its 77,091 parameters, observation features, decoder and annotation
profile (`direction-32/2.2`) fixed. It changes the number of training
experiments, the length mix, the frequency of short sideways examples or the
optimizer budget.

| Training recipe | Experiments of 256 / 512 / 1,024 steps | Instrument steps in training | Updates per seed |
| --- | --- | ---: | ---: |
| Short, 96 | 48 / 48 / 0 | 56,576 | 96 |
| Mixed, 96 | 32 / 32 / 32 | 86,016 | 96 |
| Short, 192 | 96 / 96 / 0 | 111,872 | 96 |
| Mixed, 192 | 64 / 64 / 64 | 172,032 | 96 |
| Mixed, 192, double updates | 64 / 64 / 64 | 172,032 | 192 |
| Mixed, 96, more short sideways | 32 / 32 / 32 | 86,016 | 96 |

The last recipe swaps 17 training experiments for others from the 192-member
mixed training pool. Each swap preserves experiment length and instrument
count. This increases the number of reference sideways intervals under 64
steps from 25 to 57 at the same number of instrument steps and optimizer
updates. The rule reads training labels only. An independent
[audit of the labels](label-stability.md) checks how readily the reference
rule itself can detect a short pause.

All configurations use batches of eight instrument sequences and seeds 11, 23
and 47. Each seed has three validation checkpoints. The configuration with
the highest mean of the three seed-specific best scores wins. Within each
configuration the highest-scoring seed and checkpoint supply the evaluated
weights. The same 30 validation experiments have 256 or 512 steps. Length
generalization is measured on the independent holdout, which includes 64
experiments at each of 256, 512 and 1,024 steps, half with two instruments.
At a fixed update count, a batch of longer sequences can still entail more
computation and supervised steps.

The short 96-experiment control exactly reproduced the three best validation
scores and selected checkpoints from the earlier TCN 48 fit. At 96 updates
the control processed about 286,000 actual instrument steps per seed. The
96-experiment mixed recipe processed about 460,000 per seed because its
sequences were longer. Both counts include repeated visits to training data
and exclude padding.

## Reproduce

The preparation and reservation commands need an inventory of previously
used experiments. On a fresh checkout, recreate the two preceding populations
with the commands below. They require the locked ML dependencies. If you
already have these manifests, use their paths directly.

```sh
uv run --locked --extra ml python scripts/check_label_stability.py var/tcn-widths --generate
uv run --locked --extra ml python scripts/compare_temporal_architectures.py reserve var/temporal-prior --development var/tcn-widths
uv run --locked --extra ml python scripts/build_exposure_index.py var/prior.json --dataset var/tcn-widths/dataset.json --dataset var/temporal-prior/holdout/dataset.json
```

Add earlier inventories with `--index PATH` and any other previously used
datasets with `--dataset PATH`. The comparison reported here checked 7,959
historical seeds and 36,005 identity keys before adding the development
experiments. A locally rebuilt inventory can certify independence only
against the inputs you supply.

Choose a new destination. Preparation reuses the original training and
validation populations. Reservation checks the independent holdout against
the exposure inventory and the new training population. Fit freezes all six
candidates before score can read the holdout reference intervals.

```sh
uv run --locked --extra ml python scripts/compare_training_populations.py prepare var/populations --development var/tcn-widths --exposure-index var/prior.json
uv run --locked --extra ml python scripts/confirm_training_populations.py reserve var/populations --original var/tcn-widths --exposure-index var/prior.json
uv run --locked --extra ml python scripts/compare_training_populations.py fit var/populations
uv run --locked --extra ml python scripts/confirm_training_populations.py score var/populations
```

`fitted/freeze.json` identifies the validation-selected recipe and bundle
digest. Each `fitted/<recipe>/bundle/` contains weights and preprocessing in
the native TCN format. `confirmation/results.json` contains per-experiment
counts, scores by length and paired differences between recipes.
