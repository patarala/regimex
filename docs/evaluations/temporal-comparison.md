# Temporal classifiers on profile 2.2

## Results

On 120 reserved experiments (180 instrument series), the configuration selected
by mean validation interval macro-F1, TCN 48, also had the highest pooled
interval and point scores.

| Model | Parameters | Validation interval F1¹ | Held-out interval F1 | Held-out point F1 | Short sideways matched | Sideways precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic baseline | 867 | 0.5032 | 0.3834 | 0.6661 | 3/27 | 44.8% |
| TCN 32 | 35,011 | 0.6998 | 0.6254 | 0.8531 | 3/27 | 63.1% |
| TCN 48 | 77,091 | 0.7380 | 0.6751 | 0.8836 | 1/27 | 71.4% |
| TCN 32, compact dilations | 35,011 | 0.6887 | 0.5846 | 0.8732 | 3/27 | 54.7% |
| TCN 32, interval power 0.5 | 35,011 | 0.7042 | 0.6094 | 0.8594 | 2/27 | 59.8% |
| TCN 32, interval power 1 | 35,011 | 0.7079 | 0.6298 | 0.8410 | 3/27 | 59.3% |
| Patch Transformer | 35,875 | 0.6919 | 0.6038 | 0.8640 | 1/27 | 62.9% |

¹ Mean of three seed-specific best checkpoints for neural models. The baseline has one fit.

TCN 48 gained 0.0497 interval macro-F1 over TCN 32. The paired 95% interval
for that difference was −0.0117 to +0.1104. The patch Transformer used about
as many parameters as TCN 32. Its interval difference was −0.0216 (paired 95%
interval −0.0830 to +0.0407). These results support keeping the Transformer
as an offline candidate. They cover one attention architecture and one budget.

The weighting variants found two or three of the 27 short sideways intervals.
The control found three. Raising the power to 1 increased the number of false
sideways intervals from 45 to 55. TCN 48 produced 28 false sideways intervals
and recovered one short reference. Higher overall interval F1 therefore did
little for this particular failure. The 27 references occurred in only 12
independent experiments, including 19 intervals on 1,024-step runs. A difference
of one or two matches is weak evidence for ranking configurations on this metric.
All references were generated automatically.

| Held-out interval macro-F1 by model | 256 steps | 512 steps | 1,024 steps |
| --- | ---: | ---: | ---: |
| Logistic baseline | 0.5439 | 0.3682 | 0.2983 |
| TCN 32 | 0.7301 | 0.7307 | 0.4771 |
| TCN 48 | 0.8049 | 0.7590 | 0.5351 |
| TCN 32, compact dilations | 0.7621 | 0.6398 | 0.4449 |
| TCN 32, interval power 0.5 | 0.7524 | 0.6785 | 0.4797 |
| TCN 32, interval power 1 | 0.7436 | 0.7068 | 0.5074 |
| Patch Transformer | 0.7350 | 0.7304 | 0.4517 |

Each length contributed 40 experiments. At 1,024 steps every configuration
scored below its performance at 256 and 512 steps. The
[training population comparison](training-populations.md) tests the effect
of including 1,024-step runs in training on a fresh holdout. These 120
experiments count as exposed for subsequent model selection.

## What changed between candidates

All neural candidates used the same price-derived features and missing-value
indicators. The shared decoder used price availability and trading activity.
Preprocessing used training observations only. Each candidate saw the
same batches of instrument sequences and used the same loss, optimizer and
interval decoder, except for the loss weights in the two weighting variants.

| Candidate | Change from the 32-channel TCN |
| --- | --- |
| TCN 48 | Increase the hidden width from 32 to 48 channels. |
| Compact dilations | Keep eight residual blocks and 32 channels. Use dilations 1 through 8, giving the convolution stack a 73-step receptive field instead of 511 steps. Both models also receive features computed over price windows. |
| Interval weights 0.5 and 1 | Give each labeled step a weight proportional to (sequence length / interval length) raised to the chosen power. The control uses power zero. |
| Patch Transformer | Embed consecutive groups of eight steps as tokens. Two bidirectional attention layers use four heads, a hidden width of 32 and sinusoidal positions. Each step combines its patch output with its own features before classification. |

The Transformer can use later observations because Regimex classifies a completed experiment. Padding is masked separately for each instrument, including the last partial patch. Time-series forecasting research inspired the use of patches here. The local features preserve the detail needed to classify each input step. [Nie et al., 2023](https://arxiv.org/abs/2211.14730)

## Evaluation protocol

The development corpus contained 96 training and 30 validation experiments
of 256 or 512 steps. We used the same development corpus as the
[TCN width comparison](tcn-widths.md). For each neural
configuration we ran seeds 11, 23 and 47, with 96 optimizer updates per seed,
eight sequences per batch and validation at updates 32, 64 and 96. The best
decoded validation interval macro-F1 selected the checkpoint per seed. The
mean of those three scores ranked configurations. For each configuration, the
single best validation checkpoint across seeds provided its frozen weights.
The logistic baseline used its previously fitted bundle from the same corpus.

Before training we reserved 120 fresh synthetic experiments, 40 each of 256,
512 and 1,024 steps. The reserve phase checked their seeds and identity keys
against the development data and the supplied exposure index. For the results
below, the combined inventory covered 7,839 prior seeds and 35,345 identity keys.
No holdout labels were loaded during fitting. The scoring phase verified the
frozen weights and preprocessing before opening the automatic references.
The 1,024-step group tests transfer to a length absent from training.

An interval matches a reference of the same direction at intersection over
union of at least 0.5, with one match per interval. Short-sideways recall is
the share of reference sideways intervals shorter than 64 steps that match.
Sideways precision includes predicted sideways intervals of all lengths.
Confidence intervals for differences use 2,000 paired draws of whole
experiments, retaining 40 draws of each length. They describe variation
across this population for the frozen fits. The validation repeats separately
show how much checkpoint scores varied with the training seed.

## Reproduce

With the [development corpus and baseline bundle](tcn-widths.md)
in `var/tcn-widths`, run the following phases in order. Choose a new output
directory for each comparison. Add `--exposure-index PATH` to the reservation
when a prior inventory of seeds and identity digests is available.

```sh
uv run --locked --extra ml python scripts/compare_temporal_architectures.py reserve var/temporal-comparison --development var/tcn-widths
uv run --locked --extra ml python scripts/compare_temporal_architectures.py fit var/temporal-comparison --development var/tcn-widths
uv run --locked --extra ml python scripts/compare_temporal_architectures.py score var/temporal-comparison --development var/tcn-widths
```

`fitted/freeze.json` records the checkpoint for each seed, the selected seeds,
fit durations, parameter counts and hashes of the weights and preprocessing.
The reservation records the runtime versions, source tree, scripts and lockfile
used by all three phases.
`results.json` includes per-class metrics, grouped scores by length,
per-experiment counts and paired intervals. The Transformer weights are for
this offline comparison. The service loads its installed model backends.
