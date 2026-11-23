# Training the direction forest on 1,536 experiments

This comparison tests whether more data and a different way of measuring
price movement improve the interval predictions of the
[released direction forest](direction-context.md). Every candidate predicts
sideways, growth and decline from observed prices. Seeds, simulator settings
and reference intervals are excluded from its inputs.

## Data and selection

| Purpose | Experiments | Instrument sequences | Steps across instruments |
| --- | ---: | ---: | ---: |
| Training | 1,536 | 2,304 | 1,376,256 |
| Model selection | 192 | 288 | 172,032 |
| Final comparison | 384 | 576 | 344,064 |

Each partition has equal numbers of 256-, 512- and 1,024-step experiments.
At each length, half have one instrument and half have two. Both instruments
of an experiment stay in the same partition. The population excludes seeds
and dataset identity keys from previous comparisons. The training and
validation labels come from the automatic `direction-32/2.2` profile.
Of the training steps, 1,174,557 carry direction labels (469,974 sideways,
412,022 growth and 292,561 decline). Steps without a direction label do not
contribute to the fit. One proposed training seed reproduced observations
from an earlier experiment. The generator replaced it with the next
eligible seed of the same length and instrument count before model fitting.

The release used 40 trees of depth at most 12. The three new forests used
80 trees of depth at most 15 and seed 173. **Greedy** keeps the released
price features. **Multiscale** fits the price path at three boundary
penalties and measures trend on either side of a possible price-level shift.
**Multiscale with interval weights** also gives short labeled intervals more
influence within their direction class. These options are available in the
[training form and JSON configuration](models.md#adaptive-direction-forest).

Selection uses pooled interval macro-F1 on the 192 validation experiments.
The scorer freezes the highest-scoring new bundle before generating final
references. Both the release and that challenger are evaluated on the same
384 final experiments. The paired 95% interval resamples whole experiments
within each length. The separate transfer check uses 32 fresh seed groups
and 384 price paths with direction specified by the generator before adding
noise. Each path is evaluated twice, with and without a gap.

## Results

| Model | Validation interval macro-F1 | Final interval macro-F1 | Final point macro-F1 |
| --- | ---: | ---: | ---: |
| Released forest | 0.7887 | 0.7727 | 0.9323 |
| Greedy | 0.7858 | | |
| Multiscale | 0.9461 | | |
| Multiscale with interval weights | 0.9533 | 0.9303 | 0.9798 |

Only the release and the validation winner entered the final comparison.
The winner recovered 920 of 985 reference intervals and 118 of 130 short
sideways intervals. The release recovered 717 and 46, respectively. The
interval macro-F1 gain was 0.1576 (paired 95% interval 0.1261 to 0.1877).
The improvement held at all three experiment lengths.

On the paired paths with known drift, interval macro-F1 fell from 0.9109
for the release to 0.8539 for the new model. The new model recovered more
of the 128 short pauses in complete views (103 versus 86), but scored lower
across all sideways intervals (F1 0.7384 versus 0.8216). Its output was
slightly more stable when a gap was introduced. Predictions changed on
2.96% of comparable steps, versus 3.15% for the release.

The new model is a closer match to the automatic profile. Its multiscale
price features use the same family of piecewise-linear fits as that profile.
It did not qualify to replace the release across both populations. Neither
evaluation measures accuracy against human labels or on real markets.

## Reproduce

Use the locked ML environment and the [original control bundle](direction-context.md#use-the-trained-forest).
The data archive unpacks to `direction-large-v1/` and contains the registered
splits, relative export paths and automatic training and validation references.
From the repository root, unpack it under `var/verification`. Use empty output
directories for a new comparison.

```sh
mkdir -p var/verification
tar -xzf var/release/direction-large-v1/direction-large-v1-data.tar.gz -C var/verification
run() {
  script=$1
  shift
  uv run --locked --extra ml python "scripts/$script.py" "$@"
}
run evaluate_large_direction register
run check_large_direction_transfer reserve
run train_large_direction var/verification/direction-large-v1 --workers 3
run evaluate_large_direction freeze
run evaluate_large_direction score
run check_large_direction_transfer score
```

To regenerate the observation archive from its original inputs, run the
following in place of the unpack step. The inventory, dataset and previous
transfer plan come from the [previous forest comparison](adaptive-intervals.md#training-on-more-experiments).

```sh
uv run --locked --extra ml python scripts/prepare_large_direction.py \
  var/verification/direction-large-v1 \
  --inventory var/verification/forest-training-mix/prior.json \
  --dataset var/verification/forest-training-mix/dataset.json \
  --transfer var/verification/forest-training-transfer/plan.json --workers 5
```
