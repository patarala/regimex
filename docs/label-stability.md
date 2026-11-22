# Stability of automatic direction labels

The TCN width comparison supplied only 12 sideways intervals
shorter than 64 steps for training and 10 for validation. These intervals came
from 19 of the 126 development experiments. A change of at most one tick to the
published prices preserved 21 of the 22, including both endpoints. The small
number of training examples makes short plateaus a candidate for targeted
sampling. Stability measures agreement with the automatic rule under specific
changes. Human agreement has yet to be measured.
[Development corpus](tcn.md#comparing-model-widths)

## What changes the labels?

Each count shows how many original intervals still had a one-to-one match with
the same direction and an intersection-over-union of at least 0.5. Short means
under 64 steps. The step column counts originally classifiable steps that kept
their direction, including those in intervals that span the scoring cutoff.

| Change | Short sideways | Short growth | Short decline | Sideways ≥64 | Growth ≥64 | Decline ≥64 | Same direction, steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round closes to two ticks | 21/22 | 9/10 | 20/20 | 70/74 | 75/76 | 54/55 | 63,750/64,977 |
| Remove the last 32 steps | 18/18 | 8/8 | 18/18 | 10/10 | 19/19 | 12/12 | 53,503/54,301 |
| Lower the movement threshold from 0.50% to 0.45% | 22/22 | 10/10 | 20/20 | 74/74 | 76/76 | 55/55 | 64,977/64,977 |
| Raise it from 0.50% to 0.55% | 20/22 | 9/10 | 19/20 | 74/74 | 74/76 | 55/55 | 64,725/64,977 |

- **Price precision.** Rounding to the nearest multiple of two ticks changed
  35,721 of 72,448 closes by one tick. Repeated prices stayed equal. One short
  sideways interval changed direction on a majority of its steps. Looking in
  reverse, two of the 23 short sideways intervals on the coarser grid had no
  matching interval on the original grid.
- **Recording length.** After dropping the final 32 steps, scoring also excluded
  the 32 steps immediately before the cut. All 85 original intervals ending by
  that cutoff still matched. Of the 18 short sideways intervals, 17 also kept
  identical endpoints. One short sideways interval on the shortened recording
  lacked a counterpart. An interval extending past the cutoff could still
  match a complete interval on the other timeline.
- **Direction threshold.** At 0.45% all 257 original intervals had identical
  directions and endpoints. At 0.55%, two short sideways intervals lost their
  match, but neither changed direction on a majority of its steps. A change in
  duration or a merge can lose an interval match while keeping its steps sideways.

All four comparisons preserved the availability mask on the scored steps. The
prefix comparison covers 85 complete intervals of the original 257 because it
omits intervals that reach into the excluded tail.

## Can the rule detect a known pause?

We also checked 288 independently specified price paths, each 192 steps long.
The six shapes covered a flat price, a single level shift, a slow trend,
a reversal, and a 32- or 48-step pause inside a continuing trend. Each shape
had 48 examples spanning two noise levels, both signs and 12 repetitions.
The expected directions
and boundaries came from the clean path. The annotator saw prices with
autocorrelated noise, rounded to cents.

| Split penalty | 32-step pauses matched | 48-step pauses matched | False trend intervals on flat paths | False trend intervals on level shifts | Pooled interval macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 23/48 | 33/48 | 16 | 9 | 0.7651 |
| 8 (current) | 10/48 | 31/48 | 3 | 2 | 0.7662 |
| 16 | 0/48 | 22/48 | 1 | 0 | 0.6753 |

Halving the penalty recovered 13 more of the shortest pauses and introduced
20 additional false trend intervals across the 96 flat and level-shift paths.
This helps explain how stable labels can still miss a pause shorter than the
surrounding trends. These six shapes use one noise process and an observed
price at every step. Their frequencies and trade-offs can differ on exchange
data. The comparison supports keeping the current default while testing ways
to recognize short pauses without splitting noisy flat periods.

## Looking for a pause inside an existing trend

A second offline annotator revisits each growth or decline interval long
enough for two flanks and a pause. It compares a straight line with a
continuous fit consisting of a slope, a flat section and another slope.
The central section must be substantially flatter than both flanks. Both
outer segments must independently qualify as trends in the original
direction. The method can add one pause per original trend and leaves the
availability boundaries in place. At the 32-step scale it checks pause
widths of 32, 40, 48 and 56 steps.

We chose the detection thresholds on the original 288 diagnostic paths.
Confirmation compared the frozen rule, the current annotator and the
lower split penalty on two fresh suites. The first repeated the original
six shapes with new random draws (384 paths). The second used 512 paths of
224 or 320 steps with variable break positions, heavier-tailed noise and
two tick sizes. Its controls included a trend that slows down but continues
moving and a pause followed by a reversal. Scores use the latent path as
the reference. Both suites record a trade and a price at every step.

| Suite | Method | Interval macro-F1 | Short pauses matched | False trends on flat paths or level shifts | False sideways on trends or reversals |
| --- | --- | ---: | ---: | ---: | ---: |
| Original shapes, new draws | Current rule | 0.7573 | 54/128 | 4 | 20 |
| | Lower split penalty | 0.7949 | 76/128 | 13 | 45 |
| | Second pass | 0.8916 | 96/128 | 4 | 23 |
| Different lengths, shapes and noise | Current rule | 0.8802 | 129/192 | 8 | 17 |
| | Lower split penalty | 0.8656 | 140/192 | 19 | 30 |
| | Second pass | 0.9164 | 157/192 | 8 | 21 |

The second pass gained 0.1343 interval F1 on the first suite (paired 95%
interval +0.1014 to +0.1685) and 0.0362 on the second (+0.0201 to +0.0534).
Intervals use 2,000 paired draws of whole paths within shape. Most of the
gain came from same-direction trends interrupted by a 32-step pause. On
the second suite the method recovered 52/64 such pauses, up from 32/64.
All four additional false sideways intervals on trends in that suite came
from slowdowns. Pause matches before reversals stayed at 50/64.

On 192 exchange-generated training experiments the second pass inserted 51
short sideways intervals across 36 experiments. It relabeled 2,008 of
139,825 classified steps (1.44%) and preserved every unavailable interval.
After rounding observed prices to a two-tick grid, all 51 insertions still
matched a sideways interval, 47 with identical endpoints. Fifty still
counted as new against the current rule on the rounded paths. These are
measurements of stability on the exchange data, where the true directions
are unknown. The candidate has a separate offline policy identifier.
The [training target comparison](pause-supervision.md) measures what a TCN
learns from the additional intervals.

## Reproduce

Use `direction-32/2.2` on the locked synthetic corpus with 96 training and 30
validation experiments, 256 or 512 steps long, and 189 instrument sequences.
The script verifies each snapshot against the manifest before relabeling it.

```sh
uv run --locked --extra ml python scripts/check_label_stability.py \
  var/verification/tcn-width-profile22
```

For a fresh checkout, add `--generate` and give the script a new directory.
Generation recreates the original split. The audit reads observations from its
training and validation partitions only. The JSON output includes counts for
exact endpoints and unmatched intervals in each direction and duration group.

The controlled paths use a fixed random seed and require no corpus. Choose a
new file for their scores.

```sh
uv run --locked --extra ml python scripts/check_annotation_penalties.py \
  var/verification/annotation-penalties.json
```

To reproduce the two independent pause checks, use a new directory for the
reservation, the candidate digest and the resulting per-path interval counts.
The exchange audit uses only the training partition of the
[mixed 192-experiment dataset](training-populations.md#reproduce).

```sh
uv run --locked --extra ml python scripts/confirm_trend_pauses.py reserve var/trend-pauses
uv run --locked --extra ml python scripts/confirm_trend_pauses.py freeze var/trend-pauses
uv run --locked --extra ml python scripts/confirm_trend_pauses.py score var/trend-pauses
uv run --locked --extra ml python scripts/audit_pause_refinement.py var/populations/development/mixed-192 var/trend-pauses/exchange.json
```
