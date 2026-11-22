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
