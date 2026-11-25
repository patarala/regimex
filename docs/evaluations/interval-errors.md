# Sideways intervals on validation data

The frozen `profile21-tcn` candidate matched 15 of the 26 automatically
labeled sideways intervals across 24 validation experiments and 36 instrument
sequences. A match uses the evaluator's one-to-one assignment and requires
intersection over union of at least 0.5. Recall uses the number of reference
sideways intervals as its denominator.

| Reference duration (steps) | Reference | Matched | Missed | Recall |
| --- | ---: | ---: | ---: | ---: |
| 1–63 | 6 | 0 | 6 | 0% |
| 64–127 | 4 | 3 | 1 | 75% |
| 128–255 | 4 | 3 | 1 | 75% |
| 256 or more | 12 | 9 | 3 | 75% |
| All durations | 26 | 15 | 11 | 57.7% |

## Did the network or the decoder miss the short intervals?

Before decoding, each step has three probabilities from the TCN. A direction
leads when its probability is strictly higher than both alternatives. Exact
ties count separately. The decoder uses the scores across an available block
and accounts for minimum duration and the cost of switching direction.

| Missed sideways references | Intervals | Steps | Raw sideways leads | Raw growth leads | Raw decline leads | Decoded sideways steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Under 32 steps, labeled growth on the reference bounds | 3 | 56 | 0 | 56 | 0 | 0 |
| 32–63 steps, absorbed by a neighboring trend | 3 | 121 | 4 | 85 | 32 | 0 |
| All under 64 steps | 6 | 177 | 4 | 141 | 32 | 0 |

Five of the six references had a trend ahead of sideways on every step.
Only the 39-step XRP/USD interval `[244, 283)` contained any raw sideways
leads. The decoder assigned a trend to all four of those steps. This gives a
reason to investigate the network scores first. These six intervals come
from the validation split used to choose the checkpoint.

Longer misses can have a different explanation. On the 76-step ETH/USD
reference, sideways led on 72 steps and the decoder assigned sideways to all
76. Its predicted interval extended across another reference sideways period,
so the one-to-one interval match still failed.

## How the misses are counted

The script assigns one outcome to each unmatched reference, using the first
applicable row below. Overlap requires at least one shared step. All predicted
pieces touching the interior of the reference are recorded with their original
bounds, so overlapping explanations can be inspected on the chart.

| Outcome | Rule for an unmatched reference sideways interval | Count |
| --- | --- | ---: |
| Split | Two or more predicted sideways intervals overlap it. | 1 |
| Merged | A single predicted sideways interval overlaps this and another reference sideways interval. | 1 |
| Low overlap | Only one predicted sideways interval overlaps it and their overlap is too small to match. | 3 |
| Unavailable | Every step received an unavailable status. | 0 |
| Absorbed by trend | One predicted trend covers it completely and continues into an adjacent reference trend of the same direction. | 3 |
| Wrong direction | The remaining cases have at least one predicted trend step and no predicted sideways steps. | 3 |

For example, the SOL/USD reference spanning steps 148–531 in snapshot
`43d93f62-4b02-502e-a9cc-d3aa841c099e` contains two predicted sideways
pieces separated by growth. The best overlap was 0.499. The chart shows why
this particular miss sits close to the scoring threshold.

## Reproduce the report

Use the worktree and the `prepare` and `fit` steps from the
[reference comparison](reference-releases.md#reproduce-the-comparison). Run the diagnosis from
the repository root, choosing a new or empty output directory.

```sh
uv run --locked --extra ml --extra viz python scripts/diagnose_intervals.py var/profile21 var/sideways-errors
```

Open `var/sideways-errors/index.html` to inspect all eleven misses by outcome.
Each chart shows price, the three raw probability curves, and the reference
and decoded intervals on a shared step axis. Hover over a curve to read the
probability at that step. Gaps in the curves mark unavailable reference
blocks. The dotted lines mark the selected reference bounds. Keep the
generated directory together when opening it offline.

For each of the 26 sideways references, `summary.json` records the raw leader
counts, decoded sideways steps and steps where both agreed on sideways.
It also records overlapping predictions and the digests of the model, dataset
and validation references. The script loads only the validation partition and
uses the profile stored in the candidate.

## Past and future price slopes

A centered price window can include both a plateau and the trend beside it.
To test whether the TCN could distinguish them, we added separate slopes for
the prices before and after each step. Both configurations used the same 96
training experiments, seed, profile and budget of 240 updates. Decoded interval
macro-F1 on 24 validation experiments selected update 160 for the centered
model and update 240 for the model with extra slopes. Both missed all six
short sideways references on validation. Macro-F1 fell from 0.6655 to 0.6269
with the extra slopes, so the centered windows remained the default.

We froze both checkpoints before evaluating them on 96 new simulations,
32 each at 256, 512 and 1,024 steps. Predictions covered the same 66,190
classifiable steps across 144 instrument sequences. The table gives pooled
results for automatic `direction-32/2.1` references. The last column gives
the 95% interval for the change from adding slopes, using 1,000 paired
bootstrap samples of whole experiments.

| Metric | Centered windows | With extra slopes | 95% interval for the change |
| --- | ---: | ---: | ---: |
| Recall of sideways intervals under 64 steps | 11.8% (6/51) | 3.9% (2/51) | −21.7 to 0.0 percentage points |
| Interval macro-F1, all directions | 0.6229 | 0.6270 | −0.0342 to +0.0481 |
| Sideways interval precision, all durations | 56.6% | 64.6% | −0.8 to +17.3 percentage points |

The extra slopes produced fewer false sideways intervals (29 versus 43) and
also recovered fewer short plateaus. There is little evidence of a change in
overall F1. The comparison measures one training seed on generated prices.

To reproduce it, prepare `var/profile21` and fit its models as described
above. Using a new output directory, run these phases in order:

```sh
uv run --locked --extra ml python scripts/compare_tcn_slopes.py reserve --corpus var/profile21 --output var/tcn-slopes
uv run --locked --extra ml python scripts/compare_tcn_slopes.py fit --corpus var/profile21 --output var/tcn-slopes
uv run --locked --extra ml python scripts/compare_tcn_slopes.py score --corpus var/profile21 --output var/tcn-slopes
```

The reservation checks the cohort against the full profile-2.1 corpus. Supply
`--exposure-index PATH` to the reserve command to check a prior exposure
inventory as well. For the reported results, we checked nine distinct local
exposure indexes and seventeen dataset manifests. The cohort shared none of
the 7,073 seeds or 31,836 identity keys in that inventory. Its dataset SHA-256
is `5bdc936d5116f7c72c7b641a7d888c95223b94796c4f89a4f397df847079ce6d`.
`freeze.json` records both packages before scoring.
`holdout/results.json` contains the per-experiment counts and paired intervals.
Reserve a fresh cohort for subsequent model choices.
