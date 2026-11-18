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

Prepare the comparison corpus and fit its models using the first two commands
in [model quality](quality.md#reproduce-the-comparison). Run the diagnosis from
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
