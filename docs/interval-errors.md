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

None of the six shortest intervals contained a predicted sideways step.
Three were covered by a neighboring trend. In the other three, the model called
the entire interval growth on precisely the reference bounds. All three lasted
less than 32 steps. These six observations come from a validation split also
used to choose the checkpoint. They describe this candidate on these runs.

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
Each chart identifies the interval in steps and shows price, the automatic
reference and the decoded prediction. The dotted lines mark the reference
bounds. Keep the generated directory together when opening it offline.
`summary.json` contains every reference interval, its overlapping predictions,
and the digests of the model, dataset and validation references. The script
loads only the validation partition and uses the profile stored in the candidate.
