# Detecting pauses within a trend

The report can mark a short stretch of nearly level execution prices inside a
growth or decline interval. The price can resume its original direction after
several such pauses. A sideways interval is a separate classification made by
the model. The pause overlay reads the model's existing intervals and leaves
their labels and boundaries intact.

For reports generated with the current renderer, see the
[flat versus slow price comparison](pause-precision.md).

## Initial overlay results

The setting used in renderer 3.6 found **480 of 512 pauses (93.8%)**. It also marked **127
intervals in 131,072 steps of continuous slowdowns**, or 0.969 per 1,000 steps.
That exceeded the 0.5 development budget. An apparent plateau can therefore
be a continuing slow trend, especially at coarse price resolution. The overlay
is an opt-in aid for inspecting the chart.

| Detector | One pause matched (of 128) | Three pauses matched (of 384) | Second and third matched (of 256) | False intervals per 1,000 slowdown steps |
| --- | ---: | ---: | ---: | ---: |
| Historical single, published gain 4 | 126 | 126 | 90 | 1.038 (136 marks) |
| Execution-aware single, gain 16 | 97 | 109 | 75 | 0.526 (69 marks) |
| Multiple, relative slope only, gain 24 | 75 | 211 | 142 | 0.504 (66 marks) |
| Multiple, absolute drift limit, gain 4, noise multiplier 0.5 (renderer 3.6) | 124 | 356 | 239 | 0.969 (127 marks) |
| Same absolute rule, known tick supplied | 127 | 365 | 247 | 1.198 (157 marks) |

The absolute rule recovered all three pauses on 103/128 three-pause paths.
It produced three unmatched detections on paths with pauses and zero on the
steady controls (65,536 steps). Among the slowdown controls, 121/127 false
detections occurred in the near-flat group and six in the moderate group,
each with 65,536 steps. Including steady controls brings its false detection
rate across all negative paths to 0.646 per 1,000 steps.

The relative and absolute rules met a common budget on development data.
Their actual test rates differed. On the same final paths, the absolute rule
gained 37.9 percentage points of recall (paired 95% interval +31.4 to +44.3)
at the cost of 0.465 additional false marks per 1,000 slowdown steps (+0.130
to +0.793). The 2,000 bootstrap draws resample whole seed groups, keeping
their paths, schedules and methods together. They estimate uncertainty for
this constructed population. The market determines when measurements occur.
Its order book does not generate the controlled price paths.

| Execution schedule | Steps with trades | Pauses matched | False marks on slowdowns |
| --- | ---: | ---: | ---: |
| Dense | 32,516 / 32,768 (99.2%) | 244 / 256 | 63 / 65,536 steps |
| Sparse | 11,966 / 32,768 (36.5%) | 236 / 256 | 64 / 65,536 steps |

## How the initial overlay works

Within a growth or decline interval, a window has a candidate plateau in the
middle and a trend on each side. With the default 32-step scale, the plateau
can span 32, 40, 48 or 56 steps. Both sides must follow the original
direction. All three regions need executions in both halves. Fitting uses
execution prices and their original timestamps. A sufficiently supported
plateau must flatten at least fourfold relative to the flanking trends.

The report also caps the fitted price movement across the plateau using
the residual noise of the fit. When a tick is supplied, the limit allows at
least one tick of movement. Among eligible windows the detector maximizes
combined fit gain while keeping at least one trend flank between pauses.

## What the comparison measures

Each group supplies a 512-step price path in one direction. Five versions share
the same trend slope, noise, candidate locations and tick size:

- A steady trend throughout.
- A trend with one or three slowdowns. The underlying slope keeps its sign and
  falls to 8–22% or 35–65% of the original slope.
- A trend with one or three pauses. Its underlying slope is zero for 32, 40, 48
  or 56 steps. Independent of this construction, noisy execution prices can
  still move during the pause.

Each version is observed twice using dense and sparse trading schedules from
the native [exchange](../src/regimex/market.py). Each market has 13 participants
who independently choose to buy or sell with probabilities of 35% per side
(dense) or 4% per side (sparse). The exchange matches these orders. The
comparison preserves the times and volumes of actual executions. At a step
with executions it reports the noisy path price rounded to the configured tick
(0.01, 0.10 or 0.25). At other steps it carries the last execution price, with
100 as the opening price. The path determines the truth for all observations
of that group. A step without trades can occur inside any of the five shapes.

The comparison supplies the correct broad trend to every detector and applies
the usual availability rules before detection. All inserted pauses remain in
the denominator, including locations with little evidence. A match requires
at least 0.5 intersection over union with one known pause. Each detection and
each reference can match at most once. The reported recall is conditional on
the model having found the surrounding trend.

## Calibration

The development set contains 32 groups, each with two market schedules and
five price paths. The final set contains 64 entirely separate groups. All 192
native schedule seeds were checked against 8,785 seeds recorded in earlier
experiments. The report uses the information available in the export contract:
decimal precision describes how to display a price. The minimum tradable tick
is a separate input, so only a secondary comparison supplies the configured
tick to the detector.

The development budget was **at most 0.5 false detections per 1,000 steps on
continuous slowdowns**, pooling dense and sparse schedules. Each tunable detector
maximized pause matches within this budget. Ties favored fewer slowdown
detections and then fewer other unmatched detections. The historical
single-pause rule appears at its published fixed threshold. The other single
and multiple comparisons use the same execution-aware window scan.

The initial grid of gain multipliers (2, 4, 6, 8) left even the strictest
relative rule above budget. The range was extended on development data so the
relative rule could meet the same budget. During implementation, separate
unit cases identified windows straddling long execution gaps or lacking a
local return to trend. The detector and development comparison were updated
before the final set was opened. With the final rule on development data the selected
single, relative-multiple and absolute-multiple settings produced 29, 30
and 32 false detections respectively across 65,536 slowdown steps. They
matched 91, 112 and 221 of 256 pauses. The absolute setting was selected
for the report before scoring the final paths.

## Reproduce

Run the phases in order using a new output directory. The reservation binds
the paths, seed groups, exchange simulator, scientific dependencies and scoring
logic. Calibration records the detector digest, chosen settings and development
counts. Scoring checks these receipts before opening the final groups and
writes per-path counts, aggregate metrics and paired bootstrap intervals.

```sh
output=var/verification/trend-pause-comparison
uv run --locked --extra ml python scripts/compare_trend_pauses.py reserve "$output"
uv run --locked --extra ml python scripts/compare_trend_pauses.py calibrate "$output"
uv run --locked --extra ml python scripts/compare_trend_pauses.py score "$output"
```

To check historical seed overlap, add `--exclude-seeds PATH` to the reservation
for each earlier seed inventory or dataset manifest. The path generation and
scores are independent of whether these historical files are available.
