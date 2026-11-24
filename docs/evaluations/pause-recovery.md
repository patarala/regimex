# Finding pauses on noisy price paths

A short stretch of trades at roughly the same price can sit inside a longer
rise or fall. Isolated trades away from that level make some of these pauses
hard to see. On 96 development groups, renderer 3.7 found 377 of 384 pauses
on the quieter paths and 275 of 384 on the noisier ones. This comparison asks
whether the report can recover some of the missed pauses while keeping false
highlights on slow trends rare.

## What was compared

Both candidates start with the report's existing scan for a pause between
two moving flanks. They consider more proposed windows. A mark keeps the
original proposal's start and stop steps if the price evidence supports it.

- **Robust fit.** Look farther along the trend on either side. Discount
  isolated price deviations and account for the shorter time between closely
  spaced executions. Compare a flat center with continued or reversed movement.
- **Average over boundaries.** Allow some uncertainty in where the pause began
  and ended. Compare the flat and moving explanations using evidence across
  nearby boundary pairs. This reduces the influence of a single particularly
  favorable pair.

Settings were selected on 128 development groups. The primary comparison used
320 fresh groups with known surrounding directions and the same
[matching rule and price generator](trend-pause-detection.md#what-the-comparison-measures).
The paired uncertainty estimates resample whole groups, including their dense
and sparse execution schedules.

A second check used 96 further groups and the released `tcn-0.1.0` model to
predict the surrounding directions from serialized observations. Prices followed
curved trends, with occasional outliers and alternating execution prices around
the underlying path. Each group paired plateaus of varying length with slowdowns,
brief pullbacks and uninterrupted trends. All four shared the same noise and
execution schedule. Both checks use constructed prices.

## Results

Development selected the robust fit with up to three minimum durations of
context on each side. Of 1,024 known pauses it found 911. The best setting
that averaged over boundaries found 895, and the current report found 877.
Only the selected robust fit was scored against the current report on the
two reserved comparisons.

| On 320 fresh groups | Matched / 2,560 | Precision | False marks / 655,360 slowdown steps |
| --- | ---: | ---: | ---: |
| Current report (3.7) | 2,215 (86.5%) | 88.7% | 275 (0.420 per 1,000) |
| Robust fit | 2,287 (89.3%) | 89.3% | 270 (0.412 per 1,000) |

The paired 95% interval for the gain in recall was 1.7 to 4.0 percentage
points. The noisier half accounted for 61 of the 72 additional matches.
Median absolute errors at the start and end of matched pauses were two steps
for each method. The robust fit passed the primary criteria for recall gain,
slowdown false marks and precision.

| With directions predicted by the release TCN | Matched / 384 | Precision | False marks on slowdowns | On pullbacks | On steady trends |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current report (3.7) | 356 (92.7%) | 76.2% | 68 | 42 | 0 |
| Robust fit | 341 (88.8%) | 62.6% | 75 | 128 | 0 |

Each control type contributed 98,304 steps. Before scoring, the transfer check
allowed at most two percentage points of lost recall and at most 0.10 extra
false marks per 1,000 control steps. Recall fell by 3.9 points and false
marks rose from 0.373 to 0.688 per 1,000. Most of the additional marks came
from brief moves against the trend. Renderer 3.7 therefore keeps the
[existing shape check](pause-precision.md#report-behavior). Both overlays use
predictions from the same frozen model and decoder.

## Reproduce

Use unused output directories. The main comparison freezes the selected
setting and both implementations before either confirmation is scored. The
transfer check also verifies the released TCN bundle against its pinned digest.

```sh
comparison=var/verification/pause-recovery-run
transfer=var/verification/pause-recovery-transfer-run
uv run --locked --extra ml python scripts/compare_pause_recovery.py reserve "$comparison"
uv run --locked --extra ml python scripts/check_pause_recovery_transfer.py reserve "$transfer"
uv run --locked --extra ml python scripts/compare_pause_recovery.py develop "$comparison"
uv run --locked --extra ml python scripts/compare_pause_recovery.py confirm "$comparison"
uv run --locked --extra ml python scripts/check_pause_recovery_transfer.py score "$transfer" "$comparison" --bundle release/models/tcn
```

To check additional local seed inventories at reservation, pass
`--exclude-seeds PATH` to the first command. Each inventory contains an
`excluded_seeds` array of integers.
