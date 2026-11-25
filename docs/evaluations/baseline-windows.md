# Baseline window summaries on incomplete exports

We compared four [baseline summary modes](../baseline-context.md#how-a-price-gap-affects-the-default-model)
on complete exports and on the same observations with a gap in price history.

## Validation

The four configurations used the same 96 training and 24 validation
experiments from the [profile-2.1 comparison](reference-releases.md#baseline-and-tcn-on-direction-3221). Each model used
centered 32- and 128-step windows, class balancing and regularization
`C = 1`, the value previously selected for the original baseline. Training
used the original observations. The original baseline came from its frozen
bundle. The other three were fit separately, saved and restored.

For each validation export we removed the prices and their states at steps
192–255. Execution counts and volumes remained available. For 256-step
exports the gap reaches the end of the run. The clean and masked predictions
were both scored against the original automatic reference intervals. Interval
F1 includes those references even when the masked model cannot give a
direction there. The mean of clean and masked interval macro-F1 selected the
configuration before opening the fresh evaluation.

| Summary mode | Clean interval F1 | Gap interval F1 | Gap point F1 on 17,000 common steps | Changed directions on 13,458 distant steps |
| --- | ---: | ---: | ---: | ---: |
| `local_and_global` | 0.3783 | 0.0952 | 0.1108 | 10,319 |
| `local` | 0.4232 | 0.2836 | 0.6161 | 50 |
| `local_and_global_observed` | 0.4381 | 0.0952 | 0.1108 | 10,573 |
| `local_observed` | 0.4699 | 0.3125 | 0.6056 | 42 |

The observed-only calculation with whole-sequence statistics still gave
0.0952 interval F1 on the masked exports. Dropping the whole-sequence
features raised it to 0.2836 with filled values and to 0.3125 with observed
values. The latter setting also had the highest clean interval F1 and was
selected for the fresh evaluation.

## Fresh paired evaluation

All 48 new experiments had 512 steps, alternating between one and two
instruments. The mask fell inside each run. Point F1 and direction changes
use only reference-classified steps where both exports gave a direction.
Distant steps exclude the gap and a margin of 65 steps on each side, covering
the largest window radius and the one-step return. A decoded interval can
cross that margin even if distant feature windows are unchanged.

Of the 36,864 instrument-steps, 31,765 had a direction on the complete
export and 27,797 had one in both versions. Both models used the same
availability rules.

| Summary mode | Clean interval F1 | Gap interval F1 | Clean point F1 on common steps | Gap point F1 on common steps | Changes on 27,797 common steps | Changes on 19,737 distant steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local_and_global` | 0.3798 | 0.0914 | 0.6195 | 0.1524 | 21,558 | 15,190 |
| `local_observed` | 0.3642 | 0.2353 | 0.6851 | 0.6280 | 3,061 | 78 |

The observed-only local model increased gap interval F1 by 0.1438
(paired 95% bootstrap interval 0.0727 to 0.2205). Clean interval F1 fell
by 0.0156 (interval −0.1236 to 0.0962). The bootstrap sampled whole
experiments with replacement, keeping instruments and mask pairs together,
with 1,000 draws and seed 1729. These intervals describe variation across
experiments for the fitted models.

The measurements cover this specific interior price gap. For example, a gap
at the start can change which observed price anchors all relative prices in a
sequence. Performance on other gap positions and on external markets remains
unmeasured for these candidates.

## Reproduce

Prepare the comparison corpus and its frozen baseline using the worktree,
`prepare`, and `fit` commands in [the profile-2.1 protocol](reference-releases.md#reproduce-the-comparison).
From the repository root, check out the window comparison revision that matches
the recorded source hashes. The `fit` command writes a plan in a new output
directory, compares validation predictions and freezes the selected model.
The `score` command checks the recorded code, environment, bundle identities
and data before generating the final references.

```sh
git worktree add --detach var/baseline-context-source eb23ccb79b7d786093c3fd008a5364f2ca5908c3
uv run --directory var/baseline-context-source --locked --extra ml \
  python scripts/compare_baseline_context.py fit --corpus ../profile21 --output ../baseline-context
uv run --directory var/baseline-context-source --locked --extra ml \
  python scripts/compare_baseline_context.py score --corpus ../profile21 --output ../baseline-context
```

The selected model bundle is under `var/baseline-context/local_observed/`.
Its manifest digest is recorded in `freeze.json`. The per-mode validation
files give the four comparisons, and `final/results.json` gives the paired
held-out scores and bootstrap intervals.

The final cohort used seed 1947046183 and dataset digest
`010468ec0cdbe200c41fc7d77ee370039582de9447065588c0694dc945f9950f`.
Pass the previous exposure index with `--exposure-index PATH` to repeat the
overlap audit. The 48 new seeds and 384 identity keys from the 48 pristine
and 48 masked exports had no overlap with the prior index containing 7,025
seeds and 29,234 keys. The successor also records the 60 previously unseen
identities of the masked validation exports.
The recorded corpus can be reproduced by digest without the historical index.
