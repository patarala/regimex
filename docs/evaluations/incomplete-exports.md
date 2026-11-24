# Direction estimates on incomplete exports

The frozen baseline and TCN saw the same 48 new synthetic experiments with three
kinds of missing data. Removing 64 consecutive prices changed the TCN's direction
on 0.81% of the comparable steps outside the gap. The baseline changed direction
on 67.49% of those steps. Removing the entire quote and aggregate history left
every TCN direction unchanged when the price and its observed/carried state
were still available.

## Which directions changed?

The 48 experiments each had 512 steps. Half had one instrument and half had two,
giving 36,864 instrument-steps. Each comparison used the original export and
one of the following versions of that same export.

- **Price gap.** Steps 192 through 255 have unknown prices and price states for
  every instrument. The export retains the actual execution counts and volumes.
- **Execution gap.** The same 64 steps also have unknown execution counts and
  volumes. Quotes and accepted-order totals remain available.
- **Price only.** The full export retains prices and their observed/carried
  states. Both historical quote sides, execution totals and accepted-order
  totals are unknown at every step.

The table counts direction changes on steps with an automatic reference
direction and a prediction in both versions. Each F1 pair uses exactly those
same steps, with equal weight for sideways, growth and decline. References
come from the pristine export in every comparison. The two window masks gave identical
paired counts and F1 on the common steps outside the gap.

| Export | Model | Common steps | Changed direction | Pristine point F1 | Masked point F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Price gap or execution gap | Baseline | 25,722 | 17,360 (67.49%) | 0.7696 | 0.1767 |
| Price gap or execution gap | TCN | 25,722 | 208 (0.81%) | 0.8931 | 0.8965 |
| Price only | Baseline | 29,399 | 17,492 (59.50%) | 0.7755 | 0.2837 |
| Price only | TCN | 29,399 | 0 (0.00%) | 0.8962 | 0.8962 |

The baseline uses quote and activity features and summarizes each instrument's
entire sequence as well as local windows. A local gap can therefore change an
input to the baseline at every step. The TCN uses price-derived features.
These measurements cover the specified gaps on synthetic markets and measure
agreement with the automatic labeling rule. Accuracy on external markets and
agreement with human judgments remain unmeasured. A separate
[comparison of baseline windows](../baseline-context.md) tests which features
make a price gap change directions elsewhere in the run.

## When does the service stop classifying?

Both models use the same availability rules. The counts below cover the entire
36,864-step grid. Lost and gained count steps that changed direction coverage
relative to the pristine export, independent of their predicted direction.

| Export | Has direction | No trades | Insufficient data | Lost | Gained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pristine | 29,399 | 7,463 | 2 | — | — |
| Price gap | 25,722 | 7,463 | 3,679 | 3,677 | 0 |
| Execution gap | 25,742 | 6,513 | 4,609 | 3,677 | 20 |
| Price only | 29,399 | 7,463 | 2 | 0 | 0 |

The 64-step window contains 4,608 instrument-steps. Originally 1,273 of them
recorded both zero executions and zero volume. Of those, 930 belonged to a
sustained pause and 343 were shorter interruptions. With totals retained, all
930 pause steps kept the `no_trades` status. The remaining 343 zero-execution
steps had insufficient price data to classify. When execution totals also went
missing, all 4,608 window steps became `insufficient_data`, including the same
930 pause steps.

The execution gap broke the evidence for a sustained pause across its edges.
An additional 20 steps outside the gap then received a direction. Keeping the
price state in the price-only export preserved evidence of inactivity through
the carried prices. [Availability rules](../models.md)

## Reproduce the measurements

Start with the saved baseline and TCN bundles from the
[direction-32/2.1 comparison](../evaluations/reference-releases.md#reproduce-the-comparison).
Run the following command from the checkout with a new output directory.

```sh
uv run --locked --extra ml python scripts/evaluate_missing_data.py \
  --bundles var/profile21 --output var/incomplete-exports
```

The script records the protocol before scoring, verifies the frozen bundles
and generated dataset digest, and writes the paired counts to `results.json`.
The plan seed was 318751406. The measured cohort's dataset digest was
`a8f58746e20f3a55809233e0c18671aaf700e257b489b470e3966d68c3657d79`.
Both models kept their previously selected settings and weights.

| Measured bundle manifest | SHA-256 |
| --- | --- |
| Baseline | `1280d07e99c19f1e630dabf1475ae72956330e71702855115b888a40f70df443` |
| TCN | `eb7e282aac43ea7759146d851bb88c2c3ea843e61f681d7f0d1b11b5f154b27f` |

The audit checked all 48 seeds and 624 distinct identity keys against the local
exposure index of 6,977 seeds and 28,610 keys, including the earlier model
comparison. The 48 pristine snapshots contributed 264 keys. The 144 masked
snapshots added 144 snapshot identities and 216 instrument observation
fingerprints. None overlapped with the prior index. Supply that index with
`--exposure-index PATH` to repeat the historical audit and write an updated
index for future evaluations. A run without it verifies the recorded cohort
by digest. The prior index SHA-256 was
`f839d6b47c6e1d21bfb01a387d4a7db4da550aed0919d207de3523176108c4b7`.
