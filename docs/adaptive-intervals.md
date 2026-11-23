# Short intervals and missing observations

The released adaptive forest found 12 of 30 sideways intervals shorter than
64 steps on a previously used validation set. At an available block's edge
it found 10 of 18. Inside a block it found 2 of 12. Of the 18 misses, 11
had no predicted sideways overlap and seven overlapped a longer sideways
prediction. This motivated two changes to how the model reads the price path.

## Local evidence and jointly fitted segments

We trained two 40-tree forests on the same 192 experiments as
`adaptive-0.1.0`. The local model replaces nine of the 30 inputs with
windowed movement measurements at 32, 64 and 128 steps. The other model
retains the released inputs but fits the segment boundaries jointly at
each penalty level. Its interior search grid stays anchored to the original
steps when observations go missing.

Among 60 fresh validation experiments the local model scored 0.8546 interval
macro-F1, the released model 0.8087 and the joint-fit model 0.7883. Only the
local model and the release advanced to 120 reserved experiments.

| Model on the same 120 experiments | Interval macro-F1 | Point macro-F1 | Short sideways matches | False sideways intervals |
| --- | ---: | ---: | ---: | ---: |
| Released adaptive forest | 0.8070 | 0.9339 | 12 / 31 | 18 |
| Forest with local evidence | 0.8069 | 0.9347 | 13 / 31 | 22 |

The paired 95% interval for the change in interval macro-F1 was −0.0259 to
+0.0261. The validation advantage did not repeat. The local model's sideways
precision fell from 80.0% to 76.8%. Neither changed feature set qualified
to replace the release.

We separately generated 384 paths with direction set by their drift before
noise, tick rounding and execution sampling. Each had a complete view and a
second export missing 32, 48 or 64 steps. Views of the same path share all
other observations. The 32 independent seed groups include pauses of 32–63
steps, slow continuing trends and reversals near missing observations.

| Result on paired paths | Automatic labels | Released adaptive forest | Forest with local evidence |
| --- | ---: | ---: | ---: |
| Interval macro-F1, both views | 0.8575 | 0.9150 | 0.9275 |
| Known short pauses found, complete export | 100 / 128 | 90 / 128 | 101 / 128 |
| Known short pauses found, masked export | 104 / 128 | 105 / 128 | 114 / 128 |
| Directions changed on steps available in both views | 2.67% | 3.33% | 3.13% |
| Directions changed at least 129 steps from the gap | 0.15% | 0.37% | 0.39% |

On native exports the reference is the automatic annotation. On these
controlled paths we could also compare that annotation with the known drift
after model selection. In the slow-trend scenario, for example, the automatic
annotator marked 64 sideways intervals across the complete and masked views.
The released model marked 36 and the local model marked 31. Every one of those
sideways intervals was a false positive under the designed drift. The automatic
profile requires a fitted movement of at least 0.5% and a directional fit of
at least 0.5. The controlled scenario keeps the trend label while its drift
slows to 40% of its original speed.

### Reproduce the feature comparison

Use the locked ML dependencies, the previous [direction comparison](direction-context.md#reproduce)
and its verified [model bundle](direction-context.md#use-the-trained-forest).
Run the phases below in one checkout with unused output directories. To replay
a saved fit, use its original revision (`f283900` for the reported feature
comparison). The scorer checks the source fingerprint recorded during training.
Set `release` to the verified bundle directory
(manifest SHA-256 `8bf057911823ed1d149320de2d765f91937ddc4b7205c4356951d208a52b48a0`).
Reserve both populations before fitting. If the release wins validation,
leave the reserved data closed.

```sh
result=var/verification/adaptive-interval-quality
transfer=var/verification/adaptive-interval-transfer
release=var/release/adaptive-direction-0.1.0/bundle
uv run --locked --extra ml python scripts/compare_adaptive_intervals.py reserve "$result"
uv run --locked --extra ml python scripts/check_adaptive_gap_transfer.py reserve "$transfer" "$result"
uv run --locked --extra ml python scripts/compare_adaptive_intervals.py train "$result"
uv run --locked --extra ml python scripts/compare_adaptive_intervals.py freeze "$result" --control-bundle "$release"
uv run --locked --extra ml python scripts/compare_adaptive_intervals.py score "$result" --control-bundle "$release"
uv run --locked --extra ml python scripts/check_adaptive_gap_transfer.py score "$transfer" "$result" --control "$release"
```

## Training on more experiments

Both new models used the released feature set and the same 40-tree forest.
We added 192 fresh experiments to the original 192 training experiments.
The first recipe used every labeled row with its ordinary weight. The second
sampled more rows from short intervals within each class and added a
price-reflected copy of each sample, exchanging growth and decline. These two
changes to the second recipe were evaluated together.

On 96 validation experiments the ordinary 384-experiment forest scored
0.8330 interval macro-F1, the release scored 0.8232 and the reweighted,
reflected forest scored 0.8203. We froze the ordinary forest and compared
it to the release on 240 further experiments, with 80 at each series length.
The two feature and training comparisons used separate final sets.

| Result on the same 240 experiments | Released forest (192 training experiments) | Selected forest (384 training experiments) |
| --- | ---: | ---: |
| Interval macro-F1 | 0.73601 | 0.73596 |
| Interval macro-F1 at 256 / 512 / 1,024 steps | 0.8139 / 0.7796 / 0.6575 | 0.7666 / 0.8022 / 0.6689 |
| Point macro-F1 | 0.9157 | 0.9187 |
| Short sideways intervals matched | 46 / 108 | 43 / 108 |
| False sideways intervals | 69 | 58 |
| Brier score on 184,071 labeled steps (lower is better) | 0.1369 | 0.1262 |

The paired 95% interval for the change in interval macro-F1 was −0.0216 to
+0.0213. The 0.0474 loss at 256 steps exceeded the 0.03 regression margin.
Of the 50 reference declines at that length, the release matched 38 and
the selected forest matched 34. Replaying the four lost matches showed
different errors. On DOT/USD and SOL/USD the forest called the entire
256-step record sideways. On XRP/USD it merged two declines across a
32-step pause, leaving only 0.125 IoU with the last decline. On ADA/USD it
started a decline 86 steps late, bringing IoU down from 0.854 to 0.461.

The new weights also passed the separate tolerance checks on 384 fresh
controlled paths. There interval macro-F1 was 0.9111 against 0.9168 for the
release, and both forests found 97 of 128 known short pauses on complete
exports. We retained the weights from `adaptive-0.1.0`.

### Reproduce the training comparison

Use the completed feature comparison above. Its `result`, `transfer` and
`release` paths are reused below. The training comparison used revision
`eec81e2`. The new reservation excludes the previous training and evaluation
groups. The two final scoring commands require a new validation winner.
To compare the automatic labels with the known path directions after closing
the feature benchmark, run the label check once in the same ML environment.

```sh
uv run --locked --extra ml python scripts/check_adaptive_gap_labels.py "$transfer"
```

Reserve the new training and transfer populations before fitting.

```sh
mix=var/verification/forest-training-mix
paired=var/verification/forest-training-transfer
uv run --locked --extra ml python scripts/compare_forest_training_mix.py reserve "$mix" --previous "$result" --previous-transfer "$transfer"
uv run --locked --extra ml python scripts/check_forest_training_transfer.py reserve "$paired" "$mix" "$transfer"
uv run --locked --extra ml python scripts/compare_forest_training_mix.py train "$mix"
uv run --locked --extra ml python scripts/compare_forest_training_mix.py freeze "$mix" --control-bundle "$release"
uv run --locked --extra ml python scripts/compare_forest_training_mix.py score "$mix" --control-bundle "$release"
uv run --locked --extra ml python scripts/check_forest_training_transfer.py score "$paired" "$mix" "$transfer" --control "$release"
```

## Inference time

A piece can cover many consecutive steps with identical inputs to all 40
trees. The predictor now evaluates each run of identical float32 feature rows
once and expands its probabilities onto the original grid. It can traverse
several trees together while accumulating their votes in their original order.
The released weights produced bit-for-bit identical probabilities on all
64,512 validation rows before and after this change.

| Steps, one instrument | Previous prediction and decoding | Updated prediction and decoding |
| --- | ---: | ---: |
| 256 | 16.8 ms | 11.4 ms |
| 512 | 33.9 ms | 23.8 ms |
| 1,024 | 65.3 ms | 50.2 ms |
| 4,096 | 312.5 ms | 270.0 ms |

These are medians over nine warm CPU runs, alternating the implementations on
the same observations. The first three inputs are validation examples with
at least 80% price coverage. The 4,096-step input is a continuously observed
synthetic path. Timings include feature extraction, tree traversal and interval
decoding. On a separate worst case where all 4,096 feature rows differ,
tree traversal alone took 36.0 ms before and 36.3 ms after.
