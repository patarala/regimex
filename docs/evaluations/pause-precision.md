# Distinguishing price pauses from slow trends

A few execution prices can line up horizontally even while a market continues
to move. A detector that searches many overlapping windows can find such a
stretch by chance. We compared the original report overlay with tighter
thresholds and two ways of checking whether the price continued to move
through the proposed pause.

## Comparison

- **Existing overlay.** The price flattens relative to both flanks, and fitted
  movement inside the proposed interval stays below a noise allowance.
- **Tighter thresholds.** The same detector can require a larger gain from
  fitting a plateau or allow less movement inside it.
- **Flat versus slow fit.** A flat center and a center moving at a fraction of
  the surrounding trend compete to explain the same executions. Both fits can
  adjust where the pause starts and stops.
- **Estimated center slope.** Three connected lines are fitted to each
  neighborhood. The middle slope is checked against the slopes on either side.

The paired comparison uses the [same price-path design and interval matching
rule](trend-pause-detection.md#what-the-comparison-measures) with 128 new
development groups and 128 separate confirmation groups. Each group contains
growing or declining prices, with dense and sparse execution schedules. Prices
at steps without trades are carried forward. They supply no new price
observations to the fit.

Selection maximizes the fraction of known pauses found, subject to at least
80% recall and a one-sided 95% upper bound of at most 0.5 false marks per
1,000 slowdown steps. The bound comes from 4,000 bootstrap samples of whole
seed groups. The selected rule and the threshold-only alternative were frozen
before opening the confirmation set. Adoption requires at least 80% recall,
at most 0.5 false marks per 1,000 slowdown steps, higher pooled precision and
a paired one-sided 95% upper bound below zero for the change in the false-mark
rate. Precision here depends on the mix of pauses and controls in this
comparison.

### First confirmation

On development data the shape check selected a continuing center slope of
one fifth of the average outer slope. It retained 913 of 1,024 pauses. Its
upper false-mark rate was 0.484 per 1,000 slowdown steps. On the fresh
confirmation set the same rule exceeded the absolute rate limit.

| Check on 128 confirmation groups | Pauses matched / 1,024 | Precision | False marks / 262,144 slowdown steps | False marks per 1,000 steps |
| --- | ---: | ---: | ---: | ---: |
| Existing overlay | 942 | 81.7% | 199 | 0.759 |
| Tighter thresholds | 837 | 84.7% | 142 | 0.542 |
| Flat versus one-fifth slope | 922 | 86.1% | 142 | 0.542 |

The shape check found 85 more pauses than tightening the thresholds alone
at the same observed false count. Of its 142 false marks, 141 occurred on
nearly flat slowdowns. It did not qualify for adoption.

### Independent replication

A second comparison fixed a single stricter alternative before opening
384 further groups. Its competing center slope is one sixth of the average
outer slope. Its results on the original development set were available.
It had not been scored on the first confirmation set. The comparison used
the same adoption criteria.

| Check on 384 confirmation groups | Pauses matched / 3,072 | Precision | False marks / 786,432 slowdown steps | False marks per 1,000 steps |
| --- | ---: | ---: | ---: | ---: |
| Renderer 3.6 | 2,824 (91.9%) | 82.3% | 597 | 0.759 |
| Flat versus one-sixth slope (renderer 3.7) | 2,729 (88.8%) | 88.2% | 360 | 0.458 |

The new rule met all four adoption criteria. It produced 39.7% fewer false
marks on slowdowns. The paired 95% interval for the reduction was 0.240 to
0.365 marks per 1,000 steps. The one-sided 95% upper bound for its **absolute
rate** was 0.530. The 0.5 criterion applies to the observed rate and is not a
guarantee for future data.

The paired gain in precision was 5.9 percentage points (95% interval 4.7 to
7.1). Recall fell by 3.1 points (1.7 to 4.6). On paths with three pauses, all
three were found on 560 of 768 paths, down from 612. Both rules produced zero
marks across 393,216 steady-trend steps. Nearly flat slowdowns accounted for
355 of the new rule's 360 false marks.

## With TCN predictions

A separate replay passed 640 completed synthetic paths from 64 new groups
through each saved TCN, its own preprocessing and its decoder. The overlay
used those predicted intervals. The models saw prices serialized at their
declared precision and the observed activity on the original grid. The known
pause locations and surrounding direction were used only to score matches.
This replay measures the full classification and highlighting path on the
same constructed price process.

| TCN replay | Pauses matched / 512 | Precision | False marks / 131,072 slowdown steps |
| --- | ---: | ---: | ---: |
| Renderer 3.6 | 476 (93.0%) | 82.4% | 96 (0.732 per 1,000) |
| Renderer 3.7 | 447 (87.3%) | 87.0% | 66 (0.504 per 1,000) |

The release model `tcn-0.1.0` and the frozen `mixed-192` model gave identical
counts on these same paths. Neither decoder produced a separate sideways
interval here. The new overlay again produced fewer false marks, with an
observed rate just above 0.5 per 1,000 slowdown steps. Real-world recall and
precision have not been measured.

## Report behavior

Renderer 3.7 uses the flat versus one-sixth slope check. The competing fits
can move their boundaries while explaining the same executed prices around
each candidate. Stronger evidence of a brief move in the opposite direction
can also remove a highlight. The saved direction probabilities and intervals
come from the original model. Existing downloaded HTML retains the overlay of
the renderer that produced it.

A [follow-up comparison on noisy prices](pause-recovery.md#results) found a
way to recover more pauses on straight trends. It also mistook more short
pullbacks for pauses when the surrounding trend curved, so the report keeps
the one-sixth slope check.

## Reproduce

Run from revision `2dd9a58` with its locked scientific dependencies. Each output
directory must be new. Calibration, confirmation and replay write receipts
binding the settings to the code and the data groups. The second confirmation
checks the completed first comparison before opening its own groups.

```sh
first=var/verification/pause-precision-run
second=var/verification/pause-precision-replication-run
replay=var/verification/pause-precision-tcn-run
uv run --locked --extra ml python scripts/confirm_pause_precision.py reserve "$first"
uv run --locked --extra ml python scripts/confirm_pause_precision.py develop "$first" --workers 4
uv run --locked --extra ml python scripts/confirm_pause_precision.py freeze "$first"
uv run --locked --extra ml python scripts/confirm_pause_precision.py confirm "$first" --workers 4
uv run --locked --extra ml python scripts/replicate_pause_precision.py reserve "$second" --previous "$first"
uv run --locked --extra ml python scripts/replicate_pause_precision.py confirm "$second" --previous "$first" --workers 4
uv run --locked --extra ml python scripts/check_pause_overlay_on_tcn.py reserve "$replay"
uv run --locked --extra ml python scripts/check_pause_overlay_on_tcn.py score "$replay" "$first" tcn-0.1.0 release/models/tcn
```

To also check `mixed-192`, pass `mixed-192` and the corresponding frozen bundle
directory to the final command. Both bundle digests are pinned in the replay
script. See the [training population comparison](training-populations.md) for
the model recipe. If a historical seed inventory is available, add
`--exclude-seeds PATH` to the first reservation.
