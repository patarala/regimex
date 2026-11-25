# Segment MoE on Market 320

## Held-out result for the workspace model

The training worker selected seed 491 on the Market 320 validation set.
The bundle with manifest SHA-256
`054ded3554eb8d5d58e35f91eeae16222eab2fd4f3f0b4119e794ecf2ea2ec92`
was frozen before the service opened the reserved test references.
Both splits use automatic references and the same decoder and analysis profile.

| Split | Experiments | Instrument series | Interval macro-F1 | Per-step macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 63 | 96 | 0.9728 | 0.9921 |
| Held out | 126 | 188 | 0.9478 | 0.9844 |

On held-out intervals, F1 was 0.9314 for sideways, 0.9416 for growth and
0.9702 for decline. Resampling whole experiments gave a 95% percentile
interval of 0.9188–0.9727 for interval macro-F1. The reference labels come
from rules applied to the simulated observations, so these scores measure
agreement with those rules. [Metric definitions](../quality.md#read-the-metrics)

## Changing the number of experts

The default model combines an accumulation expert and a network that reads
nine neighboring segments. A three-expert variant added a network that saw
only the current segment. A four-expert variant also added a network that
saw the current segment and its two immediate neighbors. All experts had a
separate mixture weight computed from the same block statistics.

All variants trained on the same 256 experiments. The checkpoint for each
seed was selected by validation interval macro-F1 on the same 63 experiments.
The batch policy, optimizer, decoder and epoch budget were shared. The
context widths were adjusted to keep parameter counts close. Rerunning seeds
173 and 491 reproduced the original service model's validation scores at
every epoch.

| Experts | Trainable parameters | Seed 173 | Seed 491 | Seed 827 | Mean validation interval macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2,137 | 0.97046 | 0.97278 | 0.95979 | 0.96768 |
| 3 | 2,133 | 0.97278 | 0.96635 | 0.95979 | 0.96631 |
| 4 | 2,137 | 0.97278 | 0.96635 | 0.95125 | 0.96346 |

The three-expert mean difference from two experts was −0.00137 (paired 95%
bootstrap interval −0.00666 to +0.00223). For four experts it was −0.00422
(−0.01161 to +0.00091). The resampling unit was the experiment, with all of
its instruments kept together across seeds and variants. Both intervals
include zero. The extra experts showed no consistent advantage on this
development set.

A second comparison kept the nine-segment expert at its original width of
32. With seeds 173 and 491, the models had 2,137, 2,293 and 2,537 parameters
and mean validation scores of 0.97162, 0.96957 and 0.96957 respectively.
Letting the extra experts increase total capacity did not improve the mean
in those two repeats. Only the previously frozen two-expert model was
evaluated on the reserved test set.
