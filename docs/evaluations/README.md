# Evaluation reports

These reports describe particular weights, datasets and label profiles. For
scores from a model in your installation, start with [Model quality](../quality.md).

| Question | Reports |
| --- | --- |
| How did the published reference versions score on their reserved data? | [Reference release benchmarks](reference-releases.md) |
| How did the offline specialists and Segment MoE perform on exchange prices and known drift? | [Neural direction models](neural-direction.md) |
| How do missing observations or uncertain boundaries affect scores? | [Incomplete exports](incomplete-exports.md), [interval errors](interval-errors.md), [label stability](label-stability.md) |
| How did model width, input windows and training data affect direction scores? | [TCN widths](tcn-widths.md), [baseline windows](baseline-windows.md), [temporal architectures](temporal-comparison.md), [training populations](training-populations.md) |
| How did forest features and interval weighting affect direction scores? | [Adaptive intervals](adaptive-intervals.md), [large direction forest](../large-direction-forest.md) |
| Can a price pause be distinguished from a slow trend? | [Pause detection](trend-pause-detection.md), [precision](pause-precision.md), [recovery](pause-recovery.md), [supervision](pause-supervision.md) |

The [model guide](../models.md#learned-backends) describes the architectures
available for training.
