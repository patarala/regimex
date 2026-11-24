# Logistic baseline

The baseline scores one instrument at a time. It starts with price, return,
trade and order activity, order imbalance and bid–ask spread. Each numeric
feature has a missing-value indicator. At each step it collects the mean,
population standard deviation, minimum and maximum of these channels in
centered windows of 32 and 128 steps by default.

The model standardizes the summaries with statistics fitted on labeled
training steps, then applies a linear score for each direction. A softmax
turns the three scores into probabilities. Training uses class-balanced
logistic regression with the LBFGS solver. Both this model and the TCN use
the same [decoder and interval rules](models.md#observations-and-intervals).

## How a price gap affects the default model

A gap in one instrument can change its whole-sequence means, spreads and extremes.
The default summary mode supplies those same features at every step.
The `summaries` parameter controls whether to include these features and
how to handle missing values inside each window.

| Training parameter `summaries` | Centered windows | Whole-instrument window | Numeric inputs to the summaries |
| --- | --- | --- | --- |
| `local_and_global` (default) | Yes | Yes | Include training-median fills |
| `local` | Yes | No | Include training-median fills |
| `local_and_global_observed` | Yes | Yes | Observed values |
| `local_observed` | Yes | No | Observed values |

Every mode summarizes missingness indicators over all the steps in its
windows. For observed-only summaries, a completely missing numeric channel
contributes zeros in standardized feature units. Its missingness indicator
records the full absence. Local windows end at the instrument boundaries.

The [baseline window comparison](evaluations/baseline-windows.md) measures how
these four choices affect predictions inside and outside a price gap.

## Train a candidate for incomplete exports

Use the following `parameters` with a baseline
[training request](api.md#administration) to train with observed values in local
windows. Training produces a separate candidate for evaluation and
explicit publication.

```json
{"regularization": [1.0], "context_sizes": [32, 128], "summaries": "local_observed"}
```

Requests that omit `summaries` use `local_and_global`. The
`aggregate-logistic/1.0` bundle format uses this fixed configuration.
The `aggregate-logistic/2.0` format records the requested
summary mode with the numeric weights and fitted preprocessing.
