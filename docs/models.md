# Data, models and interval semantics

Regimex describes sustained price direction in a completed experiment. Models may
use the entire sequence. The result is retrospective, not a streaming signal or
a forecast of real-market prices.

## Observations and intervals

An export contains one or two instruments on a complete shared time grid,
source/run/snapshot identity, units, capabilities and explicit missingness. Limits
are 4,096 steps and 16 MiB. Observed zero trade count differs from an unknown count;
an empty book side differs from unavailable quote history.

Each instrument has contiguous, non-overlapping `[start, stop)` intervals covering
its full grid. Direction is `sideways`, `growth` or `decline`. Adjacent equal
directions merge. The default minimum duration is 32 steps; an entire observed
block isolated by unavailable data may be shorter, but cannot be split into still
shorter fragments. Imported annotations are validated, never silently repaired.

Long observed trading pauses are `no_trades`; insufficient observations are
`insufficient_data`. Both have no direction. Short gaps may remain within a
classified interval, with reduced coverage. Carried prices alone do not establish
active sideways trading. Availability is derived from observations independently
of model confidence and is shared by predictions and references.

Interval statistics include first/last observed price steps, relative price change,
trade count and volume, active-trade fraction, longest observed pause, coverage and
population standard deviation of adjacent log returns. Only known counts/volumes
are summed; return deviation is absent with fewer than two known adjacent returns.
Variability and activity remain numeric descriptors. No jump events are emitted.

## Synthetic market

The generator simulates orders, resting books and executions using integer tick,
lot and settlement units. Matching uses best price, then arrival time, supports
partial fills and releases asset reservations correctly. Separate instruments
share participant quote balances. Random participants sample BUY/SELL/HOLD;
Trend Following participants compare execution prices across their trade-history
window and hold until enough trades exist. Each step processes participants in
fixed order, with one independently configured test participant last.

The sampler varies strategies, resources and activity without scheduling target
directions or balancing labels. It alternates one/two instruments and samples
inactive, one-sided, fixed-price and heterogeneous configuration families with
probabilities 5%, 5%, 10% and 80%. Native names are BTC, ETH, SOL, XRP, ADA, LTC,
DOT or DOGE quoted in USD, selected deterministically from the seed. These names
do not imply realistic currency prices or calibrated behavior.

Prices come from executions; without a trade, the last or initial price carries
forward. Counts and volumes are always observed, including zeros. Quotes are
sampled after each step; an empty side is null. Generator assumptions include
execution at the resting price, uniform instrument choice per participant event,
incoming-remainder cancellation on self-match, and no fees, expiry, leverage or
external news. Tick and lot sizes default to 0.01. A real simulator may use
different mechanics; shared fields alone do not establish model transfer.

Generate through `POST /api/v1/datasets`; an optional `plan.market` supplies an
explicit validated market configuration. Generation settings and seed provenance
are kept outside observation-only prediction inputs. In participant profiles,
BUY/SELL/HOLD probabilities apply to Random, while history window/threshold apply
to Trend Following. The other strategy's stored action fields are inactive.

## Automatic reference labels

`retrospective-directions/3.0` is a reproducible annotation convention, not human
ground truth. After determining availability, it partitions log prices into
penalized line fits. Within each piece it compares an ordinary line with a common
time slope and separate intercepts around one internal level offset. A split
needs two observed prices on each side; missing values retain their timestamps.
Minimum residual error selects the fit, with ties preferring the ordinary line
then the earliest split. The internal split is not an output boundary or event.

The shared slope must explain at least the configured directional variance
(default 0.5). Consecutive supported pieces of one sign form a run. Their fitted
movements `slope * (stop - start - 1)` are summed and compared with
`log1p(minimum_relative_change)`, default 0.5%. Unsupported pieces break the run;
level differences between pieces do not contribute movement. Equal output
directions merge. A level offset alone can therefore be sideways even when its
end-to-end price-change statistic is large.

Only one unresolved offset per piece is modeled. Multiple offsets, unequal slopes,
noise or sparse short series remain ambiguous. A fit score is not a probability
that the interpretation is correct. Optional independent review uses
`regimex report --input export.json --output var/blind --blind`; its annotations
must remain separate from automatic labels and predictions.

## Learned backends

| Backend | Inputs and method |
| --- | --- |
| Baseline | Logistic regression on mean, population standard deviation, minimum and maximum of observation channels/masks in centered 32/128-step contexts and the whole sequence. It receives aggregates, not a flattened sequence. |
| TCN | Noncausal residual temporal convolutions over 50 price-shape/mask channels, with 32/64/128/256/512-bin contexts. It returns three probabilities per original step. |

Both fit preprocessing only on training data and restore it with weights. They
use the same availability rules and duration-constrained decoder. Prediction
receives raw observations, timing, units and missingness, never targets, reference
boundaries, seeds or generator settings. Backends own feature preparation and
their weight/preprocessing codecs; they do not share one compulsory feature matrix.

Baseline uses balanced logistic regression, LBFGS and up to 2,000 iterations.
TCN uses class-balanced cross-entropy, optional interval weighting, AdamW and
gradient clipping. Its adjacent-log-probability smoothing applies only to equal
known training targets. The four bins on each side of an internal training
direction change have twice the classification weight; neighborhoods stop at
gaps/edges and do not compound when overlapping. Inference needs none of those labels.

Whole experiments or seed groups are split before windows; both instruments stay
together. Decoded validation segment F1 selects settings, seeds and checkpoints,
with first-win ties. Final data is reserved before selection and opened after
freezing. [API workflows](api.md#administration) enforce these lifecycle boundaries.

## Artifacts and extension points

Trusted installed implementations are discovered through `regimex.models` entry
points. Lightweight descriptors expose parameters without importing scientific
libraries. A worker separately probes the declared backend and device. The current
backends support CPU. Unknown or ineffective supplied settings fail explicitly.

A bundle contains checksummed configuration, preprocessing, weights, validation
evidence and training history. Loading checks paths, sizes, archive bounds,
classes, shapes, finite values and semantic codec compatibility. Baseline arrays
use a numeric codec; TCN uses restricted CPU state-dictionary loading. Executable
estimator pickles and uploaded Python implementations are not accepted.

| Contract | Identity |
| --- | --- |
| Observation export | `2.1` |
| Generator | `4.0` |
| Direction annotations / automatic policy | `3.0` / `retrospective-directions/3.0` |
| Analysis profile / decoder | `direction-32/2.0` / `duration-viterbi/1.0` |
| Raw snapshot, dataset, reference set, bundle envelope | `1.0` |
| Baseline / TCN codecs | `aggregate-logistic/1.0` / `temporal-tcn/2.0` |
| TCN features / preprocessing | `temporal-price-shape/1.0` / `temporal-standardization/1.0` |
| Evaluation / comparison report | `2.2` / `model-comparison/2.0` |

Artifact identity includes ordered dataset content and split assignments. Raw
snapshots, training labels and evaluation references have separate identities.
Prediction never refits or retrains. Exact equality is checked when restoring
saved models. Different devices or library versions may produce different
floating-point results.
