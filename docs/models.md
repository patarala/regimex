# Models and interval semantics

Regimex labels sustained price direction in a completed experiment. Its models
can use prices before and after the step they are classifying. The
[backend comparison](#learned-backends) describes how each model scores a step.

## Observations and intervals

An export holds one or two instruments on the same grid, up to 4,096 steps and
16 MiB. Every interval belongs to one instrument. Its start is inclusive and
its stop is exclusive, so `[8, 40)` covers 32 steps. The intervals cover every
step exactly once.

| Interval status | Meaning |
| --- | --- |
| `observed` | The interval has enough price observations to assign `growth`, `decline` or `sideways`. |
| `no_trades` | The observations establish a sustained pause in executions. Direction is null. |
| `insufficient_data` | Too few prices are available to classify this part of the experiment. Direction is null. |

With the current profile, `direction-32/2.2`, a zero trade count, zero volume,
or a price state of `carried` or `no_execution` establishes that a step had no
executions. A stretch of 32 or more inactive steps forms a pause even when
prices are missing. A stretch of 32 or more missing prices otherwise marks a
gap. Either rule also applies to a shorter stretch that spans the entire sequence.

The remaining blocks need at least four prices and 80% price coverage. Short
gaps and pauses can fall inside a classified interval. The minimum duration is
32 steps, or the length of the whole available block if shorter. Adjacent
intervals with the same direction merge.

### Reading interval statistics

- **Price change** is the percentage difference between the first and last
  available prices. Their step numbers show which part of the interval the
  calculation covers. A null with two distinct endpoint steps means the
  percentage exceeds the finite numeric range.
- **Return SD** is the population standard deviation of one-step log returns.
  Both steps of each return must have prices. A gap breaks the sequence, and
  fewer than two returns gives a null result.
- **Trade count and volume** sum the observations available for each field.
  A total is null if the field is missing throughout. The coverage fractions
  show how much of the interval contributed. A null volume with positive
  coverage means the observed total exceeds the finite numeric range.
- **Active-trade fraction** is the share of steps with a positive count among
  steps with a known count. **Longest no-trade run** counts consecutive steps
  with a known zero count. The offline summary's `steps_without_executions`
  also uses volume and price state, so it can count pauses when the count is
  missing.

Saved models use the profile in their bundle. Profile `direction-32/2.0`
recognizes pauses from trade counts alone and gives missing prices precedence.
Both 2.0 and 2.1 fit absolute log prices and use an exact threshold comparison.
To use the current rules for a published model, train and publish a new version.

## Synthetic market

Create a dataset through `POST /api/v1/datasets`. Supply `plan.market` to set
the initial prices and participant profiles. With the default sampler,
experiments alternate between one and two instruments. The seed assigns each
instrument a synthetic symbol such as `BTC/USD`. Each participant shares one
cash balance across its instruments.

Each participant acts once per step in a fixed order, with the separately
configured test participant last. A random trader samples buy, sell or hold
from its profile. A trend follower compares the latest execution price with
the price a configured number of trades earlier. It holds while the history
is too short or the change falls below its threshold.

Orders match at the resting price, best price first and oldest order first at
a given price. An order can fill partially. If it reaches the participant's own
resting order, the exchange cancels the incoming remainder. Prices and quantities
use integer ticks and lots. The default tick and lot sizes are both 0.01.

After each step the generator records executed trades, accepted orders and the
best bid and ask. An empty book side has a null quote. A step without trades has
zero count and volume and carries forward the last price (initially the starting
price). The sampler draws inactive (5%), one-sided (5%), fixed-price (10%) and
mixed participant configurations (80%). Price directions emerge from the trades.

## Automatic reference labels

The policy `retrospective-directions/3.0` supplies reproducible labels for
training and evaluating models. It uses the same availability rules as the
decoder. On each available block it divides log prices into pieces by fitting
straight lines. Adding a piece incurs a penalty based on the local noise level.

To assign a direction, the policy estimates a slope in each piece. It also
tries a line with the same slope on both sides of a single price-level change,
with at least two observed prices per side. The fit with the smallest residual
error wins. This lets a price that moves once and then stays flat receive a
sideways label despite a large end-to-end change.

A slope must explain at least half of the price variation in its fit. Consecutive
supported slopes of the same sign form a run. The policy adds their fitted log
movements and assigns growth or decline when the absolute sum reaches
`log1p(0.005)`. Profile 2.2 fits log price ratios relative to the first
available price in the block. It accepts movements within floating-point
roundoff of the threshold so that changing quote units preserves a borderline
label. Other runs receive sideways. Human reference sets are recorded
separately. A reviewer can see observations with the predictions hidden by
running `regimex report --input export.json --output var/blind --blind`. The review
folder contains the observations and a draft keyed by the original export’s SHA-256.
The reference import uses that digest to match the completed draft to its snapshot.

## Learned backends

Each backend returns one probability for each of the three directions at every
original step. A shared decoder chooses intervals using the probabilities, a
penalty for switching direction and the availability and duration rules above.

| Backend | How it scores a step |
| --- | --- |
| Adaptive direction forest | Fits trends over price-dependent spans, then averages tree probabilities using the span and its neighboring trends. Fits each side of a sustained availability gap independently. |
| Logistic baseline | Applies fitted linear weights to summaries of price, trades, orders, spread and missingness. [Window settings and gap behavior](baseline-context.md) |
| Temporal convolutional network (TCN) | Learns price patterns from centered windows and whole-instrument statistics using dilated convolutions. [Inputs, network and training](tcn.md) |

Training and validation contain separate experiments. All instruments from one
experiment and all members of a split group stay together. Validation and test
reject snapshots that repeat a complete instrument series from another snapshot,
even when the run and instrument IDs differ. Their references require one
snapshot per source run, split group and generator-scoped seed. When a run has
several revisions, choose the one to evaluate before registering its reference set.

All three backends need every direction in their training targets. Learned
weights and preprocessing use training data only. Model selection requires at
least one validation interval with a known direction. The decoded validation interval
macro-F1 selects the settings or checkpoint, with the first best score winning a
tie. The administrative workflow covers benchmark reservation, freezing the
model and final evaluation. [API guide](api.md#administration)

### Adaptive direction forest

Choose **Adaptive direction forest** in the training workspace. Each run fits
one forest and reports its decoded validation interval macro-F1. The seed
controls the randomized splits. The defaults are 40 trees, maximum depth 12
and seed 173. These are the settings used in the [direction model comparison](direction-context.md).
The number of split decisions in the fitted forest depends on the training data.

For the API, set `architecture_id` to `adaptive`. The following fields configure
the forest and its single fit:

```json
{"parameters":{"trees":40,"max_depth":12},"policy":{"optimizer":"extra_trees","seeds":[173]}}
```

## Saved models

When importing a bundle with `regimex service register-model`, use
`--runtime-file PATH` to pin its inference settings. For the logistic baseline
and TCN, this configuration limits numerical work to one CPU thread and each
batch to two instrument sequences:

```json
{"device": "cpu", "threads": 1, "max_batch_size": 2}
```

The adaptive forest fits its trees sequentially (`n_jobs=1`) and predicts one
instrument at a time. A larger thread or batch limit keeps that execution order.
For a thread-count change alone the CLI also accepts `--threads N`. Each saved
analysis records its resolved inference policy. Registering an existing model
version with a different policy returns a conflict.

The inference setting `model_cache_budget_bytes` defaults to 512 MiB. Regimex
checks the bundle size at registration. The forest also checks the size of its
decoded tree arrays before inference. The service setting
`REGIMEX_CACHE_BUDGET_BYTES` limits the combined size of bundles in the
local cache and defaults to 512 MiB.

Regimex discovers trusted backends through the `regimex.models` entry point.
The API reads their descriptors. Workers load the scientific libraries and
check device support. All three built-in backends run on CPU.

Each bundle contains weights, preprocessing, configuration, class order,
validation evidence and training history. At the end of training Regimex
reloads the bundle and checks that it reproduces the validation probabilities.
Analyses use the published weights and preprocessing. The baseline stores
its coefficients as numeric arrays. The forest stores tree splits, thresholds
and class probabilities in a numeric NPZ file with a versioned preprocessing
record. The TCN stores a restricted Torch state dictionary. Its
`temporal-tcn/2.0` and `temporal-tcn/3.0` codecs share the same inference
procedure and record training by epoch and optimizer update respectively.

See the [evaluation report](quality.md) for measured agreement and boundary accuracy.
