# Models and interval semantics

Regimex labels sustained price direction in a completed experiment. Both models
can use prices before and after the step they are classifying. This makes their
intervals suitable for retrospective analysis.

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
running `regimex report --input export.json --output var/blind --blind`.

## Learned backends

Both backends return one probability for each of the three directions at every
original step. A shared decoder chooses intervals using the probabilities, a
penalty for switching direction and the availability and duration rules above.

| Backend | What it uses |
| --- | --- |
| Logistic baseline | Price, trade and order activity, spread and missing-value indicators. It computes the mean, population standard deviation, minimum and maximum over centered windows (32 and 128 steps by default). The training configuration selects whether to include whole-instrument summaries and how to handle gaps. |
| Temporal convolutional network (TCN) | Log prices, returns and price shape in centered windows of 32, 64, 128, 256 and 512 steps. The network receives 25 numeric channels and their 25 missing-value indicators. |

Training and validation contain separate experiments. All instruments from one
experiment and all members of a split group stay together. Validation and test
reject snapshots that repeat a complete instrument series from another snapshot,
even when the run and instrument IDs differ. Their references require one
snapshot per source run, split group and generator-scoped seed. When a run has
several revisions, choose the one to evaluate before registering its reference set.

Both backends need all three directions in their training targets and fit their
preprocessing on training data. Model selection requires at least one
validation interval with a known direction. The decoded validation interval
macro-F1 selects the settings or checkpoint, with the first best score winning a
tie. The administrative workflow covers benchmark reservation, freezing the
model and final evaluation. [API guide](api.md#administration)

The baseline uses class-balanced logistic regression with the LBFGS solver.
The [baseline window comparison](baseline-context.md) explains its summary
settings and the measured effect of missing prices.
The TCN uses AdamW and class-balanced cross-entropy. Within a known training
direction it can penalize changes in adjacent log probabilities. The four steps
on each side of a direction change receive double classification weight.
The weighting stops at gaps and sequence edges.

TCN training supports epoch-based early stopping and fixed optimizer updates.
For a fixed budget, set `budget` to `updates`, set the number of updates per
seed, and list checkpoints in `validation_updates`, ending at the last update.
Batches without known targets do not consume an update. The history records
the validation scores and the bundle identifies the selected seed and update.

## Saved models

Regimex discovers trusted backends through the `regimex.models` entry point.
The API reads their descriptors. Workers load the scientific libraries and
check device support. Both built-in backends run on CPU.

Each bundle contains weights, preprocessing, configuration, class order,
validation evidence and training history. At the end of training Regimex
reloads the bundle and checks that it reproduces the validation probabilities.
Analyses use the published weights and preprocessing. The baseline stores
numeric arrays. The TCN stores a restricted Torch state dictionary. Its
`temporal-tcn/2.0` and `temporal-tcn/3.0` codecs share the same inference
procedure and record training by epoch and optimizer update respectively.

See the [evaluation report](quality.md) for measured agreement and boundary accuracy.
