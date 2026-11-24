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

A sideways interval means the model found no sustained rise or decline in
the available prices. It can contain many trades. The report's **Pauses within
trends** overlay highlights trades that stay near one price inside a predicted
trend. The highlight appears only in the report. [Reading the report](usage.md#read-a-report)

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
with at least two observed prices per side. The simpler line wins if the
residual errors agree within roundoff. This lets a price that moves once and
then stays flat receive a sideways label despite a large end-to-end change.

A slope must explain at least half of the price variation in its fit. Consecutive
supported slopes of the same sign form a run. The policy adds their fitted log
movements and assigns growth or decline when the absolute sum reaches
`log1p(0.005)`. Profile 2.2 fits log price ratios relative to the first
available price in the block. It accepts movements within floating-point
roundoff of the threshold so that changing quote units preserves a borderline
label. Other runs receive sideways.

To prepare an independent review with predictions hidden, run the following
from the checkout with a new output directory.

```sh
uv run --locked --extra viz regimex report --input export.json --output var/blind --blind
```

Open `var/blind/index.html`. The reviewer's template is
`var/blind/annotation-draft.json`. Its `experiment_id` is the export's
`content_sha256`, the digest of its canonical JSON content. Reference import
uses that value to match the completed draft to its snapshot.

## Learned backends

Each backend returns one probability for each of the three directions at every
original step. A shared decoder chooses intervals using the probabilities, a
penalty for switching direction and the availability and duration rules above.

| Backend | How it scores a step |
| --- | --- |
| Direction persistence | Combines accumulated movement across consecutive price segments with a learned network that compares local slope and fit. |
| Adaptive direction forest | Fits trends over price-dependent spans, then averages tree probabilities using the span and its neighboring trends. Fits each side of a sustained availability gap independently. |
| Logistic baseline | Applies fitted linear weights to summaries of price, trades, orders, spread and missingness. [Window settings and gap behavior](baseline-context.md) |
| Temporal convolutional network (TCN) | Learns price patterns from centered windows and whole-instrument statistics using dilated convolutions. [Inputs, network and training](tcn.md) |

Training and validation contain separate experiments. All instruments from one
experiment and all members of a split group stay together. Validation and test
reject snapshots that repeat a complete instrument series from another snapshot,
even when the run and instrument IDs differ. Their references require one
snapshot per source run, split group and generator-scoped seed. When a run has
several revisions, choose the one to evaluate before registering its reference set.

The packaged backends need every direction in their training targets. Learned
weights and preprocessing use training data only. Model selection requires at
least one validation interval with a known direction. When a backend compares
settings or checkpoints, it selects the highest decoded validation interval
macro-F1 and breaks ties with the first result. The administrative workflow
covers benchmark reservation, freezing the model and final evaluation.
[API guide](api.md#administration)

### Direction persistence

Choose **Direction persistence** to fit a neural model on the selected
development dataset. The model fits slopes on adjacent price segments and
learns how much earlier and later movement supports the current direction.
A second network compares the local slope patterns in both price directions.
A learned weight combines their probabilities using the variability of price
changes in the current observed block. Availability gaps reset both contexts.

The worker initializes all three components from scratch and fits block
standardization on training experiments. The default run uses a context width
of 32, AdamW and up to 24 epochs, with one validation score per epoch. It stops
after six validation checks without improvement and saves the highest-scoring
checkpoint. Training shows the loss, validation interval macro-F1 and selected
repeat. Saved weights support validation evaluation and publication through
the same controls as the other backends.

The [offline direction-persistence comparison](evaluations/neural-direction.md#direction-persistence-across-adjacent-price-segments)
used a separate ensemble pretrained on known drift. Its benchmark scores apply
only to that saved offline bundle. A workspace fit records its own scores.

### Adaptive direction forest

Choose **Adaptive direction forest** in the training workspace. Each run fits
one forest and reports its decoded validation interval macro-F1. The seed
controls the randomized splits. The defaults are 40 trees, maximum depth 12
and seed 173. These are the settings used in the [direction model comparison](direction-context.md).
The number of split decisions in the fitted forest depends on the training data.

After fitting, the worker measures how the interval score changes as more
trees contribute their votes, in the order they were fitted. It computes
validation features once, averages the class probabilities at each sampled
tree count and applies the usual interval decoder. A smaller memory budget
can reduce the number of measurements. The reported model score uses every tree.

**Price features** controls how the forest reads the observed price path.
The default, **Greedy**, splits a span at the step that most improves its fit,
then considers its two sides separately. **Multiscale** fits a sequence of
spans at three boundary penalties. It measures the slope both with and without
an adjustment for a shift in the price level.

**Training weights** controls how much each labeled step contributes to the
fit. **Per step** weights all steps equally. **Interval** gives more weight
to steps in shorter reference intervals while preserving the total weight
of each direction class. Both settings use the same inference inputs.

For the API, set `architecture_id` to `adaptive`. The following fields configure
the forest and its single fit:

```json
{"parameters":{"trees":40,"max_depth":12},"policy":{"optimizer":"extra_trees","seeds":[173]}}
```

To use the alternative settings, add `"feature_set": "multiscale"` or
`"sample_weighting": "interval"` to `parameters`.

For a saved forest with one history point, the operator can attach a curve
measured on the original validation set. The JSON for
`--forest-validation-curve` contains `bundle_sha256`, `dataset_sha256`
(the validation dataset), `validation_reference_sha256`,
`metric` (`decoded_segment_macro_f1/iou-0.50`) and `points`.
Each point has `trees` and `validation_segment_macro_f1`. Supply at least
two points in increasing tree count. The last count and score must match
the full forest. Regimex stores the measurements with the run's metrics
and preserves the original bundle and fit history.

The imported model `direction-blend-500-v3` runs both the released greedy
forest and the larger multiscale forest, then averages their class
probabilities before decoding. Its bundle includes both parents' weights and
training records. [Evaluation and import command](large-direction-forest.md#combining-two-trained-forests)

## Direction quality reports

The operator can attach interval scores to a completed run with
`regimex service register-model --direction-quality FILE`. Each row covers
one reference and one split. A controlled generator can supply its drift
schedule as an independent reference even when the model trained entirely on
automatic labels. A low score against that schedule can reveal errors that
agreement with the training labeler hides.

| JSON field | Value |
| --- | --- |
| `bundle_sha256` | SHA-256 of the model's verified `manifest.json`. |
| `protocol_sha256` | SHA-256 of the plan defining groups, references, inference and matching. |
| `metric`, `class_names` | `decoded_segment_macro_f1/iou-0.50` and `["sideways","growth","decline"]`. |
| `rows[].reference`, `split` | `simulator` or `known_direction`, and `train`, `validation` or `independent`. |
| `rows[].reference_sha256` | Identity of the reference set used for that measurement. Training and selection references must appear in the bundle's provenance. For independent data use a digest binding the actual reference labels or their reproducible schedule. |
| `rows[].groups[]` | Each group has a stable `group_sha256` and a `counts` array of three `[true_positive,false_positive,false_negative]` triples in class order. |

Match intervals within each instrument and view before pooling counts by
dependence group. For simulator seeds the scoring script hashes the canonical
JSON array `["seed", namespace, decimal_seed_string]`. For an external
group without a seed it hashes `["group", namespace, group_id]`. Use the same
identity across lengths, instruments and views of the same group. The service
rejects group overlap between reported splits. The evaluator must also check
exposure against datasets outside the report before calling a cohort independent.

The row for the model's original validation reference set must reproduce the
selection score recorded in the bundle. The service sums each class's counts
across all groups, computes its F1 and
averages the three classes. An absent class contributes zero. A row with
no reference intervals cannot be scored. When extending a report, include
all previously submitted rows and the same protocol. Re-importing the
same measurements preserves the original history. [Score a saved model](cli.md#compare-training-and-independent-scores)

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
check device support. The packaged backends run on CPU.

Each bundle contains weights, preprocessing, configuration, class order,
validation evidence and training history. At the end of training Regimex
reloads the bundle and checks that it reproduces the validation probabilities.
Analyses use the published weights and preprocessing. The baseline stores
its coefficients as numeric arrays. The forest stores tree splits, thresholds
and class probabilities in a numeric NPZ file with a versioned preprocessing
record. Direction persistence stores a fixed layout of little-endian float32 weights
and the training mean and scale for its block features. The TCN stores a restricted Torch state dictionary. Its
`temporal-tcn/2.0` and `temporal-tcn/3.0` codecs share the same inference
procedure and record training by epoch and optimizer update respectively.

The [reference TCN benchmark](evaluations/reference-releases.md#boundary-sensitivity)
reports agreement with automatic labels at several boundary tolerances.
