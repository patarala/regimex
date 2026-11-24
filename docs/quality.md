# Model quality

On synthetic exchange experiments, interval scores measure agreement with the
automatic labeling rule. On controlled price paths, a separately recorded drift
schedule provides the reference. Each model sees observations only. Agreement
with human judgments or on another simulator has yet to be measured.
[Interval definitions](models.md)

Within one experiment and instrument, an interval matches a reference when
the directions agree and their overlap covers at least half their union.
Each interval can match once, and the evaluator finds the largest set of
valid matches. Point F1 scores individual steps with a
reference direction. If the model leaves one of those steps unclassified, the
evaluator counts a false negative for its reference class. For both metrics,
macro-F1 gives equal weight to all three directions and assigns zero to an
undefined class F1. The pooled scores use counts across all experiments.

For profile `direction-32/2.2`, the [architecture comparison](temporal-comparison.md)
tests different temporal classifiers. The [training population comparison](training-populations.md)
holds the architecture fixed and varies the data and optimizer budget. The
[annotation checks](label-stability.md) test the reference rule against changes
in observed prices and against noisy paths with known pauses.
The [pause overlay comparison](pause-precision.md) measures how often the
report highlights a genuine price pause or a continuing trend.
The [direction classifier comparison](direction-context.md) tests whether measuring
price movement on either side of a level change improves the three predicted directions.
The [adaptive forest evaluation](adaptive-intervals.md) examines short sideways
intervals and the effect of missing observations on their neighbors.
The [1,536-experiment forest comparison](large-direction-forest.md) evaluates
multiscale price features and interval weighting against the released forest.

## How training labels affect the learned direction

Two networks trained on the same 192 controlled price paths with the same
initialization, features and optimization settings. One learned the automatic
labels and the other learned the scheduled drift. On 96 fresh groups, the
first network scored 0.7939 against the drift and the second scored 0.9690
(pooled interval macro-F1, paired difference +0.1751, 95% interval +0.1275 to
+0.2199). Against automatic labels the scores were 0.9192 and 0.7686. In this
generator the supervision alone changes which direction the model recovers.
[Noise, trade frequency and short pauses](label-stability.md#scaling-the-correction-by-trading-activity)

## Neural specialists for exchange prices and known drift

One specialist learned automatic direction labels from exchange experiments
and soft predictions from a forest trained on separate experiments. The other
learned the drift schedule of 960 price paths before noise and price rounding.
The mixture combines these specialists with earlier neural models. For each
observed block, a small network chooses a mixing weight using price changes
and expert confidence. Four-fold validation kept all views of a seed together.

The weights were frozen before scoring 192 independent exchange experiments
and 96 independent drift seeds. Each drift seed supplied four schedules with
paired complete and masked observations. Scores are decoded interval macro-F1,
averaged equally over 256, 512 and 1,024 steps and, for drift, both views.

| Model | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Previous price-conditioned neural mixture | 0.9275 | 0.9700 |
| New neural mixture | 0.9215 | 0.9834 |
| New mixture with hard native labels only | 0.9257 | 0.9836 |
| Native neural specialist alone | 0.9436 | 0.7704 |
| Drift neural specialist alone | 0.8075 | 0.9852 |
| Expanded native forest | 0.9467 | 0.7795 |
| Mixed-supervision window forest | 0.9367 | 0.9472 |

Replacement required the lower bound of the paired 95% interval to exceed
zero on exchange experiments and −0.015 on known drift. Compared with the
previous mixture, the changes were −0.0060 [−0.0334, +0.0158] and +0.0133
[+0.0052, +0.0220], respectively, from 5,000 group bootstrap draws. At 1,024
exchange steps, decline F1 fell from 0.880 to 0.780. The native expert alone
scored 0.905 for decline. The new mixture remains an offline candidate.

On a sample of its exchange training groups the new mixture scored 0.9357.
The estimated training advantage over the fresh exchange panel was 0.0142
(95% interval −0.0359 to +0.0615). This sample leaves substantial uncertainty
about the size of the generalization gap.

A second experiment tested a constant mixture of the two specialists. The
training groups selected 25% weight for the exchange specialist and 75% for
the drift specialist. On a fresh panel of 192 exchange experiments and 96 drift
seeds, this mixture scored 0.9092 and 0.9770. On those same groups the earlier
price-conditioned mixture scored 0.9321 and 0.9741, and the new block-level
router scored 0.9157 and 0.9837. The constant mixture's exchange score fell by
0.0229 relative to the earlier mixture (paired 95% interval −0.0437 to −0.0032).

If you have the generated `direction-quality-mixture-v2` package, copy the
whole directory to a machine with Regimex and its ML dependencies installed.
The package manifest SHA-256 is
`457298f1f1387f1e8002759e611c46b83847fce6401bd55c358f79123fae3f96`.
With the ML Python environment active, run these commands from the directory
containing the package and your export. The predictor verifies its weights
and writes a new JSON file containing the intervals for each instrument.

```sh
sha256sum direction-quality-mixture-v2/manifest.json
python direction-quality-mixture-v2/predict.py market-export.json --output directions.json
```

### Direction persistence across adjacent price segments

Three neural candidates used the same decoder and complete, ordered observations
for one instrument. The first fitted a slope on each price segment, allowing a
one-time shift in price level, and learned how much movement to accumulate across
consecutive segments with the same slope sign. Its second head weighs this
exchange specialist against a network trained on known drift. A gap resets the
window. The other candidates used a sign-symmetric multilayer perceptron and
a learned choice between existing specialists based on local price windows.
[Persistence architecture](../src/regimex/backends/price_persistence.py)

An already opened panel of 192 exchange experiments and 96 drift seeds served
as the final **selection** set. The rule maximized the smaller of the candidate's
two margins over the best of three forests in each population.

| Candidate | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Direction persistence | 0.9474 | 0.9749 |
| Sign-symmetric network mixed with the previous model | 0.9037 | 0.9778 |
| Learned choice based on local windows | 0.9296 | 0.9743 |
| Best forest on each population | 0.9411 | 0.9577 |

The selected model and all comparison models were then scored on 288 new
exchange experiments and 144 new drift seeds (four schedules per seed, with
complete and masked views). The table shows decoded interval macro-F1 at
IoU ≥ 0.50, averaged equally across lengths and, for drift, across views.

| Model on independent groups | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Direction persistence | 0.9360 | 0.9729 |
| Large interval forest | 0.9419 | 0.7930 |
| Expanded native forest | 0.9418 | 0.7952 |
| Mixed-supervision window forest | 0.9276 | 0.9397 |
| Previous price-conditioned neural mixture | 0.9159 | 0.9629 |
| Previous block-level neural mixture | 0.9198 | 0.9816 |

The acceptance rule required positive one-sided simultaneous 95% lower bounds
against all three forests in both populations. Paired resampling used 20,000
draws of whole seed groups with a Bonferroni correction for six contrasts.
All three drift contrasts passed. Against its strongest forest the drift gain
was +0.0332 (adjusted lower bound +0.0172). None of the exchange contrasts
passed. Against the large interval forest the difference was −0.0059
(lower bound −0.0396). The largest exchange deficit was at 512 steps, with
growth F1 of 0.890 versus 0.933 for the large forest. At 1,024 exchange steps
the network scored 0.9368 overall versus 0.9268 for that forest.

On 120 exchange training groups, after matching the independent panel's
instrument-count mix, the network scored 0.9598. The estimated training
advantage was +0.0238 (independent group bootstrap 95% interval −0.0225 to
+0.0674). The known-drift training generator uses different schedules and noise
from the independent panel. These measurements only cover simulated prices.

The model remains available for offline use. Its CPU inference bundle has
manifest SHA-256 `f427e2b3c638facf901c5a55ffcaec02395b9a5dfac914661b01fecd4aa937d3`.
To classify an export, unpack the bundle in `var/release/direction-persistence`
and run the following command.

```sh
uv run --locked --extra ml python var/release/direction-persistence/predict.py \
  market-export.json --output intervals.json
```

The command writes a new file containing intervals per instrument, the export
digest and the model digest. To reuse the weights in one Python process, add
the bundle directory to the import path and call
`accumulation_direction.load_predictor()` once.

With warm weights and one CPU thread, median prediction and decoding time was
51, 89 and 180 ms per instrument at 256, 512 and 1,024 steps. The expanded
forest took 19, 48 and 102 ms. These timings include feature extraction. They
use three alternating repeats on each of 12 exchange development series per
length and exclude service queueing, file I/O, report rendering and model load.

## Baseline and TCN on direction-32/2.1

| Steps | Baseline interval F1 | TCN interval F1 | Baseline point F1 | TCN point F1 | Classifiable steps / all steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.4435 | 0.8119 | 0.5781 | 0.9149 | 8,416 / 9,984 |
| 512 | 0.4329 | 0.6648 | 0.7510 | 0.9132 | 16,896 / 18,944 |
| 1,024 | 0.3594 | 0.5738 | 0.6266 | 0.9127 | 27,191 / 37,888 |
| Pooled | 0.4080 | 0.6691 | 0.6756 | 0.9160 | 52,503 / 66,816 |

Both models classified every step with a reference direction. These 52,503 steps
accounted for 78.6% of the complete grid. Across 72 held-out experiments, the
TCN's pooled interval macro-F1 exceeded the baseline's by 0.2611 (paired 95%
bootstrap interval 0.1752 to 0.3664). Resampling kept all instruments in each
experiment together, using 1,000 draws and seed 1729. This interval describes
uncertainty across experiments for these two fitted candidates. On 1,024-step
runs the TCN's point F1 was 0.9127 while interval F1 fell to 0.5738.
The [sideways interval breakdown](interval-errors.md) uses validation examples
to show which periods the TCN merged, split or classified as a trend.

Both candidates used the same 96 training experiments and 24 validation
experiments. Each of the three lengths contributed 32 training, eight validation
and 24 held-out experiments. Each experiment and its instruments stayed in one
partition. The exchange generator recorded prices and trade aggregates at every
step. The [incomplete-export evaluation](incomplete-exports.md) compares the
same frozen models on paired full and masked exports from a separate population.

Validation selected regularization 1 from 0.1, 1 and 10 for the baseline. The
32-channel TCN trained for 240 optimizer updates with seed 29. Of checkpoints
80, 160 and 240, update 160 scored highest. Validation interval macro-F1 was
0.3783 for the baseline and 0.6655 for the TCN. Both offline candidate bundles
passed a save/load prediction check and were frozen before test references
were generated.

### Reproduce the comparison

Run from the checkout with an unused output directory. Generation writes
snapshots and a split manifest. Fitting records the environment, training
settings and bundle digests. Scoring verifies the freeze before computing test
references and writes per-length reports and paired differences to the output
directory.

```sh
uv run --locked --extra ml python scripts/compare_profile_21.py prepare var/profile21
uv run --locked --extra ml python scripts/compare_profile_21.py fit var/profile21
uv run --locked --extra ml python scripts/compare_profile_21.py score var/profile21
```

All three commands use the same deterministic population. The recorded
preparation compared all 192 experiments against the local exposure index of
6,785 seeds and 27,554 identity keys. It found no overlap. To repeat that audit,
supply the index as `--exposure-index PATH` during preparation. The population
digest is checked even when the historical index is unavailable. Include all
192 members in the exposure history for subsequent model selection.

| Input | SHA-256 |
| --- | --- |
| Dataset manifest content | `682b79aa7915a2d864ec35b3f781c254f643570b38ba8e6b1e9630de3a29ec5e` |
| Test-reference content | `f6f1630cf5dc73a56894935a59dd396e45ede28f9fac130c70104293535958be` |

## Training gaps in the 0.1.0 release

The published baseline and TCN used the same 512-step exchange experiments
and automatic references. Rescoring the saved weights gives the following
pooled interval macro-F1. The benchmark column reproduces the original
published evaluation with the 64 groups held out at the time.

| Model | Training (128 groups) | Validation (32) | Original benchmark (64) |
| --- | ---: | ---: | ---: |
| `baseline-0.1.0` | 0.7991 | 0.2652 | 0.4700 |
| `tcn-0.1.0` | 0.6437 | 0.7160 | 0.6396 |

The baseline scores 0.3291 higher on its training examples than on the
benchmark. For the TCN this gap is 0.0041, while its validation score exceeds
the benchmark score by 0.0764. The validation estimate uses only 32 experiments. [Compare scores in Training](usage.md#compare-training-and-independent-scores)

## Reference model on direction-32/2.0

The published `tcn-0.2.0` used a different training budget, profile and
population from the comparison above. These results describe its original
384-experiment benchmark.

| Experiment length | Interval macro-F1 | Point macro-F1 |
| --- | ---: | ---: |
| 256 steps | 0.8334 | 0.9130 |
| 512 steps | 0.7124 | 0.9026 |
| 1,024 steps | 0.6288 | 0.8808 |
| All lengths, pooled | **0.7129** | **0.8932** |

| Direction | Interval precision | Interval recall | Interval F1 |
| --- | ---: | ---: | ---: |
| Sideways | 0.6469 | 0.6547 | 0.6508 |
| Growth | 0.8121 | 0.7883 | 0.8000 |
| Decline | 0.7454 | 0.6389 | 0.6880 |

The model classified all 283,946 steps that had a reference direction. On the
longest experiments it produced 372 intervals against 412 references. Its
median interval lasted 285.5 steps, compared with 193.5 for the references.
This helps explain why the model can agree on many individual steps while
missing shorter intervals.

### Boundary sensitivity

A typed boundary records both directions of a transition, such as growth to
decline. The evaluator pairs transitions of the same type within blocks where
both the reference and the prediction have a direction. Typed boundary F1
counts the largest number of one-to-one pairs within the allowed displacement
(four steps in a standard evaluation). The separate lag summary pairs as many
transitions as possible without a distance cutoff, minimizing their total
absolute lag. A positive lag means the prediction was late.

The model produced 414 transitions against 454 references. Varying the allowed
displacement on these same transitions gave the following scores.

| Allowed displacement, steps | Typed boundary F1 |
| --- | ---: |
| 0 | 0.0392 |
| 4 | 0.1613 |
| 8 | 0.2327 |
| 16 | 0.3203 |
| 32 | 0.4124 |
| 64 | 0.4724 |

Few boundaries fall on the exact reference step. Larger tolerances admit
nearby transitions and can pair distinct changes. These scores describe how
sensitive agreement is to placement under the automatic labeling rule.

### Evaluation protocol

Training used 512 experiments of 512 steps (769 instrument sequences).
Validation used 192 experiments, with 64 each at 256, 512 and 1,024 steps.
Whole experiments and related seed groups stayed in one partition.
Preprocessing was fitted on training data.

Each training seed ran for 1,200 optimizer updates. Validation scored updates
300, 600, 900 and 1,200. The selected checkpoint was update 900 for seed 29.
The model, inference settings and evaluation code were frozen before opening
the reserved final data.

The final population included 128 experiments per length, half with one
instrument and half with two. All 576 instrument sequences and 344,064 grid
rows were retained. Further model selection requires a fresh final benchmark.

### Reproducing the reference scores

The `tcn-0.2.0` release includes the evaluator, references and per-step
probabilities used for the 384-experiment benchmark. The root of the unpacked
archive contains `BUILD.json` and `uv.lock`. Run the scoring command there to
use the evaluator shipped with the model.

```sh
uv run --locked --extra ml python -I release/evaluation/reproduce.py score --output var/reproduced-scores
```

If you have only the assets, unpack them under `release/` in your Regimex
checkout. A separate Git worktree provides the source revision recorded in
the release's `BUILD.json`. Run the commands from your current checkout.

```sh
git worktree add --detach var/tcn-0.2.0-source 6365c9daa4ce41f361ec9c56644905b1cf8e4176
uv run --project var/tcn-0.2.0-source --locked --extra ml python -I tests/support/check_reference_scores.py release/evaluation
uv run --project var/tcn-0.2.0-source --locked --extra ml python -I release/evaluation/reproduce.py score --output var/reproduced-scores
```

The quick check rescores the first experiment's 512 saved predictions and
compares its statistics with the release. For the full run, choose a new
output directory. The scorer verifies the packaged hashes and
compares every scientific field with the archived reports. For this release,
`verification.json` records 384 experiments, 576 instrument sequences and
344,064 steps across those sequences. Add `--predict` to also check that the
saved model produces exactly the archived probabilities on CPU.
