# Neural direction models on synthetic exchange and controlled drift

These experiments use frozen offline predictors. Exchange scores measure agreement
with automatic direction labels. Drift scores measure agreement with the schedule
used to generate controlled price paths. Each comparison below names its own
observations and reference. For a model trained in the service, read the evaluation
for its saved version in the workspace. [Find a model’s evaluation](../quality.md#find-the-score-for-a-model)

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
and expert confidence. In `direction-quality-mixture-v2`, three predictors
contribute to the exchange branch and four to the drift branch, including
models trained with different seeds. The gate assigns a weight to each of
the two branches. Four-fold validation kept all views of a seed together.

The weights were frozen before scoring 192 independent exchange experiments
and 96 independent drift seeds. Each drift seed supplied four schedules with
paired complete and masked observations. Scores are decoded interval macro-F1,
averaged equally over 256, 512 and 1,024 steps and, for drift, both views.

| Model | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Price-conditioned neural mixture | 0.9275 | 0.9700 |
| Block-level neural mixture (`direction-quality-mixture-v2`) | 0.9215 | 0.9834 |
| Block-level mixture trained with hard exchange labels only | 0.9257 | 0.9836 |
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

## Segment MoE on exchange prices and known drift

Three neural candidates used the same decoder and complete, ordered observations
for one instrument. The offline Segment MoE combined two independently trained
experts. The exchange expert fitted slopes allowing for a shift in price level
and learned how much movement to accumulate across consecutive segments with
the same slope sign. An availability gap reset its window. The other expert
had been trained to recover known drift. With their predictions fixed, a gate
learned how much weight to give each expert from price-change statistics in
each observed block. The other two candidates used a sign-symmetric multilayer
perceptron and a learned choice between existing specialists based on local
price windows. The [workspace architecture](../models.md#segment-moe) trains
its two experts and gate together on the selected training references.

An already opened panel of 192 exchange experiments and 96 drift seeds served
as the final **selection** set. The rule maximized the smaller of the candidate's
two margins over the best of three forests in each population.

| Candidate | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Segment MoE (offline) | 0.9474 | 0.9749 |
| Sign-symmetric network mixed with the previous model | 0.9037 | 0.9778 |
| Learned choice based on local windows | 0.9296 | 0.9743 |
| Best forest on each population | 0.9411 | 0.9577 |

The selected offline Segment MoE and the comparison models
were then scored on 288 new exchange experiments and 144 new drift seeds
(four schedules per seed, with complete and masked views). The table shows
decoded interval macro-F1 at
IoU ≥ 0.50, averaged equally across lengths and, for drift, across views.

| Model on independent groups | Exchange automatic reference | Known drift |
| --- | ---: | ---: |
| Segment MoE (offline) | 0.9360 | 0.9729 |
| Large interval forest | 0.9419 | 0.7930 |
| Expanded native forest | 0.9418 | 0.7952 |
| Mixed-supervision window forest | 0.9276 | 0.9397 |
| Price-conditioned neural mixture | 0.9159 | 0.9629 |
| Block-level neural mixture (`direction-quality-mixture-v2`) | 0.9198 | 0.9816 |

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

The frozen CPU bundle has manifest SHA-256
`f427e2b3c638facf901c5a55ffcaec02395b9a5dfac914661b01fecd4aa937d3`.
Its predictor uses the segment feature code from the evaluated revision.
Place the unpacked bundle in `var/release/segment-moe` with `predict.py` at the
top level. From the repository root, create the matching checkout once.

```sh
git worktree add --detach var/segment-moe-source e50e3df88862e010468d2dad8e93446246387619
```

Run the predictor on an experiment export from the repository root.

```sh
uv run --project var/segment-moe-source --locked --extra ml python var/release/segment-moe/predict.py \
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
