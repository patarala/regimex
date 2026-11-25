# Reference release benchmarks

These benchmarks cover the original 0.1.0 release, `tcn-0.2.0` and a separate
pair of offline fits. To find the score attached to the version serving your
installation, see [Model quality](../quality.md).

## Baseline and TCN on direction-32/2.1

Two offline candidates were fitted for this comparison. The table reports their
scores on 72 reserved experiments with automatic `direction-32/2.1` labels.

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

Run from the repository root with an unused output directory. The worktree
uses the source and lockfile recorded in the original training plan. Generation
writes snapshots and a split manifest. Fitting records the environment, training
settings and bundle digests. Scoring verifies the freeze before computing test
references and writes per-length reports and paired differences to the output
directory.

```sh
git worktree add --detach var/profile21-source 99c5bf7455c4efd2b31af5b778a0922a19332408
uv run --directory var/profile21-source --locked --extra ml \
  python scripts/compare_profile_21.py prepare ../profile21
uv run --directory var/profile21-source --locked --extra ml \
  python scripts/compare_profile_21.py fit ../profile21
uv run --directory var/profile21-source --locked --extra ml \
  python scripts/compare_profile_21.py score ../profile21
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
the benchmark score by 0.0764. The validation estimate uses only 32 experiments. [Compare scores in Training](../cli.md#compare-training-and-independent-scores)

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
