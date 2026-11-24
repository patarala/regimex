# Direction classifiers

The comparison predicts sideways, growth and decline from completed observations.
Its primary metric gives each direction equal weight. A predicted interval
matches a reference of the same class when their intersection covers at least
half their union. Each interval can match once.

## Models and data

- **Window model** (`context`). Two logistic classifiers answer separate
  questions. Is there a trend, and if so, is it rising? They use windows of
  32, 64, 128 and 256 steps. Each window measures the slope with and without
  allowing an abrupt change in level. Reflecting the log-price path swaps the
  growth and decline probabilities.
- **Adaptive model** (`adaptive`). This model splits an observed block where
  two fitted lines explain the price better than one. It repeats the split at
  three penalties. A forest of 40 randomized trees uses the movement within
  each piece, its fit after removing a single level change, and its neighbors.

Both models keep the original time grid. Unavailable blocks split the context.
They can use later observations in the same block because the service analyzes
completed simulations. Their outputs are unsuitable as live forecasts.

The training partition supplied 192 experiments and 139,825 steps with a
reference direction. On 72 fresh validation experiments we compared each model
alone, a half-and-half probability average of each with a 48-channel TCN
(`mixed-192`), and that TCN alone. Selection used pooled interval macro-F1
after decoding. The reserved final set contains 144 experiments, 48 at each
of 256, 512 and 1,024 steps, half with two instruments in each length group.
All instruments in an experiment stay together for selection and uncertainty
estimates.

## Results

The standalone adaptive forest won on validation with interval macro-F1
0.7575. The same forest averaged with the TCN scored 0.7483, the window
model 0.7295, the window model averaged with the TCN 0.6767, and the
training-matched TCN 0.6279. Only the validation winner advanced to the
reserved comparison with the TCNs.

| Model on the same 144 held-out experiments | Interval macro-F1 | Sideways F1 | Growth F1 | Decline F1 | Point macro-F1 | Short sideways matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive forest | **0.7859** | **0.7317** | **0.8400** | **0.7861** | 0.9157 | 34 / 68 |
| Published TCN 0.2.0 | 0.7259 | 0.6476 | 0.7786 | 0.7514 | 0.9153 | 32 / 68 |
| TCN trained on the same 192 experiments | 0.6491 | 0.5633 | 0.7173 | 0.6667 | 0.8635 | 15 / 68 |
| TCN 0.1.0 | 0.6272 | 0.5473 | 0.6975 | 0.6369 | 0.8662 | 9 / 68 |

Short sideways intervals last fewer than 64 steps. Each model used its
own decoder settings against the same automatic profile 2.2 references.
Decoding the published models with the common profile gave the same scores
on these exports.

The adaptive forest gained 0.0601 interval macro-F1 over the published TCN
0.2.0 (paired 95% interval +0.0131 to +0.1020). Its point macro-F1 gain
was 0.0004 (−0.0219 to +0.0193). The new model located whole intervals
better with similar agreement on individual steps. The intervals come
from 2,000 paired draws of whole experiments within each length group
and describe these fitted weights.

| Experiment length | Adaptive interval macro-F1 | Training-matched TCN interval macro-F1 |
| --- | ---: | ---: |
| 256 | 0.8741 | 0.7673 |
| 512 | 0.7207 | 0.5699 |
| 1,024 | 0.7860 | 0.6433 |

Across 109,452 steps with a reference direction, the adaptive forest also
reduced multiclass Brier score from 0.1964 to 0.1191 and log loss from
0.3679 to 0.2334 relative to the training-matched TCN. Both scores
measure the probabilities before interval decoding. Lower is better.

## Independent price paths and gaps

The native benchmark measures agreement with the automatic labels. Those
labels and the forest both use local line fits, so we also tested trajectories
whose direction was specified before adding noise, tick rounding, executions
or missing data. The 32 independent seed groups covered steady trends,
isolated level changes, pauses, reversals and pauses before reversals. Each
of the 320 paths was scored with complete data and with a 40-step gap.
Full and masked exports of the same path form one pair.

| All 640 views, 2,240 reference intervals | Interval macro-F1 | Sideways F1 | Growth F1 | Decline F1 |
| --- | ---: | ---: | ---: | ---: |
| Adaptive forest | 0.9629 | 0.9268 | 0.9863 | 0.9756 |
| Training-matched TCN | 0.8421 | 0.6697 | 0.9127 | 0.9439 |

On the masked views alone, interval macro-F1 was 0.9655 for the forest and
0.8252 for the TCN. Removing the 40 steps also changed the forest's predicted
direction on 2,553 of 117,760 steps still available in both views (2.17%).
The TCN changed 548 (0.47%). A gap can change the forest's choice of fitted
pieces on the remaining observations. Accuracy on these synthetic paths
does not establish agreement with a human analyst or a different simulator.

## Reproduce

Start in a Git checkout that contains the historical commits below. The
original fits and scores are bound to the source at `7090f35`. Use its
scripts and locked ML environment for the comparison. Input and output
paths are relative to the checkout where you run the commands.

- Place the [mixed 192 training population and its frozen TCN](evaluations/training-populations.md#reproduce)
  under `var/verification/training-populations/`. The comparison checks the
  control bundle's manifest against its recorded SHA-256.
- The TCN 0.1.0 bundle belongs in `release/models/tcn/`. The TCN 0.2.0 check
  also requires its publication working directory, including the original
  training manifest and exposure inventory, under `var/model-finalization/`.
  Pass `--publication` to both published-model commands to use another path.
- Supply an exposure index at `var/verification/direction-context-prior.json`.
  Build one from known dataset manifests and indices with
  `scripts/build_exposure_index.py`. Reproducing the archived evidence digests
  requires the original index, which covered 9,873 seeds and 37,952 identity keys.

Use new directories for `trial` and `transfer`.

```sh
trial=var/verification/direction-context
transfer=var/verification/direction-context-transfer
prior=var/verification/direction-context-prior.json
source_dir=var/direction-comparison-source
git worktree add --detach "$source_dir" 7090f35
run_comparison() {
  script=$1
  shift
  uv run --project "$source_dir" --locked --extra ml python "$source_dir/scripts/$script.py" "$@"
}
run_comparison compare_direction_context reserve "$trial" --prior "$prior"
run_comparison check_direction_context_transfer reserve "$transfer" "$trial" --prior "$prior"
run_comparison check_direction_release reserve "$trial"
run_comparison compare_direction_context train "$trial"
run_comparison compare_direction_context freeze "$trial"
run_comparison compare_direction_context score "$trial"
run_comparison check_direction_context_transfer score "$transfer" "$trial"
run_comparison check_direction_release score "$trial"
```

Native scoring writes `$trial/results.json`, transfer scoring writes
`$transfer/results.json` and
the published-model comparison writes `$trial/published/results.json`.

To inspect the winner's intervals and three probabilities per step on your own
export, use the same shell function and frozen trial. The output records the input
and selection digests and preserves instrument and step order.

```sh
run_comparison predict_direction_context "$trial" example.json var/predictions.json
```

## Use the trained forest

In the training workspace, choose **Adaptive direction forest** to fit a new
model on your own dataset. The [model guide](models.md#learned-backends)
describes its settings. To package the fitted model evaluated above, set
`trial` and `transfer` to the archived comparison directories. Packaging
checks their result digests. It also records the backend source in the
manifest, so recreating the published bundle requires the original packaging
revision, `a27c034`. Use new paths for the worktree and output.

```sh
packager=var/adaptive-package-source
git worktree add --detach "$packager" a27c034
git show 7090f35:scripts/adaptive_direction_model.py > var/frozen-adaptive.py
uv run --project "$packager" --locked --extra ml python "$packager/scripts/package_adaptive_direction.py" \
  "$trial" "$transfer" var/verification/training-populations/development/mixed-192 \
  var/frozen-adaptive.py var/release/adaptive-direction-0.1.0
```

Packaging checks the recorded model and gates, restores the native bundle,
and compares every validation probability with the frozen implementation.
Use the emitted manifest SHA-256 when registering the resulting `bundle/`
with your configured service. Activate the version to make it available to
workspace users. An already published version can still be selected by ID
after another model becomes active. [Model registration and activation](deployment.md#start-the-service)
