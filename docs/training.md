# Train and publish a model

You need an administrator token and a running `admin` worker. Open **Training**
and choose **New run**. To follow a job submitted from a terminal, open the run
link printed by the command. [Terminal workflows](cli.md#find-a-run-started-in-the-terminal)

## Prepare the data

1. Choose **Prepare training data**, set the experiment counts and length,
   and select **Generate datasets**. The service creates observations and
   automatic direction labels in the background. You can close the tab and
   open its URL on another device, or find it under **Saved preparations**.
   **Retry stage** resumes a failed or cancelled job using the completed
   stages. To continue a setup saved by an earlier browser version, select
   **Resume previous preparation** in the browser where you started it.
   Whole experiments stay together in each split.
2. Review the inventory of previously used data. If prior uses are missing,
   [register them through the API](api.md#2-reserve-final-data-before-selection).
   Check the confirmation box, choose **Reserve benchmark**, then **Use
   prepared data**. Final test labels stay sealed during model selection.
3. For another run on the same development data, select its dataset,
   compatible training and validation references, and existing reservation.
   For imported observations, follow the [API preparation steps](api.md#1-create-observations-and-references).

## Start training

Choose an architecture and review **Training policy**. A run name is optional.
The form fills in defaults for the chosen model. **Advanced settings** lets
you override model parameters and runtime limits. To enter the whole run
configuration as JSON, choose **Edit JSON** at the top of the form. **Use form**
applies your JSON edits to the controls. If a value cannot be shown in the
form, you can correct it in the editor or choose **Discard JSON edits** to
return to your previous settings. **Start training** submits from either view.

| Architecture | What to look for during training |
| --- | --- |
| Adaptive direction forest | After fitting the trees, the worker scores forests of increasing size on validation data. The saved model uses all its trees. [Forest features and weights](models.md#adaptive-direction-forest) |
| Logistic baseline | One validation score per regularization setting. [Logistic baseline](baseline-context.md) |
| Temporal convolutional network (TCN) | Loss by epoch or optimizer update and validation scores at scheduled checks. [TCN settings and checkpoints](tcn.md) |
| Segment MoE | Training loss for the mixture of experts and decoded validation interval macro-F1 by epoch. [Experts, gate and model settings](models.md#segment-moe) |

## Read a run

The run page checks for new measurements while it is visible. If you switch
tabs, training continues and the charts catch up when you return. Hover over a
chart to inspect a measurement. Validation scores appear only for epochs or
updates where a check ran. For repeated fits, **Training repeat** selects a
curve and **used for this model** identifies the repeat that supplied the saved
weights. For each repeat, the run shows why training ended and the epoch or
update with its best validation score when the backend recorded it. Tied scores
keep the earlier checkpoint for TCN and Segment MoE.

To change a workspace run, choose **New run from this run**. The form opens
with that run's data and settings. To compare two runs, choose **Select runs**
in the history, select both runs, then choose **Compare**. The comparison
updates while either run is in progress. For workspace runs,
the score difference appears when both used the same development dataset,
validation labels and analysis profile. Imported runs can use different training
datasets and still be compared when their validation dataset, validation
partition, labels and analysis profile match.

## Evaluate and publish

Under **Model evaluation → Validation**, choose **Evaluate model** to score the saved weights
against the run's validation references. Once it finishes, inspect the interval
and step scores and the results for each direction on the run page.
When the report includes confidence intervals, the range below each F1 score
describes sampling uncertainty across experiments. The evaluation resamples
whole experiments, keeping all their instruments together.
**All fields** opens the evaluation details. **Download JSON** saves them.
Under **Final benchmark**, choose **Freeze this model** after training finishes
and you have selected a candidate. Confirm the version and benchmark. Freezing
closes selection for every run using that reservation. Choose **Open held-out
test** to unseal the test references, then choose **Evaluate frozen model**.
You can close the page and return to follow the job or read its report.
The report identifies the reference source used for the held-out experiments.
Further model selection requires a fresh benchmark.
[Final evaluation](api.md#4-freeze-and-evaluate-final-data) · [How scores are measured](quality.md)

**Use as default** publishes the model for new analyses after confirmation.
Publication records its completed final evaluation when available, otherwise
its validation evaluation. Accepted analyses retain their original model version.

## Manage saved runs

**Archive run** moves a finished run out of the current history and keeps its
weights, charts and evaluations. Use **Archived runs → Restore run** to bring
it back. To free storage, open **Run details and settings → Delete model
weights**. The preview shows anything blocking deletion. You can also archive
the run when confirming. A retired version retains its recorded metrics and
cannot be used to make new predictions.
