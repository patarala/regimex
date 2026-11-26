# Analysis workspace

Sign in with your access token. For a first analysis you can download the
[included JSON export](../tests/fixtures/experiment-v2.json) or
[generate your own](cli.md#generate-and-analyze-an-experiment).

## Analyze an experiment

1. Open **Analyses → New analysis**, then drop the export or select **Choose JSON files**.
2. Check the run and instrument names, choose a published model and select **Analyze**.
3. When the job finishes, the preview appears on the result page. **Download
   report** saves an interactive HTML file. **Download JSON** saves intervals,
   probabilities and statistics.

The worker continues if you leave the page. Find the result in **Experiments**
or return to its URL. If the connection drops during submission, use **Retry
request** in the same tab to recover the request. If processing fails, the
result page shows the cause and what to try next. Open **Details** for the analysis ID.

To analyze several exports with one model, select the files together. The
button shows the file count, for example **Analyze 3 files**. Select it and
keep the batch open while files are waiting to be submitted. Use **Resume
batch** after an interruption or **Retry failed** for
unsuccessful files. Accepted analyses appear in **Experiments**.

## Read a report

Open an experiment from **Experiments** and choose **Instrument** in its
report. The price chart shows executions and historical quotes. The timeline
below it shows growth, decline and sideways intervals. A long stretch without
executions or with missing prices gets an availability status and no direction.
Select a timeline interval to zoom the charts. The table below lists each
interval's statistics. An interval ends just before its stop step.

**Return standard deviation %** describes variability in adjacent log price
changes inside an interval. It requires at least two observed changes.
The missing-data plot shows fields with gaps. An empty book side means there
was no resting order on that side. An unknown side means quote history was
unavailable. [Interval definitions](models.md#observations-and-intervals)

Select **Pauses within trends** to highlight stretches where execution prices
changed little within a predicted rise or decline. A slow trend can also
trigger the overlay.
[Pause overlay accuracy](evaluations/pause-precision.md)

## Run again and compare

Open a result, select a model under **Analyze this experiment again** and
choose **Run again**. Both analyses use the same saved snapshot. In
**Experiments**, expand its analysis count, select two successful runs and
choose **Compare selected**. You can select runs on different pages or use
**Clear selection** to start again. Select a row of disagreements to zoom both
predictions to the same steps. **Download comparison** saves the comparison
for offline use.

## Delete saved work

**Details → Delete analysis** removes one result. **Manage → Delete experiment**
removes the export and its analyses. The preview lists dependencies and
anything that blocks deletion. Files awaiting cleanup still count toward your
quota. For an interrupted deletion, open **Pending deletions** and select
**Check deletion** or **Retry cleanup**.

## If something fails

| Symptom | Action |
| --- | --- |
| No model is available | Ask an administrator to [train and publish a model](training.md#evaluate-and-publish) or [activate a verified bundle](deployment.md#start-the-service). |
| Export rejected | Check the named field. A version 2.1 export supports one or two instruments, up to 4,096 steps and a 16 MiB file. [Export contract](integration.md#export-contract) |
| Job stays queued | Check the worker status on the job. An operator can inspect the installation with [service doctor](deployment.md#health-shutdown-and-maintenance). |
| Session expired | Sign in again with the token for the same account and reopen the result. |
| Storage full | Review pending deletions or ask the operator to check the [storage quotas](storage.md#owner-and-connection-quotas). |
