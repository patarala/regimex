# TCN widths on direction-32/2.2

Four TCN widths were fitted to 96 synthetic training experiments with 147
instrument series. Validation used 30 separate experiments with 42 series.
Both partitions contained equal numbers of 256-step and 512-step experiments.
The generator assigned complete seed groups to partitions before simulation.
Automatic references and decoding used the service profile `direction-32/2.2`.

Every width used the same features, dilations, optimizer settings and batch
size of eight sequences. Each training seed received 96 optimizer updates.
The table shows the best decoded validation interval macro-F1 at updates
32, 64 or 96 for each seed. The mean gives each of the three repeats equal
weight.

| Hidden channels | Trainable parameters | Seed 11 | Seed 23 | Seed 47 | Mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 9,315 | 0.6938 | 0.6748 | 0.7138 | 0.6941 |
| 32 | 35,011 | 0.7044 | 0.7089 | 0.6862 | 0.6998 |
| 48 | 77,091 | 0.7470 | 0.7274 | 0.7396 | **0.7380** |
| 64 | 135,555 | 0.7253 | 0.7243 | 0.7297 | 0.7264 |

The 48-channel network scored highest for every seed. Its mean exceeded the
32-channel default by 0.0382, at about 2.2 times the parameter count.
Checkpoint selection and scoring used the same 30 experiments, so these
figures describe validation performance under this update budget.

The logistic baseline scored **0.5032** on the same validation partition.
It fitted 867 coefficients and intercepts using L-BFGS. Validation selected
`C = 1` from 0.1, 1 and 10.

## Independent validation

The best checkpoint across the three repeats of each width was frozen before
generating 64 further experiments, 32 at each length. Together they contained
97 instrument series. Each score below belongs to one fitted model. The
comparison above averaged three checkpoint scores per width.

| Frozen model | Selected seed | Selected setting | Independent interval macro-F1 |
| --- | ---: | --- | ---: |
| Logistic baseline | 11 | `C = 1` | 0.5097 |
| TCN, 16 channels | 47 | Update 96 | 0.7037 |
| TCN, 32 channels | 23 | Update 32 | 0.6509 |
| TCN, 48 channels | 11 | Update 96 | **0.7200** |
| TCN, 64 channels | 47 | Update 32 | 0.6893 |

The 48-channel model scored highest again. Its advantage over the 32-channel
model was 0.0691. The paired 95% bootstrap interval ranged from −0.0109 to
+0.1544, allowing either model to lead on another sample. The interval uses
2,000 draws of whole experiments with seed 1729 and keeps every instrument
with its experiment. It describes sampling uncertainty for these two frozen
models on this simulator.

To reproduce both comparisons, pass an unused directory to the first command.
It saves the training and validation data, plans, bundles and per-seed
scores. The second command verifies the frozen bundles and evaluates them on
the independent cohort. It checks for overlap with the first corpus before
reading annotations. Pass `--exposure-index PATH` to include additional known
seeds and identity hashes in this check.

```sh
uv run --locked --extra ml python scripts/compare_tcn_widths.py var/tcn-widths
uv run --locked --extra ml python scripts/confirm_tcn_widths.py var/tcn-widths
```
