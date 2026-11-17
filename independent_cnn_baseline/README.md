# Independent CNN Baseline

Hein's independent baseline package for the group project.

**Five-hour continuation stopped and verified, 2 November 2026 13:30:23 MYT.**
The deadline was 13:27:11 MYT; the active crop epoch finished and checkpointed
before exit. New outcomes: **0 COMPLETE, 5 PRUNED, 1 pending** (trials8–13),
separate from the inherited2 COMPLETE/6 PRUNED. R0 remains the eligible winner:
epoch15, gamma0, seen/PU Top1 **89.6752%/4.5028%**, H1 **8.5751%**;
seen/PU Top5 **98.6079%/27.5797%**, H5 **43.1037%**. TPE never activated because
there are still only two COMPLETE observations. Pruned recipes are not proven
inferior at every later epoch; this window does not exhaust the search.

New runtime: **4h57m07s target-epoch time**, **5h03m11s post-audit wall time**,
35 paired epochs plus one crop epoch (71 target epochs). Trial13 is deliberately
RUNNING in SQLite with crop3/disease2, paired2; next target is disease epoch3.
Both full optimizer/scaler/RNG/loader checkpoints are preserved. Controller30408
and launcher40268 exited positively, all six OS locks are free, and all input,
parent130/historical131 preservation, terminal, logit and checkpoint audits passed.
No new official inference, calibration or refit occurred. Existing verified R0
full-data weights can be reused. Further compute requires explicit authorization.
The completion monitor is removed after reporting; the chat remains available.

[Verified stop report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_report_verified_20261102.md),
[audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_stop_verified_20261102.json),
[validation metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/validation_metrics_verified_20261102.json),
[selected recipe](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/selected_recipe_for_window.json),
[summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/study_summary.json),
[results](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/results.md),
[crop selection weights](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000/crop/best_epoch_15.pt)
and [disease selection weights](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000/disease/best_epoch_15.pt).

**Five-hour continuation launch record, 2 November 2026.**
This continuation carries forward all eight verified gamma-zero observations
(2 COMPLETE, 6 PRUNED) and starts new trial numbers at8. The five-hour deadline
is the stopping rule; there is **no fixed trial-count cap**. The original completed
run and its checkpoints/reports stay preserved. A verified SQLite backup in the
new output folder retains previous recipes, objectives, intermediate curves,
states and sampler numbering. Only verified COMPLETE trials can win.

Controller `optuna_gamma0_continue.py`, fixed
[continuation manifest](configs/optuna_convnext_base_gamma0_continue_20261102.json),
launcher `run_optuna_gamma0_continue.ps1 -Detach`, output
`runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102`.
Hidden launcher PID40268 / controller PID30408 passed input audits and CUDA
preflight. The five-hour window began **2 November
08:27:11 MYT** with deadline **13:27:11 MYT** (finish/checkpoint the active target
epoch at the cap). Gamma0, the original search space, seeds,
training schedule and median-pruner settings are unchanged. Prior completed
curves inform pruning immediately; TPE still requires three COMPLETE observations
(two inherited). Twelve continuation/controller checks passed before launch.
Six OS locks protect the new run, its completed parent and the other preserved
studies. No official inference, calibration or full-data refit is included.
[Live status](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/status.json).
[Summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/study_summary.json),
[results](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/results.md),
[history copy verification](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/history_import_verified.json)
and [launch verification](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/launch_verified.json).
Summaries refresh at terminal outcomes; status/logs show the current active epoch.

**Gamma-zero Optuna selection completed and verified, 1 November 2026 20:54 MYT.**
All eight fresh paired trials are terminal: **2 COMPLETE, 6 PRUNED, none pending**.
Fresh R0 wins at **epoch15 / gamma0**: seen validation Top1 **89.68%**,
pseudo-unseen **4.50%**, H1 **8.5751%**; H5 **43.1037%**. The best completed
sampled candidate is trial1, epoch2, H1 **6.8246%**. No sampled recipe improved R0.
Only two trials completed, so the three-COMPLETE threshold for TPE was never
reached; the sampled recipes used the verified deterministic startup streams.

Fresh selection used **10h11m20s target-epoch time** (including validation) across
144 target epochs, and **10h22m38s post-audit wall time**. These exclude the earlier
calibrated window, original R0 selection and all earlier full-data refits. The
authorized ten-hour extension was safely applied once; all eight outcomes became
terminal well before its final deadline. Controllers/launchers/supervisor exited,
all study/supervisor locks are free, and all retained model/logit/terminal hashes,
source14, pretrained, data/cache/code and 131 preservation hashes passed.

Existing verified full-data R0 weights and separately labeled gamma-zero metrics
match this selected recipe and can be reused. Historical calibrated outputs/app
gamma metadata remain preserved. No new full-data refit or official inference
was launched by this search. The completion monitor is removed after reporting.
See [verified report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selection_report_verified_20261101.md),
[controller results](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/results.md),
[study summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/study_summary.json),
[selected recipe](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selected_recipe.json),
[validation metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/validation_metrics_verified_20261101.json),
[completion audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selection_completion_audit_20261101.json),
[crop selection weights](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000/crop/best_epoch_15.pt)
and [disease selection weights](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000/disease/best_epoch_15.pt).
The corrected controller is `optuna_gamma0.py`, with fixed manifest
[configs/optuna_convnext_base_gamma0_seed42.json](configs/optuna_convnext_base_gamma0_seed42.json)
and launcher `run_optuna_gamma0.ps1 -Detach -Hours 10`. It trains a fresh R0 pair
first, then sampled recipes, with at most eight fresh pairs total. Gamma is zero
for checkpoint/epoch selection, early stopping, pruning, TPE and ranking. No
historical calibrated observations are imported; all older runs stay preserved.
The run folder is `runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42`.
The clock starts after audits/CUDA preflight. Recovery uses `-Resume -DeadlineUtc`
with the exact saved deadline and never silently opens another window. Official
test inference and final full-data refitting remain separate later work.

**Tuning objective corrected on 1 November 2026: gamma zero throughout.**
Trial ranking, checkpoint selection, early stopping, pruning and TPE observations
must use uncalibrated validation composition Top-1 H. The preceding run optimized
calibrated H; its 62.95% result is historical and does not establish a tuning
improvement under the corrected objective. At retained checkpoints, gamma-zero H
is R0 8.58%, trial 1 6.51% and trial 2 5.46%. Some other epochs' gamma-zero peaks
exist only in logs because their model pairs were not retained.
The historical database/checkpoints remain intact. The old launcher is blocked;
the new authorization and separate gamma-zero controller are described above.
See the [corrected plan](OPTUNA_CONVNEXT_BASE_PLAN.md)
and [objective review](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/uncalibrated_objective_review.md).

**Historical calibrated window stopped on 1 November 2026, 06:14 MYT.**
Two new trials completed, three were pruned, and trial 6 is checkpointed after
crop/disease epoch 1; two configurations have not started. The best completed
candidate so far is trial 2, epoch 30 / gamma 6.4: held-out validation Top-1 H
**62.95%**, versus R0 **59.74%**. Gamma-zero validation H is **5.46%**, versus
R0 **8.58%**. These are training-holdout results; no new official evaluation or
full-data refit occurred. Controller exit, locks and checkpoints are verified.
Further training requires an explicitly approved additional compute budget.
See [window results and checkpoint links](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/results.md),
[validation metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/window_validation_metrics.json)
and [stop audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/window_stop_verified_20261101.json).

**Historical Optuna authorization, superseded by the objective correction:**
[ConvNeXt-Base IN22K -> IN1K](OPTUNA_CONVNEXT_BASE_PLAN.md), validation selection
only, at most ten hours / eight new paired trials. The existing R0 is a verified
incumbent. Full-data refitting and official test inference are separate later work.
The fixed manifest is [configs/optuna_convnext_base_seed42.json](configs/optuna_convnext_base_seed42.json).
The old `run_optuna.ps1` launcher now rejects this superseded objective; `status.json` in
`runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42` records progress.
The initial ten-hour window has expired; this launcher does not grant another
budget. The SQLite study and active full recovery checkpoints survive pauses. Recovery
uses `-Resume -DeadlineUtc '<saved deadline>'`, after confirming exit/lock release.
Creating `STOP.request` in that run folder pauses after the active target epoch;
only explicit user resume may add `-ClearStop`. The controller never extends the
saved selection budget or automatically starts final refitting.

**Tiny follow-up completed and verified on 30 October 2026 at 01:27 MYT.**
The original Tiny recipe was freshly refitted on all 8,000 images for 17 crop /
16 disease epochs, gamma zero. Official Top-1 H is **19.85%** (original 21.16%);
Top-5 H is **64.06%** (original 58.08%). Training took **1h 36m 13s** and
post-audit finalization **1h 42m 10s**. The completion monitor is stopped.
See [Tiny results and checkpoint links](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/results.md),
[verification](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/final_review_verified.json)
and [TINY_REFIT_PLAN.md](TINY_REFIT_PLAN.md). The completed R0 results below remain preserved.

**Completed and verified on 29 October 2026 at 04:06 MYT.**
All six fresh R0 crop/disease refits finished on all 8,000 training images,
followed by official evaluation on 2,000 seen and 1,448 unseen images against
all 333 compositions. ConvNeXt-Base achieved calibrated Top-1 H **71.44%**
(seen **67.00%**, unseen **76.52%**); gamma-zero H is **23.56%**.

See the [verified final results](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/results.md)
for all three models, gamma-zero/calibrated Top-1/Top-5, correctly labeled factor
accuracies, validation choices, runtime, overfitting discussion and checkpoint links.
The [independent artifact audit](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/completion_audit.json)
checks hashes, finite logits/metrics, prediction counts, confusion matrices and app parity.
Final target training took **3h 31m 02s**;
finalization took **3h 57m 23s** after startup auditing.
The controller has exited and the completion monitor is stopped.

The completed scope and frozen source choices are recorded in
[REFIT_ONLY_PLAN.md](REFIT_ONLY_PLAN.md) and
[configs/refit_r0_seed42.json](configs/refit_r0_seed42.json).

The original full tuning study below is retained for history in
[PLAN.md](PLAN.md) and [configs/study_seed42.json](configs/study_seed42.json).
Its unfinished BiT R1 run and all completed checkpoints remain preserved.

## Backbones

| ID | timm pretrained model | Normalization | Interpolation |
|---|---|---|---|
| bit50 | `resnetv2_50x1_bit.goog_in21k_ft_in1k` | mean/std 0.5 | bilinear |
| convnext_base | `convnext_base.fb_in22k_ft_in1k` | ImageNet mean/std | bicubic |
| efficientnetv2_l | `tf_efficientnetv2_l.in21k_ft_in1k` | mean/std 0.5 | bicubic |

BiT is the ResNet-v2 variant with GroupNorm and weight standardization. Each
backbone has two independent networks: 55 crop classes and 175 disease classes.
Composition scores are `log P(crop) + log P(disease)` over all 333 pairs.
Composition Top-5 requires the complete correct pair in the five highest ranked
pairs. It is different from checking the two factor Top-5 lists independently.

The shared evaluator ranks by descending score and resolves exact ties by the
lower composition ID. Metrics, prediction CSVs, confusion matrices and app
rankings use the same rule. The completed baseline results were rechecked from
saved scores after this correction: all reported metrics are unchanged.
[Ranking update audit](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/tie_policy_update_20261029/audit.json)
records the regenerated CSVs and preserved originals.


All models receive 224-pixel images for this comparison; this deliberately differs
from the larger native pretrained evaluation resolutions of BiT and EfficientNet.
Weights are pinned by Hugging Face revision and SHA-256, then loaded locally.

## Original study design (historical)

One saved validation split uses seed 42, with 20 held-out training compositions
as pseudo-unseen validation classes. Official test images never select recipes,
epochs or gamma. The audit saves the exact fitting/validation CSVs and hashes.
Final selected models are freshly refitted on all 8,000 official training images.

The seed-42 split contains 5,798 fitting, 862 seen-validation and 533
pseudo-unseen-validation images. Another 807 images from the held-out pairs are
excluded during selection by the fixed 40-image-per-pair validation cap; they
return for the final full-data refit. `selection_excluded.csv` records them.

Common settings: AdamW, effective batch 32, physical batch starting at 4,
30 maximum epochs, 3 warmup epochs, cosine decay to 1%, smoothing 0.1,
clipping 1 and early-stopping patience 8 on paired validation Top-1 harmonic mean.
The head LR is ten times the backbone LR. Biases and normalization parameters
have no weight decay. EfficientNet BatchNorm running statistics are frozen;
affine parameters remain trainable. DropPath is 0 for BiT and 0.1 for the others.
CUDA FP16 AMP is required for training. Preflight may reduce physical batch to
2 or 1 for memory, preserving effective batch 32, then freezes that choice.

| Round | Change |
|---|---|
| R0 | LR 5e-5 / 5e-4, WD 0.05, head dropout 0.2 |
| R1 | LR 2e-5 / 2e-4 |
| R2 | LR 1e-4 / 1e-3 |
| R3 | Selected LR, WD 0.01 |
| R4 | Selected LR/WD base, head dropout 0 |
| R5 | Same selected LR/WD base, head dropout 0.4 |

Each round runs across all three backbones before the next round. R4 and R5
branch separately from the same base. Select highest validation composition
Top-1 harmonic mean; exact ties retain the earlier recipe and epoch. Gamma is
selected on validation from 0 through 9.9 in steps of 0.1, with smallest gamma
retained on ties. Real unseen candidates are excluded during validation.

There are **18 recipe pairs (36 target trainings), then 6 final target refits**.
This complete study can require several overnight windows. An unfinished queue
does not establish a winning model.

## Original full-study commands (historical)

From the repository root in PowerShell:

```powershell
# Download/pin weights, check CUDA memory, audit originals and build image cache.
.\independent_cnn_baseline\run_overnight.ps1 -Prepare

# Start the first ten-hour window in a hidden process.
.\independent_cnn_baseline\run_overnight.ps1 -Detach -Hours 10

# After confirming the previous controller exited, continue the same study.
.\independent_cnn_baseline\run_overnight.ps1 -Detach -Resume -Hours 10
```

The launcher finds the repository `.venv` or the existing sibling
`Plant_Disease/.venv`; `-Python <absolute python.exe>` overrides that discovery.
The data root is configured as `../Plant_Disease`. Training never falls back to
CPU. Preparation runs do not fit on the dataset. The ten-hour training timer
starts after preparation/auditing and finishes the active target epoch before
checkpointing and exiting, so the exit can be slightly later than ten hours.
The controller keeps Windows awake only while it runs.

For recovery within an interrupted window, use `-Resume -DeadlineUtc <UTC ISO
timestamp>` with that window's original deadline. The runner verifies this
against its saved start and duration and rejects deadline extensions. A later
user-authorized overnight window uses the normal `-Resume -Hours 10` command.

An OS lock rejects duplicate controllers. Interrupted partial epochs repeat from
the previous completed epoch, preserving optimizer, scaler, random state and
loader generator. Completed trials are skipped after verifying checkpoint hashes.
The LR schedule keeps its original 30-epoch horizon during final refitting.

Unfinished jobs keep full `last.pt` recovery states. Completed selection jobs keep
paired `best.pt` weights, validation logits, histories and learning curves; their
large optimizer recovery states are removed only after verified completion.
Existing group outputs and the earlier ConvNeXt-Tiny experiments are preserved.

## Outputs and app integration

Study status, log, audit and trials live in:

`runs/baseline/independent_cnn_baseline_study_seed42/`

Use `status.json` and `training.log` for progress. `runner.pid` identifies the
Python controller, and `launch.json` records the hidden launch and output logs.
`prepared_models.json` records model revisions, weight hashes, physical batches,
CUDA memory and synthetic throughput. `study_config.json` snapshots the recipe.

Final app-compatible runs are `runs/baseline/independent_cnn_baseline_<id>_seed42/`.
Each contains `crop_net.pt`, `disease_net.pt`, `metrics.json`,
`uncalibrated_metrics.json`, `factor_logits.npz`, `scores.npz`, prediction CSVs,
the seen confusion matrix and `completion_verified.json`. The study summary
and `results.md` appear after all final evaluations pass verification.

Report all-333-candidate composition results at both gamma zero and the
validation-selected gamma. The separately labeled unseen-only diagnostic is
restricted to 10 candidates and is not generalized accuracy. Shared factor
accuracy describes the winning composition's factors; `raw_factor_metrics`
describes each independent head before composition constraints/calibration.
Validation accuracy refers to the saved training holdout, not official tests.

The shared app's model loader routes only configurations marked
`implementation: independent_cnn_baseline` to this package. Live inference
reproduces the resized JPEG cache and model-specific transforms, with saved-score
parity checked before official results are marked complete. Other model loaders
and group training recipes keep their existing behavior.

## Verification

```powershell
python -B -m unittest discover -s independent_cnn_baseline/tests -v
```

Generated-image tests cover exact checkpoint recovery, interrupted paired epochs,
partial gradient accumulation, gamma search against the shared evaluator,
Windows worker loading, the OS lock and the complete adaptive queue with test
evaluation occurring only after selection is frozen. CUDA preflight additionally
loads each actual pinned backbone and executes forward/backward/optimizer steps.
