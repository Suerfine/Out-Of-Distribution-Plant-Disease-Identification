# Independent CNN Baseline

Hein's independent baseline package for the group project.

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
