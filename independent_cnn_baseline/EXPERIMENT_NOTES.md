# Independent CNN Baseline experiment notes

## Setup verified, 28 October 2026

The approved study is implemented in this folder. Nine generated-image tests pass,
including exact interrupted-epoch recovery, paired checkpoint recovery, partial
accumulation, validation calibration, the adaptive 18-pair queue, Windows workers,
app score parity and routing of existing group models. `pip check` passes.

Runtime: Python 3.12.10, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128,
timm 1.0.30, CUDA 12.8, RTX 5060 Laptop GPU (8 GiB).

Each pinned pretrained model passed real CUDA forward/backward/AdamW checks at
physical batch 4; accumulation 8 preserves effective batch 32.

| Backbone | Peak allocated CUDA memory | Synthetic optimizer step median |
|---|---:|---:|
| BiT ResNet-v2-50x1 | 0.73 GiB | 0.094 s |
| ConvNeXt-Base | 1.70 GiB | 0.156 s |
| EfficientNetV2-L | 2.43 GiB | 0.406 s |

These synthetic timings include an optimizer update every batch; they are not
dataset epoch times or a full-study completion estimate. AMP scaling can skip
initial overflowed updates while adjusting its scale; successful updates and
finite weights were verified for all three models.

All 11,448 originals passed decoding and exact-content leakage checks. Saved
seed-42 selection split: fitting 5,798; seen validation 862; pseudo-unseen
validation 533 across 20 held-out pairs. The fixed 40-image-per-held-out-pair cap
excludes another 807 training images during selection. All 8,000 training images
are used for final refitting. Official seen/unseen tests remain 2,000/1,448 images.

Pinned revisions, SHA-256 hashes, native preprocessing, CUDA checks, environment
and data provenance are saved under
`runs/baseline/independent_cnn_baseline_study_seed42/`.
`verification_setup.json` records setup checks and existing group-source hashes.

The first ten-hour window starts after preparation/auditing. `status.json` and
`training.log` record actual execution. The complete study is 36 selection target
trainings plus six final target refits, and may require more than one window.
No test result or selected winner is available at setup time.

## First window running

Started 28 October 2026 at 01:09:12 MYT (27 October 17:09:12 UTC),
with controller PID 4356 and hidden launcher PID 33036. The active first
trial is BiT ResNet-v2-50x1 R0, crop epoch 1. The ten-hour deadline is
28 October 11:09:12 MYT; the current target epoch may finish after that time.
Recovery uses the existing launcher with `-Detach -Resume -Hours 10` after
confirming this controller has exited. Another window requires user instruction.

The `independent-cnn-baseline-overnight` monitor checks every ten minutes and
stays quiet during healthy progress. It reports a checkpointed deadline stop,
verified completion or an actionable failure, then pauses when the window ends.

## Calibration tie recovery, 28 October 2026

At 06:23:44 MYT the controller stopped on its calibration consistency check,
after both EfficientNetV2-L R0 target checkpoints had saved epoch 2. One
pseudo-unseen validation image had exactly tied composition scores for pairs
307 and 309. The optimized `argmax` calibration path and the shared evaluator's
`argpartition` Top-1 ranker selected different members of that tie.

The owned calibration function now calls the shared evaluator's calibration and
Top-1 ranking functions directly for every gamma and each validation split.
The shared group evaluator, master recipe, weights, split and hyperparameters
were not changed. All 11 tests pass, including new 333-class tie and fixed-window
recovery checks. Saved EfficientNet epoch-2 validation agrees exactly with the
shared evaluator (H 53.4073%, gamma 6.6); these are validation results.

The two completed R0 selections were rechecked over the full gamma grid at
their saved best epochs: BiT epoch 3 and ConvNeXt-Base epoch 15 retain their
selected gamma/H, and all four best-weight SHA-256 hashes match their completion
records. Original error status, source provenance and the tied validation scores
are retained in `recovery_20261027_gamma_tie/` under the study output.

Both old process IDs were absent, the failed status/log had stopped changing,
and the OS lock was positively acquired and released before recovery. Resuming
with `-Resume -DeadlineUtc 2026-10-28T03:09:12.367334+00:00` preserves the first
window's original 11:09:12 MYT deadline instead of starting another ten hours.

Recovery was verified at 06:35 MYT: controller PID 29952 (launcher 30356) kept
both completed trials, processed the saved epoch-2 validation and continued at
EfficientNetV2-L crop epoch 3. `status.json` records the unchanged original
window start and deadline. No completed target was retrained.

## Paused at the user's request, 28 October 2026 morning

The controller and its data-loader workers were stopped, process exit was
verified and the study OS lock was positively acquired/released. The heartbeat
monitor is PAUSED. No automatic restart is scheduled.

Three initial recipe pairs are complete: six of 36 selection target trainings.
Their best calibrated composition Top-1 validation results are:

| Initial recipe | Selected epoch | Seen validation | Pseudo-unseen validation | Harmonic mean |
|---|---:|---:|---:|---:|
| BiT ResNet-v2-50x1 R0 | 3 | 59.98% | 37.71% | 46.31% |
| ConvNeXt-Base R0 | 15 | 83.41% | 46.53% | 59.74% |
| EfficientNetV2-L R0 | 3 | 61.83% | 47.65% | 53.83% |

These are held-out validation results from the initial recipes, not official
test accuracy or final selected winners. All six final full-data refits and
official test evaluations remain pending.

The fourth pair, lower-LR BiT R1, has three completed paired epochs. Its verified
full recovery checkpoints save crop epoch 4 and disease epoch 3, including model,
optimizer, scaler and RNG state. The partial disease epoch 4 will repeat on
resume; completed targets/recipes will be preserved. Fifteen recipe pairs remain
unfinished, including this active pair, followed by six final target refits.

`status.json` is marked `paused`; `manual_pause.json` records the verified state
and progress under the study output. To resume a later authorized window from
the repository root:

```powershell
.\independent_cnn_baseline\run_overnight.ps1 -Detach -Resume -Hours 10
```

## Smaller R0-only plan prepared, training still paused

The user chose preparation only for final refits/evaluation of the three completed
R0 recipes. See `REFIT_ONLY_PLAN.md` and `configs/refit_r0_seed42.json`.
The frozen target epoch counts remain 3/15/3 for crop and disease, with gammas
6.6/6.2/6.6. Remaining work is six target refits and 42 target epochs, with no
additional LR/WD/dropout trials. The unfinished BiT R1 files are preserved but
are excluded from this plan's selections.

The new `finalize.py` entry point verifies pinned R0 completion/checkpoint hashes,
the original pretrained weights and exact saved data/partition identity. It
freezes all three sources before refitting, finishes all six refits before test
evaluation, and supports checkpointed cooperative pause requests. Fifteen tests
pass, including source-tamper rejection, exclusion of partial R1, preparation
without fitting, final-refit-before-test ordering and user-stop handling.

The original tuning study and monitor remain paused. Starting the smaller plan
later uses `run_overnight.ps1 -RefitOnly -Detach`; the original full-sweep resume
command above is retained as historical context.

## Final refits started, 29 October 2026

The user explicitly authorized starting the prepared refit-only run. Both study
locks were available, all six source R0 checkpoint hashes and prepared Python
source hashes matched, and 19.5 GiB of disk headroom was available before launch.
The complete data/partition audit and frozen source checks passed again.

Hidden launcher PID 23328 started controller PID 5328 with
`run_overnight.ps1 -RefitOnly -Detach`. The training window began at
29 October 00:09:03 MYT (28 October 16:09:03 UTC). The invocation cap is
29 October 10:09:03 MYT; the expected total work remains approximately 4–6 hours.
BiT crop epoch 1 was confirmed active on CUDA. The finalizer will perform only
the six frozen refits (3/15/3 epochs per target), then official evaluation.

The existing `independent-cnn-baseline-overnight` monitor was updated to the
refit-only scope and re-enabled as "Independent CNN baseline final refits".
It stays quiet during healthy progress and reports verified completion,
actionable failure, a checkpointed window stop or required input. The original
tuning study and unfinished BiT R1 checkpoints remain preserved and paused.

## Verified final R0 results, 29 October 2026

All six fresh full-data refits and all three official evaluations completed at
04:06:27 MYT. The original full sweep remains paused. Source R0 checkpoints and
partial BiT R1 recovery files are preserved. No further hyperparameter trials ran.
The frozen per-target epochs were BiT 3, ConvNeXt-Base 15, EfficientNetV2-L 3;
validation-selected gammas were 6.6, 6.2, 6.6. All refits used the full 8,000-image
training split and completed before official test inference.

The tables below are official per-image composition accuracies on 2,000 seen
and 1,448 unseen images, with all 333 candidates. They are separate from source
held-out validation results. No epochs or gamma were chosen from official tests.

### Frozen calibrated official test

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| BiT ResNet-v2-50x1 | 51.05% | 57.53% | 54.10% | 82.00% | 93.51% | 87.38% |
| ConvNeXt-Base | 67.00% | 76.52% | 71.44% | 88.35% | 98.76% | 93.26% |
| EfficientNetV2-L | 51.15% | 78.38% | 61.90% | 85.45% | 97.31% | 90.99% |

### Gamma-zero official test

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| BiT ResNet-v2-50x1 | 63.00% | 5.59% | 10.28% | 86.90% | 32.60% | 47.41% |
| ConvNeXt-Base | 73.80% | 14.02% | 23.56% | 91.55% | 47.44% | 62.50% |
| EfficientNetV2-L | 65.70% | 8.84% | 15.58% | 89.80% | 41.37% | 56.64% |

### Raw independent factor-head accuracy

| Model | Seen crop head | Seen disease head | Unseen crop head | Unseen disease head |
|---|---:|---:|---:|---:|
| BiT ResNet-v2-50x1 | 87.90% | 64.35% | 69.89% | 6.01% |
| ConvNeXt-Base | 94.30% | 74.25% | 85.64% | 14.50% |
| EfficientNetV2-L | 90.55% | 66.40% | 74.03% | 12.43% |

These raw head metrics precede valid-pair restriction and gamma. The JSON
`crop_top1`/`disease_top1` fields in composition evaluation instead use the factor
labels of its winning pair; both definitions are tabulated in the final report.
Restricted unseen-only diagnostics use 10 candidates and are separately labeled
there (Top-1: 76.73%, 91.44%, 83.22% respectively).

The copied source selection learning curves show substantial disease overfitting:
last-epoch fitting reaches 97.03%/99.91%/98.69% while held-out disease accuracy is
54.48%/56.92%/55.20%. Validation H selected the frozen earlier 3/15/3 epochs.
There is no validation loop during the final full-data refits.

Actual final target epoch training: 3h 31m 02s
(12662.328s); timed finalization after startup auditing:
3h 57m 23s (14243.389s).
Earlier R0 source selection target epochs totaled 8h 01m 25s,
reported separately and excluding partial R1 and nontraining overhead.

Independent artifact verification passed: six final targets/42 epochs; all source,
pretrained, split/config and final checkpoint hashes; finite raw tensors, scores,
logits and metrics; reconstructed score arrays; prediction contents and counts;
recomputed gamma-zero/calibrated metrics and H; and each 323x333 seen confusion
matrix totaling 2,000. All three `completion_verified.json` markers are present.
App inference score parity passed all 24 sample comparisons with zero difference.
Both controller PID 5328 and launcher PID 23328 are absent; both OS locks were
successfully acquired and released. The stale status preparation flag for official
evaluation was corrected to true after verifying the complete artifacts.

[Full final results and checkpoint links](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/results.md)
and [independent audit](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/completion_audit.json)
record the evidence. The completion monitor is stopped after this report; no new
training window is scheduled. The chat remains open.

## Shared evaluator tie consistency corrected, 29 October 2026

At the user's request, composition ranking now uses descending score followed
by ascending composition ID for exact ties. The evaluator's Top-1/Top-5 hits,
factor diagnostics, prediction tables, confusion matrices, live app ranker and
saved-score app view share this rule. Five new regression tests cover 333-class
exact ties, Top-5 cutoff ties, calibration before ranking, hidden candidates and
agreement across metrics/CSV/confusion/app. All five pass; the 15 existing
baseline selection/refit/recovery/inference tests also pass.

Recomputed every official metric for all three completed baselines at gamma zero
and the frozen calibrated gamma. All metrics, including per-class means, factor
diagnostics, unseen-only diagnostics and harmonic means, are exactly unchanged.
The three confusion matrices are unchanged. Six prediction CSVs were refreshed:
11 rows reorder equal-score Top-5 predictions, including one EfficientNet seen
Top-1 label that is incorrect under both tie choices. All correctness flags are
unchanged. Checkpoints, raw scores, factor logits, frozen epochs/gammas and source
validation records were not changed, and no training or test-driven selection ran.

Original prediction CSVs and completion markers are preserved under
`runs/baseline/independent_cnn_baseline_r0_finalization_seed42/tie_policy_update_20261029/`.
Its `audit.json` records changed rows, old/new hashes and unchanged metrics.
Final completion markers now record the tie policy and updated CSV hashes.
Future evaluations also record the tie policy in their completion markers.
Historical selection records retain their original evaluator provenance.
