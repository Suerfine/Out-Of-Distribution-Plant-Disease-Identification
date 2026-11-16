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

## Original Tiny full-data refit authorized, 29 October 2026

The user requested a fresh full-data Tiny refit and another official evaluation.
The frozen source is the original Tiny recipe, crop epoch 17 / seed 42 and disease
epoch 16 / seed 43, not the later tuned LR recipe. Gamma remains zero. Both networks
restore all 8,000 training images and retain the original 30-epoch LR horizon and
all original preprocessing, augmentation, optimization and batch settings.

The new runner uses pinned legacy Tiny helpers, verifies all original metadata
and image hashes, preserves originals, supports epoch-boundary checkpoint recovery
and user pause, and performs official inference only after both target refits.
A new app adapter retains the original RGB/native-timm preprocessing path.
The fixed manifest and exact plan are in `configs/tiny_full_refit_seed42.json` and
`TINY_REFIT_PLAN.md`. New outputs are isolated under
`runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42`.

Preparation passed all 11,448 original image hashes and the source/pretrained
artifact hashes. Actual batch-8 CUDA forward/backward and native inference-path
parity checks passed. Five new Tiny tests cover source/epoch tampering, full-data
fixed refits, exact RNG/checkpoint recovery, both-refits-before-test ordering and
app score parity; the existing baseline tests also pass.

The hidden launcher PID 17052 started actual controller PID 6412.
The post-audit window began 29 October 2026 23:45:08 MYT. Crop epoch 1
is active on CUDA, and the first 250 of 1,000 training batches were observed.
The completion monitor `convnext-tiny-full-data-refit` is active and stays quiet
during healthy progress. The ten-hour deadline is recorded in status.json.

## Tiny full-data refit completed and independently verified, 30 October 2026

The original Tiny crop 17 / seed 42 and disease 16 / seed 43 refits finished all 33
fixed target epochs on all 8,000 training images. Gamma remained zero and the
original 30-epoch LR horizon, initialization, native preprocessing, augmentation
and optimizer settings were preserved. Both targets finished before official
inference. No test-based epoch/gamma/recipe change followed the evaluation.

Official tests: 2,000 seen / 1,448 unseen images, all 333 composition candidates.

| Run, gamma zero | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Original Tiny, 6,746 fitting images | 71.05% | 12.43% | 21.16% | 90.25% | 42.82% | 58.08% |
| Tiny full-data refit, 8,000 images | 71.20% | 11.53% | 19.85% | 90.25% | 49.65% | 64.06% |
| Base full-data refit, 8,000 images | 73.80% | 14.02% | 23.56% | 91.55% | 47.44% | 62.50% |

Compared with original Tiny, Top-1 H fell 1.31 points while Top-5 H
rose 5.98 points. This single-seed result does not show a Top-1 unseen
benefit from full-data refitting. Base still has higher Top-1 H, while Tiny now
has higher Top-5 H; different initialization/augmentation/selection protocols
prevent an architecture-only conclusion. Original Tiny metrics were rechecked
from its saved logits using the corrected shared evaluator and agree within
floating-point rounding.

Raw independent factor-head accuracy:

| Independent head | Seen Top-1 | Seen Top-5 | Unseen Top-1 | Unseen Top-5 |
|---|---:|---:|---:|---:|
| Crop | 92.80% | 98.10% | 80.59% | 92.82% |
| Disease | 71.80% | 88.55% | 13.88% | 32.73% |

The separate ten-unseen-candidate diagnostic is Top-1
88.81% / Top-5 98.00%.
It does not replace all-333 unseen results. Winning-composition factor accuracies
are separately defined in the final report.

Source selection curves show crop fitting 100% vs validation 96.65%, and disease
fitting 99.97% vs validation 84.93% at the selected epochs. Disease validation CE
is lowest at epoch 5 (.7460) and rises to .8135 at selected epoch 16, indicating
source-run overfitting. New full-data training reaches crop 100% / disease 99.98%;
it has no held-out validation loop and does not supply a new validation estimate.

Actual new target epoch training: 1h 36m 13s;
post-audit finalization: 1h 42m 10s; launch through
completion: 1h 42m 48s. Completion was
30 October 01:27:18 MYT. Prior original Tiny 4h 01m 14s and tuning 9h 25m 44s are separate.

Independent verification passed 19 source artifact hashes, legacy/pretrained/code
identities, final raw weights and recovery-state equality, frozen epochs/seeds
and LR trajectory, all finite logits/scores/metrics, reconstructed scores,
recomputed metrics/H, ordered prediction rows 2,000/1,448 and the entire 323x333 seen
confusion matrix totaling 2,000. All 8 app parity checks passed; maximum score error
9.54e-07. All 36 hashed newer baseline artifacts remain
intact. Controller 6412 and launcher 17052 have exited; the OS lock is free.

[Full results and checkpoint links](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/results.md)
and [independent audit](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/final_review_verified.json)
record the evidence. The completion monitor is stopped after this final report;
no new window or additional experiment is scheduled. The chat remains open.

## ConvNeXt-Base Optuna plan prepared, 30 October 2026

The user selected ConvNeXt-Base IN22K -> IN1K and explicitly requested a plan
only. `OPTUNA_CONVNEXT_BASE_PLAN.md` proposes a bounded validation-only search
on backbone LR, weight decay and head dropout, with head LR ratio and all other
controls fixed. Primary objective remains calibrated held-out composition H.

The original saved seed-42 partition stays fixed: 5,798 fitting / 862 seen
validation / 533 pseudo-unseen validation, with 807 cap-excluded images restored
only during final full-data refitting. Base R0 epoch-15 validation logits were
re-evaluated read-only with the corrected evaluator: gamma 6.2 and H 59.7357%
are unchanged; gamma-zero H is 8.5751%. No official inference was performed.

The proposed ten-hour/eight-new-trial selection ceiling is not an approved
execution budget. Recorded Base timing is about 8m 35s per paired selection
epoch; a full 30-paired-epoch trial can require about 4h 18m. Final full-data
refits and test verification need their own budget. The plan explicitly avoids
promising a large number of trials or a guaranteed performance gain.

Optuna is not installed in the CUDA venv. No package was installed, no runner
was implemented or launched, no new study database/configuration was created,
and no completion monitor was scheduled. All completed studies remain intact.

## ConvNeXt-Base Optuna execution authorized, 31 October 2026

The user authorized starting the prepared Base IN22K -> IN1K search. The first
selection window is bounded by ten hours after auditing and at most eight new
paired trials. Full-data refits and official test inference remain separate later
work; the new controller cannot invoke either. The original study stays paused,
and all existing completed runs and partial BiT R1 state remain intact.

The fixed manifest `configs/optuna_convnext_base_seed42.json` pins 14 source
artifacts and the complete R0 pipeline. Search dimensions are backbone LR
1e-5..1.5e-4 (log), weight decay .005...10 (log), and head dropout
0/.1/.2/.3/.4. Head LR ratio 10, DropPath .1, 224 input, augmentation, batch 4 /
accumulation 8, seeds 42/42, three-epoch warmup, 30-epoch cosine horizon and
patience eight remain fixed. R0 epoch 15/gamma 6.2/calibrated validation H
59.7357% is the completed incumbent, without importing its old intermediate
curve. Gamma-zero validation H is a secondary diagnostic, not the objective.

Optuna 5.0.0 is installed and pinned separately. SQLite retains trial identity
through process exits. Independent TPE uses stable per-parameter seed streams
derived from master seed 42, trial number and parameter index so a partial
parameter commit can be replayed exactly. Three COMPLETE observations precede
TPE suggestions; median pruning needs two completed observations, starts at six
paired epochs and checks every three. Only verified COMPLETE trials can replace
the incumbent; pauses, failed or pruned trials are ineligible.

Seven generated-data Optuna tests pass: partial SQLite suggestion recovery,
trial cap and identity, preserved source controls/LR horizon, user pause/deadline,
exact paired recovery, recovery during best-pair commit, pruning/eligibility/ties,
and controller isolation from refitting/official inference. The existing twenty
baseline tests and five shared ranking tests pass. Windows SQLite connections
use NullPool so completed transactions release their file handles promptly.

The startup audit checks the exact saved partition, original/pretrained/source
hashes and cached JPEG preprocessing before any DPD fitting. The search acquires
its own lock and the original-study, R0-finalization and Tiny-refit OS locks.
The clock starts after auditing/CUDA preflight; a window stop preserves active
checkpoints and never silently creates another budget. `STOP.request` asks for
a safe user pause; only explicit user resume can clear it.

Preparation passed all source/initialization hashes, all 8,000 training originals
and the 7,193 selected cached images. The current environment reports RTX 5060
Laptop GPU, torch 2.10.0+cu128 and timm 1.0.30. Actual physical-batch-4 CUDA
preflight produced three finite optimizer updates, peak allocated memory
1,831,088,640 bytes and median synthetic step .109 seconds. Historical timing
estimates are not guarantees on this current machine.

The hidden launcher PID 41156 started actual Python controller PID 42220.
Selection began **31 October 2026 20:12:38 MYT**, with saved deadline
**1 November 2026 06:12:38 MYT** (UTC 2026-10-31T22:12:38.316162+00:00).
Trial 1 began crop epoch 1 on CUDA: backbone LR 3.39353532372713e-5,
head LR 3.39353532372713e-4, WD .03420375726781103, dropout .3.
All four controller locks are held, with one controller identity; all 14 source
artifact hashes were independently rechecked unchanged after launch. The
completion monitor `convnext-base-optuna-selection-completion` is active and
quiet during healthy progress. The original study's user pause is preserved.

## Optuna ten-hour window stopped and independently verified, 1 November 2026

The saved deadline was 06:12:38 MYT; the active disease target epoch finished,
checkpointed and the controller stopped at **06:14:13 MYT**. This is a capped,
incomplete selection search, not study completion. Controller PID 42220 and
launcher PID 41156 were positively verified exited through OS process evidence;
all four OS locks were acquired and released. No process was restarted.

New trials: 1 COMPLETE after 24 paired epochs (best 16); 2 COMPLETE at the
30-paired-epoch ceiling (best 30); 3, 4 and 5 PRUNED at paired epoch 6 by the
planned median rule. Trial 6 is RUNNING only as a recovery identity: crop and
disease epoch 1 are both saved, paired validation/reporting is committed, and
the next work is crop epoch 2. Its full optimizer/scaler/RNG/loader states and
finite model weights match the saved best pair. Trials 7 and 8 are unstarted.
Pruned/interrupted trials cannot win; no selected_recipe.json is frozen.

Best completed candidate so far: trial 2, backbone LR 3.499478844186344e-5,
head LR 3.499478844186344e-4, WD .06542664314595986, head dropout .4,
selected paired epoch 30 and validation gamma 6.4. All other frozen controls
match Base R0. Fixed validation candidates remain the 323 training compositions,
with the ten real-unseen candidates hidden.

| Trial 2 held-out validation | Seen Top-1 | Pseudo-unseen Top-1 | H Top-1 | Seen Top-5 | Pseudo-unseen Top-5 | H Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Gamma zero | 90.02% | 2.81% | 5.46% | 98.38% | 30.21% | 46.22% |
| Validation gamma 6.4 | 79.81% | 51.97% | 62.95% | 96.75% | 96.62% | 96.69% |

The calibrated objective improves 3.21 percentage points over R0 validation H
59.74%; gamma-zero H falls from 8.58% to 5.46%. This is not evidence of an
uncalibrated or official-test improvement. The 862/533 held-out training images
are distinct from the official 2000/1448 splits. No new official inference,
calibration on official tests or full-data refit was performed.

Learning curves still show a large raw factor fitting/validation gap at trial 2
epoch 30: crop 100%/96.27%, disease 99.81%/56.92% on the combined validation
partition. Disease validation CE remains approximately flat at 2.62 late in the
run. Held-out compositions contribute distribution difficulty; regularization
has not removed the generalization gap. Do not interpret a single repeated
validation partition as proof of broad generalization.

New training: 146 target epochs (73 paired epochs), **9h 47m 15s** summed
recorded target-epoch time including per-epoch validation. Post-audit window wall
time: **10h 01m 35s**. Prior R0 selection, prior refits and this stop audit are
separate. Startup preparation is outside the window clock.

Independent checks passed all five terminal records and artifact hashes,
recomputed validation gamma/H, SQLite outcomes/intermediates and immutable
sampled recipes, full LR histories, active recovery states, source14 and pinned
pretrained/code identities, 8000 original training image hashes and 7193 cache
hashes. All **49** hashed historical final artifacts remain intact. The original
study is still paused with partial BiT R1 crop4/disease3 preserved.

[Window report and selection checkpoint links](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/results.md),
[validation metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/window_validation_metrics.json)
and [independent stop audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/window_stop_verified_20261101.json)
record the evidence. The existing `-Resume -DeadlineUtc` command retains the
expired deadline and will not grant more training. `-Resume -Hours 10` is rejected.
Further selection requires explicit approval for additional compute and a
recorded extension mechanism preserving all existing identities/checkpoints.
No additional budget or final refitting is assumed. The completion monitor is
retired after this report; the chat stays open.

## User correction: tuning must use gamma zero, 1 November 2026

The user clarified that calibration must not be used for tuning. The prior plan
and controller optimized calibration-selected validation composition Top-1 H.
That objective drove best-checkpoint retention, early stopping, pruning and TPE
suggestions, so the reported 62.95% cannot be presented as an improvement for
the intended uncalibrated tuning task. This was the wrong objective for the
user's intended comparison.

The corrected objective is held-out validation composition Top-1 H at gamma=0,
using the same fixed 862 seen / 533 pseudo-unseen validation split and 323 training
composition candidates. H0 must govern all adaptive decisions, including epoch
selection and checkpoint retention, not just the final displayed score. No gamma
sweep, tuning offset or optional calibration run is part of the corrected scope.
The model, initialization, fitting data, search dimensions and other training
controls remain as previously prepared.

Recomputed gamma-zero H from all retained checkpoint logits using the shared
evaluator: R0 8.5751%; completed trial 1 6.5079% at calibration-selected epoch16;
completed trial 2 5.4579% at calibration-selected epoch30. These are retrospective
diagnostics at historical checkpoints, not outputs of an H0 search. The logs'
highest H0 for trial 1 is 7.2051% at epoch19 and trial 2 is 6.8212% at epoch2;
those model pairs were not retained under the old calibrated policy. They cannot
be recovered from logs or validation logits. Old stopping/pruning decisions and
TPE history also cannot simply be relabeled.

All original trial values, immutable configs, retained weights, trial6 full
recovery checkpoints, status and historical reports are preserved. The new
`uncalibrated_objective_review.json/.md` records the recomputation and source
hashes. `OBJECTIVE_SUPERSEDED.json` blocks the old PowerShell launcher. The prior
plan is copied to `OPTUNA_CONVNEXT_BASE_CALIBRATED_HISTORY.md`; the active plan
now specifies H0 and a separate future study identity. No new training, official
inference or monitor was launched, and the expired budget was not extended.

The corrected `uncalibrated_objective.py` validation helper passes generated-logit
tests showing a calibration-only gain cannot enter the objective, real unseen
candidates remain hidden, and invalid logits/labels are rejected. Full corrected
controller/manifest integration is still required before any future GPU launch.
This correction supplies no additional compute authorization.

## Fresh gamma-zero Optuna window authorized, 1 November 2026

The user explicitly authorized another ten-hour cycle after the objective
correction. This is a separate gamma-zero study, not recovery/relabeling of the
old calibrated database. Fixed manifest:
`configs/optuna_convnext_base_gamma0_seed42.json`; controller `optuna_gamma0.py`;
launcher `run_optuna_gamma0.ps1`; output
`runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42`.

A fresh R0 pair is queued as trial 0, followed by newly sampled recipes. The
maximum is eight freshly trained pairs total including R0, within ten hours
after startup audit/preflight. No old completed, pruned or interrupted objective
observations are imported. The historical 8.58% R0 checkpoint is a fixed
diagnostic reference only; a winner does not exist in the new study until a new
gamma-zero trial is verified COMPLETE.

Uncalibrated validation composition Top-1 H controls best-checkpoint retention,
epoch selection, patience-eight stopping, paired intermediate reports, median
pruning, TPE and recipe ranking. Every row/config/terminal record declares the
objective and gamma=0. The controller cannot use the legacy gamma sweep or
calibrated selection function. A fresh SQLite namespace and fingerprint prevent
cross-objective recovery. Pruned/interrupted recipes remain ineligible; exact
ties retain earlier trials/epochs. Learning curves label H0/H5 without calibrated
series. Search space, model, initialization, seeds42/42, saved split, augmentation,
batch4/accum8, FP16, and original 30-epoch schedule remain unchanged.

The new input audit checks source14/pretrained/helper hashes, all 8000 originals
and 7193 cache image hashes against the previously verified JPEG pipeline. It
uses only that historical input identity, not past trial scores or sampler
history. Five OS locks cover the new search, original study, R0 finalization,
Tiny refit and old calibrated Optuna run. Official tests and final refits are
excluded. Recovery retains the new saved deadline; user STOP.request is preserved.

Seven new controller tests pass with generated data: a fresh queued R0 and no
incumbent import; mandatory objective identity and frozen source controls; exact
mid-pair checkpoint recovery; pair-commit recovery without duplicate epochs;
gamma-zero pruning/reporting at paired epoch six; eligible winner/tie rules; and
the eight-pair cap with no final refit/official inference. Two objective tests
also pass, including a constructed calibration-only gain that must stay out of
the objective. Full baseline/evaluator regression checks precede GPU launch.

The full suite passed **36 owned baseline tests plus 5 shared evaluator tests**.
Before launch all five locks were acquired/released, historical calibrated
study/recovery hashes and 49 final artifact hashes were verified. New startup
audit passed source14, pretrained/helper code, 8000 original training images and
7193 selected cache hashes. Physical-batch-4 CUDA preflight produced three finite
optimizer updates, peak allocated memory 1,831,088,640 bytes, median synthetic
step .125 seconds. CUDA remains mandatory.

Hidden launcher PID **32572** started actual Python controller PID **37564**.
The new window began **1 November 2026 10:31:50 MYT** and its immutable saved
deadline is **1 November 2026 20:31:50 MYT**
(UTC 2026-11-01T12:31:50.263638+00:00). Fresh trial 0 crop epoch 1 is progressing
on CUDA at R0 LR5e-5/head5e-4, WD.05, dropout.2, gamma0. No old objective values
or sampled trial states are imported. The first summary has best_eligible=null
until a new trial completes; the live status/log are authoritative during
training and study summaries refresh at terminal outcomes and window stops.

Five locks are held by the single current controller. After launch **131**
historical study/final artifact hashes were rechecked unchanged against
`historical_preservation.json`; no calibrated-study file or checkpoint changed.
The monitor `convnext-base-gamma-zero-optuna-completion` is active, quiet during
healthy progress, and will stop/delete after verified completion or a checkpointed
window/pause stop. No subsequent window or final refit is authorized automatically.

## Additional ten hours explicitly authorized, 1 November 2026

The user requested **"extend for 10 more hours"** while controller PID37564 was
healthy (launcher PID32572). The new authorized deadline is
**2 November 2026 06:31:50 MYT** / UTC
**2026-11-01T22:31:50.263638+00:00**, ten hours after the original deadline.
Original start 1 November 10:31:50 MYT is retained: twenty hours total, at most
eight fresh paired trials including R0. Gamma-zero selection and every scientific
control remain fixed; official inference and final refits remain outside scope.

No healthy training process is restarted. The original controller retains its
monotonic deadline until it checkpoints/exits at 1 November 20:31:50 MYT. A separate
standard-library administrative supervisor `extend_gamma0_window.py` waits for
that cap stop, confirms positive process exit, stable files and all five free
study locks, records the authorized budget change once, then invokes the unchanged
launcher with -Detach -Resume and the exact extended deadline. A user pause is
preserved and never cleared. No pinned code/config/identity or scientific artifact
is rewritten. Setup rechecked all eleven pinned code hashes unchanged. The
authorization, supervisor hash and process identities are pinned in
`extension_authorization_20261101.json`; progress is
`extension_status_20261101.json`. The original status is retained before applying
the budget, and `extension_applied_20261101.json` records the administrative audit.

Six extension tests passed, covering exact ten-hour extension, preserved partial
paired checkpoints, idempotence, active/paused/scope/config guards, exclusive OS
locks, access-denied process checks, one resume attempt and exact launcher
arguments. These checks do not train models or access official tests. The existing
quiet completion monitor is updated to continue across the original cap and stop
only after verified completion, pause, actionable failure or the extended cap.

The hidden extension supervisor launched at **1 November 19:39:11 MYT**:
launcher PID35888, actual supervisor PID39524. Its exclusive lock is held and
`extension_status_20261101.json` confirms `waiting_for_original_cap`. Controller
PID37564 remains live and owns its study lock; no training restart has occurred.
The monitor's saved prompt is ACTIVE with the extended deadline and continuation
instructions. The run's monitoring metadata records both supervisor identities.

At the original cap, crop epoch5 of trial7 completed and checkpointed; the
controller stopped at **20:37:17 MYT**. The supervisor positively confirmed both
original processes exited, stable files and all five free OS locks. The budget
was applied once at **20:37:38 MYT**, retaining start and identity and all trial
files. Launcher PID32752 / controller PID41184 resumed with the exact extended
deadline. Re-audit passed and training resumed **20:38:21 MYT** at trial7 disease5
from crop5/disease4/pair4. SQLite remains two COMPLETE, five PRUNED and trial7
RUNNING; no new trial or official inference/refit occurred. The continuation
audit independently verifies original-process exit, all five held study locks,
unchanged identity/config and eleven pinned code hashes. The supervisor may
retain launching_authorized_resume while its PowerShell subprocess stays open;
the healthy new controller is verified and must never be duplicated. See
`extension_continuation_verified_20261101.json` in the run directory.

## Gamma-zero Optuna selection complete and verified, 1 November 2026

Controller PID41184 reached selection_complete at **20:54:28 MYT**. Exactly
eight fresh trial outcomes match SQLite: trials0/1 COMPLETE; trials2..7 PRUNED.
None are pending. All pruning reasons and grace/interval/completed-curve medians
were independently replayed. Only COMPLETE trials are eligible; exact ties retain
earlier trials/epochs. There were zero imported historical observations. Only
two COMPLETE observations were available, so the three-completion TPE threshold
was never met: trials1..7 used the verified master42 per-parameter startup random
streams. This is a bounded Optuna study, with no adaptive TPE phase reached.

Fresh R0 trial0 wins at **paired epoch15 / gamma0** with validation Top1 seen
**89.6752%**, pseudo-unseen **4.5028%**, H1 **8.5751%**; Top5 seen **98.6079%**,
pseudo-unseen **27.5797%**, H5 **43.1037%**. These rank all323 original training
compositions on 862 seen / 533 pseudo-unseen validation images. Best COMPLETE
sampled candidate trial1 selected epoch2: Top1 **79.8144% / 3.5647% / H6.8246%**;
Top5 **96.8677% / 11.2570% / H20.1701%**. No sampled candidate improved fresh R0.
The winner is LR5e-5/head5e-4, WD.05, dropout.2, original DropPath.1, seeds42/42,
original pretrained hash, and unchanged original horizon30 / warmup3 controls.
R0 stopped at epoch23 and trial1 at epoch10 under patience8 on H0.

Raw independent factor Top1 for R0 is seen crop/disease **98.2599%/89.7912%**,
pseudo-unseen **92.1201%/3.7523%**. Candidate1 raw heads are seen
**97.7958%/79.3503%**, pseudo-unseen **90.6191%/4.1276%**. These are kept separate
from factors extracted from the winning composition. The R0 pseudo-unseen-only
20-candidate diagnostic (59.4747% Top1 / 98.3114% Top5) is also kept separate
from the all323 validation metric and official10-unseen diagnostic.

Selected R0/candidate learning curves were visually inspected. R0 epoch15 has
crop fit99.9655% versus combined validation95.9140%, disease fit99.5861% versus
56.9176%; by epoch23 disease fit99.9138% still has validation56.9176%. H0 never
beats epoch15 despite near-perfect fitting. CE largely plateaus with noise.
This large disease fit/validation gap and weak held-out disease accuracy show
limited generalization; H5 or fitting improvements do not establish H1 gains.

The CPU-only completion audit recomputed every retained trial's H0/Top5 from
saved finite logits with direct stable gamma-zero ranking and matched all paired
best epochs. All16 retained selected model states are finite. All96 terminal,
verification and artifact hashes; source14; eleven code hashes; pretrained SHA;
all8000 originals;7193 cache hashes; and131 historical preservation hashes,
including49 earlier final artifacts, passed. The historical calibrated database,
partial BiT R1 and all prior Base/BiT/EfficientNet/Tiny outputs remain intact.
Original/resumed controller/launcher and supervisor processes exited positively.
All five study locks plus supervisor lock were acquired and released. No new
official inference, full-data refit or gamma sweep occurred.

Fresh selection used **36,680.093s (10h11m20s)** target-epoch time including
validation, across **144 target epochs / 72 paired epochs**. Post-audit wall was
**37,357.911111s (10h22m38s)**. Initial segment:36,327.059315s; handover/re-audit:
54.919237s; continuation:975.932559s. Completion left9h37m22s unused in the
authorized twenty-hour ceiling. The earlier calibrated window is separate:
35,234.666s target epochs / 36,095.205822s wall. Earlier original R0 selection and
all earlier refits are excluded from the fresh runtime.

Existing full-data Base R0 weights match all scientific settings and epoch15;
their preserved full8000 refit histories have250 optimizer steps per epoch and
15 epochs per target (prior target time7,753.063s / 2h09m13s). They and existing
separately labeled gamma-zero metrics can be reused without retraining. Historical
calibrated predictions/metrics/app gamma metadata remain preserved. A separate
unlaunched proposal describes reuse or optional isolated15+15 replication
(roughly2.2–3.0h training plus audits/evaluation). No next phase is launched.

Artifacts in the gamma-zero run: `selection_report_verified_20261101.md`,
`selection_completion_audit_20261101.json`, `validation_metrics_verified_20261101.json`,
`selected_recipe.json`, `study_summary.json`, `results.md`,
`existing_R0_reuse_verified_20261101.json`, `final_refit_proposal_20261101.md`, and
trial000 crop/disease `best_epoch_15.pt` / `validation_epoch_15.npz` / histories.
The completed monitor is deleted after the report; the chat remains available.

## More trials within five hours authorized, 2 November 2026

The user requested **"extend for more trials, time limit 5 hours"** and asked
to retain what the last run learned. A deadline-only continuation starts at
trial8, retaining all eight prior gamma-zero trial states, recipes, scores and
best-so-far paired intermediate curves. There is no eight-trial ceiling in the
continuation. Prior COMPLETE recipes remain eligible; pruned/pending trials do
not. The median pruner uses completed parent curves immediately. The existing
three-COMPLETE TPE threshold is retained (two inherited), as are all search ranges,
model/pretrained/seed/input/training controls and gamma0 adaptive decisions.

The completed parent is preserved read-only in its original folder. Its SQLite
is copied using a read-only backup and verified table-for-table before new writes.
New output: `runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102`.
Manifest `configs/optuna_convnext_base_gamma0_continue_20261102.json`; controller
`optuna_gamma0_continue.py`; launcher `run_optuna_gamma0_continue.ps1`. Parent
scientific identity is unchanged; continuation code/config/provenance is pinned
separately and included in new checkpoint config fingerprints. All130 parent
files, including its database and selected/terminal artifacts, are pinned; all
eleven original scientific code hashes were unchanged before launch.

Five continuation tests plus seven existing gamma-zero-controller regression
tests passed. They cover exact history copy and immutable source DB, additional
trial numbers past8, inherited median pruning, pending/truncated trials excluded
from selection, earlier ties, terminal-write-before-SQLite-tell recovery, exact
mid-pair target checkpoint recovery, and preserved user pause. No real dataset
training or official inference occurs in those tests.

Before launch all prior controller/launcher/supervisor exits were confirmed and
all six relevant OS locks were acquired/released. The GPU remained available;
CUDA preflight is mandatory. Hidden launcher PID40268 started actual controller
PID30408 at **2 November 08:25 MYT**. Startup audits are in progress; the five-hour
training window begins only after these and the physical-batch4 CUDA preflight.
The run holds six locks: new continuation, completed gamma-zero parent, original
study, R0 finalization, Tiny refit and historical calibrated Optuna. No prior
calibrated observation, full refit, official inference or automatic next budget
is authorized. A quiet five-hour completion monitor follows saved state/deadline.

Startup audit and CUDA preflight passed. Parent observations copied and verified
table-for-table; source database remains unchanged. The five-hour window started
**2 November 2026 08:27:11 MYT**, with saved deadline **13:27:11 MYT**
(`2026-11-02T05:27:11.939366+00:00`). Continuation identity:
`a1baa762627511c592c1d8a9db2cd32754c35256e899e3c4be0f2355656a8760`.
All six study locks are held by the healthy run. Physical4 CUDA preflight had
three finite updates, peak1,831,088,640 bytes, median.156s. Trial8 crop epoch1 is
training at LR1.1182763375409254e-5/head1.1182763375409254e-4, WD.007480096160281177,
dropout.1, gamma0. Initial best eligible remains inherited fresh R0 / H0.08575053675.
Monitor `convnext-base-gamma-zero-five-hour-continuation` is ACTIVE and quiet
during healthy progress. `launch_verified.json` and `history_import_verified.json`
record exact process/window/history checks; no new fixed trial ceiling applies.

## Five-hour continuation cap stop verified, 2 November 2026

State `window_complete` at **13:30:23 MYT**, deadline13:27:11. The active crop
epoch completed/checkpointed before exit, consuming191.224613s beyond the cap.
New trials8–12: five PRUNED, zero COMPLETE. Trial13 is pending/RUNNING in SQLite,
crop3/disease2/paired2, next target disease3. It retains full finite model,
optimizer/scaler, loader generator, Python/NumPy/Torch/CUDA RNG and horizon30
scheduler state; it is not eligible to win. Inherited2 COMPLETE/6 PRUNED remain
unchanged, so totals are2 COMPLETE/11 PRUNED/1 pending. There was no trial-count
cap. TPE never activated because the unchanged three-COMPLETE threshold was not
reached. No further compute, protocol change or automatic renewal was launched.

New runtime is **17,826.926s / 4h57m07s target-epoch time** across35 paired epochs
plus one crop epoch (71 targets), and **18,191.224613s / 5h03m11s wall time** from
the post-audit start. Parent36,680.093s target/37,357.911111s wall, prior calibrated
work and all earlier full-data refits are excluded from these new figures.

R0trial0 remains the verified COMPLETE winner, epoch15/gamma0: seen/PU Top1
89.6752%/4.5028%, H1 **8.5751%**; seen/PU Top5 98.6079%/27.5797%, H5 **43.1037%**.
Its LR5e-5/head5e-4, WD.05, dropout.2 match the reusable existing full-data weights.
Best new retained PRUNED recipe is trial10, epoch3, H1 6.8199%; it cannot win.
It used LR1.615153807802008e-5, head10x, WD.07548889368492154, dropout.4.
All validation values rank323 original training compositions, hiding actual10
unseen candidates. R0 raw independent PU crop/disease Top1 is92.1201%/3.7523%;
factor accuracy from its winning composition is89.1182%/5.4409%, a separate metric.
PU-only20-candidate R0 diagnostic is59.4747% Top1/98.3114% Top5, also separate.

New terminal learning curves show disease fit84–95% while combined validation
disease stalls around54.6–56.0% by epochs6–9; trial10's disease fit92.22% versus
validation55.84% at9 is consistent with limited transfer/overfitting. H0 is noisy
and often peaks early, while R0 peaked at15. Correctly applied median pruning
does not establish what pruned candidates would score after more epochs. Two
COMPLETE curves remain a small reference set; this is not exhaustive evidence
that CNNs or hyperparameter tuning cannot improve.

CPU audit `audit_gamma0_continuation_stop.py` confirmed positive current WinAPI
exit for30408/40268 and acquired/released all six OS locks. All130 parent and131
older historical hashes (49 final artifacts), source14/pretrained/original8000/
cache7193/scientific-code/continuation-identity hashes passed. All new terminal,
verification and artifact hashes, finite selected weights/logits, independently
recomputed H0/Top5, SQLite params/states/reports, deterministic sampling streams,
paired best epoch/earlier ties, pruning reasons and LR horizon30 passed. Original
parent DB observations match row-for-row. Logs/history/checkpoints remain stable.
No official inference, full-data refit or calibration occurred. The matching R0
full-data weights can be reused; further compute needs explicit authorization.

[Verified window report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_report_verified_20261102.md),
[stop audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_stop_verified_20261102.json),
[validation metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/validation_metrics_verified_20261102.json),
[selected recipe](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/selected_recipe_for_window.json),
[summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/study_summary.json),
[results](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/results.md).
The completion monitor is removed after reporting; the chat remains available.
