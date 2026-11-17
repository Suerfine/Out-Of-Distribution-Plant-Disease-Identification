# Optuna study: ConvNeXt-Base IN22K -> IN1K

**First window checkpointed at its ten-hour cap, 1 November 2026 06:14:13 MYT.**
Two new trials completed; three were pruned; trial 6 retains both epoch-1 recovery
checkpoints and its RUNNING SQLite identity. Trials 7/8 have not started.
Selection is incomplete and no new budget or final refit is authorized.
[Verified window report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/results.md).

**Status: execution authorized 31 October 2026.** Initially prepared as a plan
on 30 October, this now has user approval to start the selection search for
`convnext_base.fb_in22k_ft_in1k`. The accepted first window is at most ten hours
and eight new paired trials. Final full-data refitting and official testing are
separate later work. The immutable manifest is
[configs/optuna_convnext_base_seed42.json](configs/optuna_convnext_base_seed42.json).

This uses the existing Base architecture. Both independent networks already
fine-tune pretrained weights; Optuna would search their training hyperparameters.
It is not a new architecture search and does not guarantee better generalization.

## Baseline and frozen data

Reuse the existing pinned Base initialization, revision
`67758fcbcf9f007ec7630bf8636b74a8f24e52a0`, SHA-256
`4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381`.
Every new trial starts both targets freshly from that original checkpoint.
Resumption may restore only that same trial's own recovery state.

Reuse the saved seed-42 partition under
`runs/baseline/independent_cnn_baseline_study_seed42/`:

| Partition | Images | Purpose |
|---|---:|---|
| Fitting | 5,798 | Fit each selection trial |
| Seen validation | 862 | Validation on trained compositions |
| Pseudo-unseen validation | 533 | Validation on 20 held-out compositions |
| Selection-excluded | 807 | Original held-out-pair cap; excluded during selection |
| Full official training set | 8,000 | Restore only for the final refit |

Do not re-sample splits between trials. Hash the exact files, order, labels,
partition, input/pretrained weights and recipe before executing a trial.
Validation candidates are the original 323 training compositions: hide the ten
real unseen compositions, retain the 20 pseudo-unseen compositions, and apply
seen/pseudo-unseen membership from this fixed partition.

The R0 reference was rechecked on 30 October from its saved **validation** logits
using the corrected shared evaluator. Epoch 15 / gamma 6.2 remains:

| Validation metric | Value |
|---|---:|
| Seen composition Top-1 | 83.41% |
| Pseudo-unseen composition Top-1 | 46.53% |
| Calibrated Top-1 harmonic mean | 59.74% |
| Gamma-zero Top-1 harmonic mean | 8.58% |

These validation values are the search reference. Official test scores are not
an objective, acceptance threshold, pruning signal or reason to keep searching.
The original R0/final outputs and unfinished BiT R1 remain preserved.

## Proposed first search space

Keep the first search small; share one sampled recipe across the crop and
disease targets, as in R0.

| Parameter | Proposed distribution | R0 reference |
|---|---|---:|
| Backbone LR | Log-uniform 1e-5 to 1.5e-4 | 5e-5 |
| Weight decay on eligible weights | Log-uniform 0.005 to 0.10 | 0.05 |
| Head dropout | 0.0, 0.1, 0.2, 0.3, 0.4 | 0.2 |

Head LR stays ten times the backbone LR. Biases and normalization parameters
retain zero decay. Keep DropPath 0.1, input 224, cached resize 256, original
bicubic/ImageNet preprocessing and augmentation, smoothing 0.1, gradient clipping
1, CUDA FP16, physical batch 4 / accumulation 8 / effective batch 32, workers 2,
and independent target seed 42 for both crop and disease. Tiny's disease seed
43 is not transferred to Base.

Keep the original three warmup epochs, cosine horizon 30, minimum LR ratio 1%,
30 paired-epoch maximum and eight-paired-epoch early-stopping patience. Do not
compress the LR schedule to a shortened trial or final-refit length. No search
on resolution, augmentation, label smoothing, DropPath, batch size, seed, network
architecture or validation partition in this first study.

## Objective, sampler and pruning

Use one maximization objective: **held-out validation composition Top-1 harmonic
mean after validation-selected calibration**, consistent with the existing Base
study. Save gamma-zero H and raw factor metrics as secondary diagnostics. If the
future goal is specifically gamma-zero performance, change that objective before
starting rather than switching objectives after inspecting outcomes.

One trial is a paired crop+disease training. After both targets finish the same
paired epoch, score the saved seen/pseudo-unseen validation images. Sweep gamma
from 0 to 9.9 by 0.1 using the shared deterministic evaluator. Save current and
best H, epoch, gamma, both head metrics and logits for the best paired checkpoint.
Return the best eligible validation H reached by that completed trial.

Use a seeded TPE sampler and median pruning. These are supported Optuna sampling
and pruning mechanisms; the training objective must report intermediate values
and explicitly check whether pruning is requested.
[Official Optuna overview](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

Proposed pilot settings: sampler seed 42, three completed observations before
TPE suggestions; median pruning after two completed observations, a six-paired-
epoch grace period, then checks every three paired epochs. A pruning step must
have at least one completed comparable curve. Report best-so-far validation H
at each paired epoch for all new trials. Never report or prune using a crop-only
half-epoch, a test score, or a different epoch's final R0 score.
[MedianPruner semantics](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html).

The existing R0 can be registered as a completed incumbent only after matching
all identities and the recalculated validation objective. This avoids redundant
baseline retraining. Its saved best validation logits support epoch 15, not a
recomputed full curve: do not blindly import old intermediate scores into the
pruner. Missing comparison steps must remain unavailable.
[Registering completed trials](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.add_trial).

Only completed, verified trials and the audited R0 incumbent can win. Pruned,
failed and budget-interrupted trials are not eligible. Exact ties retain the
incumbent/earlier trial, earlier epoch and smaller gamma. If nothing improves
on R0 on validation, keep R0 and report the search outcome honestly.

## Proposed compute budget and realistic timing

Authorized first selection window:
**10 hours, at most eight new distinct trial pairs, one GPU job at a time**.
The audited cached R0 reference does not count as a newly trained pair.
Count the trial cap across the whole resumed study; resuming a paused trial
must not create another trial or reset either limit.

Measured Base R0 selection time was 11,852.532 seconds across 23 paired epochs:
about 515 seconds (8m 35s) per paired epoch. At that recorded speed:

| Work | Approximate target-epoch time |
|---|---:|
| Six paired epochs, earliest proposed pruning point | 52 minutes |
| Fifteen paired epochs | 2h 09m |
| Thirty paired epochs | 4h 18m |

Validation, CPU gamma sweeps, checkpoint traffic and machine load can change
wall time. A ten-hour cap may produce only a handful of completed/pruned trials;
eight is a ceiling, not a promise, and 20 full trials would be far too expensive.
A four-hour window would be a small pilot rather than a convincing search.

Use both a study-level limit and a controller deadline checked at safe epoch
boundaries, so a long-running trial cannot silently consume an additional
window. Save full optimizer/scaler/loader/RNG state when budget or user pause
interrupts training. A pause is not algorithmic pruning or a failed scientific
trial. Do not automatically open another window.
[Optuna study limits](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).

Final refitting/evaluation is a separate compute allocation, not included in the
proposed ten-hour search window. At the measured Base full-data rate, two
15-epoch target refits took 2h 09m; two 30-epoch target refits would be about
4h 18m, plus inference/verification. The combined scope and budget must be clear
before a future launch. A small validation gain merits additional seed/split
confirmation before strong claims; those repeats are not silently added here.

## Final refit and official evaluation, after selection

If a new recipe wins, freeze its configuration, selected paired epoch and
validation-selected gamma before touching official test scores. Freshly refit
both networks on all 8,000 official training images for that epoch count,
retaining the original LR horizon. Finish both before official inference.

Report gamma-zero and frozen-calibrated all-333 composition Top-1/Top-5 and H
on 2,000 seen / 1,448 unseen images; raw independent-head metrics separately from
winning-composition factors; and separately labeled ten-unseen-only diagnostics.
Verify checkpoint hashes, finite weights/logits/scores, prediction order/counts,
the 323x333 seen confusion matrix totaling 2,000, and app score parity.

If R0 remains the winner and the identity/epoch/gamma are unchanged, preserve
and reuse its already verified final checkpoints/results rather than repeating
the same refits and official evaluation. The official test sets have already
been viewed during project development: later results must be described as
iterative benchmark results, not a newly untouched blind holdout.

## Implementation and artifacts needed before a launch

Optuna 5.0.0 is pinned in `requirements_optuna.txt` and installed in the CUDA venv.
The isolated controller is `optuna_search.py`; the hidden PowerShell launcher is
`run_optuna.ps1`. The old fixed master spec and old study controllers stay intact.

Implementation detail: independent TPE sampling uses reproducible parameter RNG
streams derived from `SeedSequence([42, trial_number, parameter_index])`. This
keeps suggestions identical even after a process exits between SQLite parameter
commits. Three **COMPLETE** observations are required before TPE; until then use
uniform/log-uniform random sampling in the same fixed distributions. Training
and loader seeds stay 42 for both targets. SQLite retains completed/pruned/running
trial identities and intermediate reports. R0 has no imported intermediate curve.

Best checkpoint pairs use separate epoch files and one atomic pair-state pointer.
A partially committed pair is evaluated again from its saved target logits,
without training either target twice. Pauses and deadline stops keep the Optuna
trial RUNNING and preserve all optimizer/scaler/loader/RNG state. Terminal trial
hashes, finite selected weights and recomputed validation objectives are verified
before SQLite completion/pruning and recovery-file retirement. A user pause is
preserved until explicit `-Resume -ClearStop`; recovery cannot open a new budget.

Startup holds this search's OS lock plus the original study, R0 finalization and
Tiny refit locks. Auditing reuses the exact saved CSVs, checks 14 source artifact
hashes and the pretrained hash, all 8,000 original training image hashes, and the
7,193 selected cache images against the original JPEG preprocessing. The clock
starts after audits/CUDA preflight. No official image inference occurs.

From the repository root:

```powershell
.\independent_cnn_baseline\run_optuna.ps1 -Detach

# Only after confirming controller exit and lock release, within the same window:
.\independent_cnn_baseline\run_optuna.ps1 -Detach -Resume -DeadlineUtc '<exact saved UTC deadline>'
```

`STOP.request` in the new run folder asks for a safe pause. A ten-hour cap stop
is incomplete selection work; it does not make interrupted trials eligible or
authorize a new selection window. The monitor must report the checkpointed state
and stop instead of automatically restarting.

Isolated output directory:
`runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/`.
Use persistent SQLite study storage, a versioned fixed manifest, immutable
per-trial configurations, source/data/code hashes, parameter and paired-epoch
histories, best validation checkpoints/logits and trial-state/pruning reasons.
Checkpoint/cache identity must include the trial configuration, data order,
preprocessing, seed and weight hashes. Retain full recovery state for active or
paused trials; release large optimizer recovery files after a trial becomes
terminal while retaining its best checkpoint pair and history. Keep retention
within available disk without deleting historical studies.

Before launching: test fresh-trial initialization, objective/pruner behavior,
paired reporting, hidden real-unseen candidates, exact ties, deadline/user-pause
recovery, database/trial identity, unchanged LR horizon during refits, and final-
refit-before-test ordering using generated data. Recheck CUDA memory and source
hashes. A single OS lock must reject duplicate controllers; healthy processes
must not be restarted. Recovery requires positive exit evidence and the exact
saved deadline. Preparation and all validation trials must avoid official test
inference. Seven new Optuna generated-data tests, twenty existing baseline tests
and five shared ranking tests passed. Original/cache/source/pretrained audits and
the physical-batch-4 CUDA preflight passed before launch. The accepted search
started 31 October 2026 at 20:12:38 MYT and its recorded deadline is
1 November 2026 at 06:12:38 MYT. The first trial is running; inspect status/logs
for current progress. The completion monitor stays quiet during healthy training.

## Planning references

- [Original Base R0 completion record](../runs/baseline/independent_cnn_baseline_study_seed42/trials/independent_cnn_baseline_convnext_base_r0_seed42/complete.json)
- [Current three-backbone results](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/results.md)
- [Completed Tiny refit, a different architecture/recipe](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/results.md)

Base R0 completion-record SHA-256:
`c6ab42e8aa04243214e18507d64b8c6cadd557a5eb57611c678c1a3190013d05`.
Saved best validation-logits SHA-256:
`e9fbd709e49689183c5fd58076f342825a19f727c9cc49af791d54e257be33e3`.
