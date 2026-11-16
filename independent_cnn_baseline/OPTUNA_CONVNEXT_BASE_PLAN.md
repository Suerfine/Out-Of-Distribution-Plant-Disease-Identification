# ConvNeXt-Base tuning: uncalibrated objective

## Five-hour continuation stopped and verified, 2 November 2026

Verified checkpointed cap stop at **13:30:23 MYT**, after the exact 13:27:11 MYT
deadline. Additional trials8–12 are PRUNED; trial13 remains RUNNING in SQLite
with crop epoch3/disease epoch2, two paired epochs and next target disease3.
New counts: **0 COMPLETE, 5 PRUNED, 1 pending**; inherited counts stay2 COMPLETE,
6 PRUNED. Total2 COMPLETE/11 PRUNED/1 pending. R0trial0 remains the only winning
recipe, selected epoch15/gamma0/H0 **8.5751%**. TPE did not activate: its unchanged
three-COMPLETE threshold was not reached. No pruner/warmup/search change was made.

New compute: **17,826.926s target epochs (4h57m07s)**,35 paired epochs plus one
crop epoch,71 target epochs; **18,191.224613s post-audit wall (5h03m11s)**. The
191.224613s deadline overrun finished/checkpointed crop3. These exclude parent
36,680.093s target/37,357.911111s wall, historical calibrated work and all refits.
Positive WinAPI exit checks passed for controller30408/launcher40268. All six
locks were acquired/released. Parent130 and historical131 hashes, scientific and
continuation identities, original8000/cache7193/source14/pretrained/code hashes,
all new terminal artifacts and finite retained weights/logits passed CPU audit.
H0/Top5 were recomputed; paired epoch/tie/prune reasons and LR horizon30 match.
Pending optimizer/scaler/RNG/loader/scheduler states are intact. Further compute
requires explicit authorization; no refit, official inference or calibration ran.
This is a completed budget window, with an unfinished recipe and non-exhaustive
search. Existing matching R0 full-data weights can be reused. The quiet monitor
is removed after reporting; the chat is not archived.

[Verified report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_report_verified_20261102.md),
[audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/window_stop_verified_20261102.json),
[metrics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/validation_metrics_verified_20261102.json),
[summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/study_summary.json),
[results](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/results.md),
[selected recipe](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102/selected_recipe_for_window.json).

## Five-hour continuation authorization and launch record

User: **"extend for more trials, time limit 5 hours"**, followed by a request to
use what the previous run learned. The continuation starts additional trial
numbers at8 and carries forward all eight verified gamma-zero observations:
two COMPLETE and six PRUNED, including parameters, scalar/intermediate H0,
original states and parameter-stream numbering. It has **no fixed trial-count
ceiling**; sequential new trials continue until the saved five-hour deadline.
The original completed eight-trial output stays unchanged and is referenced
read-only; its SQLite is backed up into a separate continuation output folder.
Zero observations from the superseded calibrated study enter this continuation.

Scientific controls remain exactly those below. Gamma0 governs every adaptive
decision. The median pruner can use inherited completed curves immediately.
The existing three-COMPLETE threshold for TPE is retained; with two inherited
COMPLETE trials, startup sampling continues until another completes. No search
space narrowing, pruner change or TPE warmup change is implicitly authorized.
Pruning is evidence of an early trajectory, not proof a recipe can never improve.

Manifest `configs/optuna_convnext_base_gamma0_continue_20261102.json`; controller
`optuna_gamma0_continue.py`; launcher `run_optuna_gamma0_continue.ps1 -Detach`;
output `runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102`.
The inherited scientific input identity remains unchanged. A separate immutable
`continuation_identity.json` pins the new administrative controller, manifest and
read-only parent. New trial configs include that continuation identity in their
checkpoint fingerprint. All130 parent files, original source/data/cache/pretrained
hashes and131 historical preservation hashes are checked. Six OS locks cover the
new root, completed gamma-zero parent, original study, R0 finalization, Tiny refit
and superseded calibrated Optuna root.

The new five-hour clock begins after audits and CUDA preflight. Read exact start
and deadline from status.json; finish/checkpoint the active target epoch at the
cap. STOP.request pauses safely and is never auto-cleared. Recovery only after
positive current-process exit, stable logs/checkpoints and six free OS locks:

```powershell
.\independent_cnn_baseline\run_optuna_gamma0_continue.ps1 -Detach -Resume -DeadlineUtc '<exact saved deadline>'
```

Verified window: **2 November 2026 08:27:11–13:27:11 MYT**, exact deadline
`2026-11-02T05:27:11.939366+00:00`. Launcher PID40268 / controller PID30408
started trial8 after audits/CUDA preflight. At launch all six locks were held, parent SQLite
and pinned continuation code are unchanged, eight inherited trial observations
were verified, and the completion monitor was active. CUDA physical4 preflight
passed three finite optimizer updates, peak1,831,088,640 bytes, median step.156s.

There is no -Hours recovery parameter or automatic new budget. Existing COMPLETE
parent trials remain eligible; PRUNED or interrupted new trials cannot win. At
the five-hour stop freeze selected_recipe_for_window.json from verified COMPLETE
trials and report new complete/pruned/pending counts separately from the inherited
eight observations. It is completion of the authorized window, not exhaustive
completion of the search space. Preserve pending full optimizer/RNG checkpoints.
No final refit, official inference or calibration is authorized. Update owned
notes/results, report actual new runtime separately from prior windows, remove
the quiet monitor at verified cap/pause completion, and do not archive the chat.

## Completed initial gamma-zero study

**Completed and independently verified: 1 November 2026 20:54:28 MYT.**
All eight fresh paired trials are terminal (2 COMPLETE, 6 PRUNED). Fresh R0 wins
at paired epoch15 / gamma0, validation seen89.6752%, pseudo-unseen4.5028%,
H1 **8.5751%** and H5 **43.1037%**. Best completed sampled candidate: trial1,
epoch2 / H1 **6.8246%**. The additional budget was applied safely at the original
cap; the study finished before its extended deadline without adding trials.
The TPE threshold of three COMPLETE observations was never reached. No final
refit, official inference or calibration occurred. Controllers and locks are free.
[Verified report](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selection_report_verified_20261101.md),
[summary](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/study_summary.json),
[recipe](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selected_recipe.json),
[audit](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/selection_completion_audit_20261101.json).
Fresh runtime is 10h11m20s target-epoch time / 10h22m38s wall, excluding historical
selection/refits. The existing full-data R0 recipe/epoch matches and can be reused
with separately labeled gamma-zero outputs; no additional refit is launched.

**Corrected by user instruction on 1 November 2026: tuning uses gamma zero.**
**New ten-hour gamma-zero window authorized on 1 November 2026.** The isolated
controller is `optuna_gamma0.py`, manifest
`configs/optuna_convnext_base_gamma0_seed42.json`, launcher
`run_optuna_gamma0.ps1`. A fresh R0 pair runs first; the total ceiling is eight
fresh paired trials including that reference, one CUDA training at a time.
The initial ten-hour clock starts after audits and CUDA preflight. The user
subsequently authorized ten additional hours on 1 November 2026; the administrative
continuation described below retains the same eight-trial ceiling. No final refit or
official test inference is included. The preceding ten-hour run optimized calibrated validation
H and is historical evidence only; it is not an uncalibrated tuning result.
Its database, source manifest, checkpoints and audit records remain unchanged.
The old launcher is blocked by `OBJECTIVE_SUPERSEDED.json` in its run directory.

[Correction review and recomputed diagnostics](../runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/uncalibrated_objective_review.md)
records the evidence. The previous plan is preserved in
[OPTUNA_CONVNEXT_BASE_CALIBRATED_HISTORY.md](OPTUNA_CONVNEXT_BASE_CALIBRATED_HISTORY.md).

## Objective that must govern the whole search

Maximize **uncalibrated held-out validation composition Top-1 harmonic mean**:

`H0 = 2 * seen_validation_top1 * pseudo_unseen_validation_top1 / (seen_validation_top1 + pseudo_unseen_validation_top1)`

Use zero when the denominator is zero. Compose raw scores as
`log P(crop) + log P(disease)` and keep **gamma fixed at 0**. There is no gamma
sweep, seen-score penalty or calibration-dependent choice in tuning.

The same H0 must control every adaptive decision:

- paired best-checkpoint retention and selected epoch;
- eight-paired-epoch early stopping;
- Optuna intermediate values and median pruning;
- TPE trial observations and suggestions;
- completed-recipe ranking and the final frozen recipe.

Report seen and pseudo-unseen Top-1 separately, Top-5/H5 and raw independent
factor accuracies as secondary diagnostics. Do not silently switch to seen-only
accuracy, factor accuracy or calibrated H as the objective. Ties keep the earlier
trial and epoch. Both targets must finish the same epoch before reporting.

The CPU validation implementation is `uncalibrated_objective.py`. Generated-logit
tests check that a calibration-dependent gain cannot enter H0, real unseen
candidates stay hidden, and invalid logits/validation labels are rejected.
`optuna_gamma0.py` integrates this objective into paired checkpoint selection,
early stopping, pruning, TPE reports and final recipe ranking. The new SQLite
study starts with a queued **fresh** R0 and imports zero historical observations.

## Frozen model, data and training controls

Keep `convnext_base.fb_in22k_ft_in1k`, original pretrained revision
`67758fcbcf9f007ec7630bf8636b74a8f24e52a0`, SHA256
`4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381`.
Use the exact saved seed-42 partition: fitting 5,798, seen validation 862,
pseudo-unseen validation 533 from 20 held-out training compositions, and 807
cap-excluded images. Validation ranks against the 323 training compositions;
the ten real unseen candidates are hidden. Never use official tests for tuning.

Retain the prepared search dimensions: backbone LR 1e-5..1.5e-4 log, weight decay
.005...10 log, head dropout 0/.1/.2/.3/.4. Head LR remains ten times backbone LR;
biases/normalization retain zero decay. Keep DropPath .1, 224 input/JPEG256 cache,
original normalization/augmentation, smoothing .1, clipping 1, batch4/accum8,
workers2, CUDA FP16 and both target/loader seeds42. Keep the original 30-epoch
cosine horizon, three warmup epochs, minimum LR ratio .01 and 30 paired epochs
maximum. Short refits must retain that original horizon.

Sampler/pruner controls stay as prepared: reproducible independent TPE from
master seed42, three COMPLETE observations before TPE, median pruning after two
completed observations, grace six paired epochs and interval three, with at least
one comparable completed curve. The intermediate values are best-so-far **H0**.
No old calibrated intermediate values, pruning decisions or TPE observations
are eligible for import into a corrected study.

## Meaning of the work already performed

R0's saved epoch-15 checkpoint has independently recomputed H0 **8.58%**.
The retained calibration-selected checkpoints for completed trials 1 and 2 have
H0 **6.51%** and **5.46%** respectively. The previously reported **62.95%**
is calibrated H for trial 2; it is not the required tuning score.

Historical per-epoch logs contain gamma-zero diagnostics, but checkpoint retention
followed calibrated H. Trial 1's logged H0 peak is epoch19 / 7.21%; only its
calibration-selected epoch16 weights survive. Trial 2's logged H0 peak is epoch2 /
6.82%; only epoch30 weights survive. The missing model pairs cannot be recovered
from metric histories or logits. Pruning/early stopping and later suggestions
would also differ under H0, so relabeling the database cannot repair the search.

Treat R0 as a fixed reference until any proposed incumbent import has separately
verified the gamma-zero selection/stopping trajectory and available checkpoint.
Do not import the old calibrated completion record as a new gamma-zero trial.
Preserve every existing completed/pruned trial and trial6 recovery checkpoint.

## Separate study identity and authorization

The corrected run uses a separate immutable objective-bearing manifest
and SQLite/output namespace:
`runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/`.
Do not overwrite the historical study or resume its RUNNING trial under changed
selection rules. Any reuse of saved training state needs an explicit, audited
protocol that accounts for prior pruning, checkpoint loss and sampler history.

The user authorized a new ten-hour window after the objective correction. This
does not reopen the old calibrated study or extend its deadline. New trial 0 is
R0 freshly trained under H0; trials 1..7 are new sampled recipes. Only COMPLETE,
verified trials from this new study are eligible. Its first summary has no winner
until a new trial completes. The old 8.58% R0 checkpoint remains a historical
fixed reference, not an imported completed observation.

```powershell
.\independent_cnn_baseline\run_optuna_gamma0.ps1 -Detach -Hours 10

# Recovery only after positively confirming controller exit and all locks free:
.\independent_cnn_baseline\run_optuna_gamma0.ps1 -Detach -Resume -DeadlineUtc '<exact saved UTC deadline>'
```

Recovery retains the new window's exact deadline. `-Resume -Hours 10` is rejected;
there is no automatic renewal. `STOP.request` in the new output folder asks for
a safe pause after the active target epoch; `-ClearStop` is allowed only for an
explicit user resume. Active checkpoints include optimizer/scaler/RNG/loader
state; terminal trials retain their gamma-zero-best weights/logits and histories.
The controller holds five OS locks: its own, the original study, R0 finalization,
Tiny refit and the historical calibrated Optuna run. No healthy controller is
restarted. A quiet monitor reports only completion, checkpointed cap/pause stops,
actionable failures or required input, and is removed when this window ends.

## Explicit ten-hour extension authorized, 1 November 2026

User instruction: **"extend for 10 more hours"**. Add ten hours to the original
cap, retaining start **2026-11-01T02:31:50.263638+00:00**. Original deadline is
**2026-11-01T12:31:50.263638+00:00**; extended deadline is
**2026-11-01T22:31:50.263638+00:00**, or **2 November 06:31:50 MYT**.
The total authorized time is twenty hours, with at most eight fresh paired trials
including R0. It may finish earlier when all eight are terminal.

The original running controller is healthy and keeps a monotonic deadline in
memory. Do not change its pinned sources or edit its live status to simulate a
hot extension. `extend_gamma0_window.py` is a separate administrative supervisor
whose hash and authorization are recorded in
`extension_authorization_20261101.json` in the gamma-zero run. It waits for
`window_complete` / `checkpointed` at the original cap, positively confirms
controller/launcher exit, checks stable log/history/checkpoint evidence, acquires
and releases all five study OS locks, and applies the additional budget once.
The original start and every trial/checkpoint are preserved; status records
`window_hours: 20`, the extended deadline and an explicit budget-extension event.
Before/after audit files record the change. No scientific identity migration,
observation import, trial reset, final refit or official inference occurs.

After lock release it calls the unchanged launcher with `-Detach -Resume
-DeadlineUtc '2026-11-01T22:31:50.263638+00:00'`. `-Resume -Hours 10` remains
invalid. A STOP.request or paused state is preserved; extension alone does not
authorize clearing a pause. Failure requires inspection. Never duplicate a
healthy controller or repeat an ambiguous launch. The supervisor has a separate
OS lock and records one resume attempt; six generated-data/administrative tests
cover exact budget, preserved partial pairs, idempotence, pause/live-state guards,
immutable configuration, duplicate locks, denied process queries and exact resume
arguments. Original eleven pinned scientific/helper code hashes remain unchanged.

The completion monitor remains active across the original cap and follows the
supervisor's audit/progress records. That intermediate cap stop is already
authorized to continue. Notify only completion, user pause, failure or the final
extended cap stop. Further compute beyond the extended deadline still requires
new explicit authorization. Keep new target-epoch runtime and the two authorized
budget segments distinguishable from the previous calibrated window/refits.

## Final refit and official evaluation after corrected selection

After a completed H0 search and separate refit authorization, freeze the selected
recipe and gamma=0, freshly refit both targets on all 8,000 official training
images for the selected paired epoch count, and finish both before official
inference. Evaluate 2,000 seen / 1,448 unseen images against all 333 compositions.
Report raw composition Top-1/Top-5 and harmonic means, raw factors separately from
factors of the winning composition, and ten-unseen-only diagnostics separately.
A calibration run is not part of this corrected scope.

Verify finite weights/logits/scores/metrics, hashes, prediction order/counts,
323x333 seen confusion matrix totaling 2,000, and app score parity. Official tests
have already been viewed during project development; further results are
iterative benchmark results rather than a new untouched blind holdout.
