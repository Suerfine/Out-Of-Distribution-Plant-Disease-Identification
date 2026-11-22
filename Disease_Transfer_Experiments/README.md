# Disease-transfer experiments

The user authorized [two ConvNeXt-Base detail experiments](cnn_detail_base_v1/README.md)
on 8 November 2026: 384-resolution disease training, followed automatically by
the same global view plus four local patches and learned attention. Both use
the original pinned ImageNet Base initialization, fitting5798/seen862/pseudo533
development split, saved crop logits and baseline reference. Inputs are built
from original images into a new 512 cache. Both remain the same CNN backbone;
no other CNN or transformer is included. Their [fixed protocol](cnn_detail_base_v1/PLAN.md)
uses raw validation H, max30/patience8 per arm, checkpoints and shared 15-minute
progress logs. The queue stops after both development reports.

The [feature-composition Base development run](feature_composition_base_v1/FINDINGS.md)
completed on 8 November 2026 at 5:02 pm Malaysia time on the first worker attempt,
passing 7,747 verification checks. It stopped after 13 epochs and selected epoch
5. Raw seen/pseudo-unseen/H were 87.47%/2.44%/4.75%, below both the saved Base
reference (89.68%/4.50%/8.58%) and SSL development reference
(89.79%/3.75%/7.20%). The synthetic training exposures and fitting-only donors
were verified, but the recipe did not improve real held-out-combination accuracy.
Keep the saved references; this completed run has no full-data refit or official
test evaluation queued.

The user authorized [ConvNeXt-Base compositional feature augmentation](feature_composition_base_v1/README.md)
on 8 November 2026. This development experiment reuses the fitting-only SSL
encoder and original Base split/reference. It trains on real features and learned
crop/disease recombinations from fitting-only donors, with the full Base backbone
trainable. The [fixed protocol](feature_composition_base_v1/PLAN.md) declares the
new auxiliary modules, synthetic target balance and extra donor exposures. Raw
validation H selects checkpoints with max30/patience8. Existing models remain
preserved; this run stops after development reporting. Execution state is in its
`runs/development_seed42` status files.

The [full-data selected SSL run](simsiam_base_full_selected_v1/FINDINGS.md)
completed on 8 November 2026 at 1:36 pm Malaysia time on its first worker attempt,
passing 33,937 verification checks. It reused the verified full-data 20-epoch SSL
encoder and trained a fresh disease model for the development-selected 9 epochs.
Raw official seen/unseen/H were 73.00%/14.30%/23.91%, versus Base's
73.80%/14.02%/23.56%. The small raw H gain (+0.35 points) has a composition-bootstrap
interval including zero. Secondary fixed-gamma H was 71.14%, versus Base's 71.44%.
Preserve both the candidate and original reference; this is not a confirmed
improvement. No further training is queued by this completed run.

The user authorized the [full-data selected SSL recipe](simsiam_base_full_selected_v1/README.md)
on 8 November 2026. It verifies and reuses the identical existing 20-epoch
ConvNeXt-Base SSL encoder trained on all 8,000 images, then trains a fresh disease
head and encoder for the development-selected 9 supervised epochs. The original
Base crop logits and comparison are reused. Official 2,000 seen / 1,448 unseen
test evaluation follows the frozen budget; raw gamma zero is primary and the
unchanged fixed gamma 6.2 is secondary. See its [fixed protocol](simsiam_base_full_selected_v1/PLAN.md)
and run status files for execution state. Earlier models and results are preserved.

The [SimSiam Base development run](simsiam_base_development_v1/FINDINGS.md)
completed on 8 November 2026 at 5:32 am Malaysia time and passed 131 verification
checks. At gamma zero, seen/pseudo-unseen/H were 89.79%/3.75%/7.20%, versus the
saved Base development reference's 89.68%/4.50%/8.58%. It completed 20 SSL and
17 supervised epochs, selecting epoch 9 and stopping with patience 8. It did not
improve the primary development measure. Keep the Base reference; this run has
no official-test evaluation or full-data refit queued.

The user authorized the [SimSiam Base development run](simsiam_base_development_v1/README.md)
on 8 November 2026. It restores the exact original Base development partition:
5,798 fitting, 862 seen-validation, 533 pseudo-unseen-validation images, and
807 additional held-out images excluded during development. Its
[fixed protocol](simsiam_base_development_v1/PLAN.md) uses 20 fitting-only SSL
epochs, then at most 30 supervised disease epochs with patience 8 and gamma-zero
validation H selection. It reuses the saved development crop logits and baseline;
no control retraining, automatic full-data refit, or official-test evaluation is
queued. Execution state is recorded in that run's status files.

The [Base diagnostic review](diagnostics/base_model_review_20261107/REPORT.md)
is complete (8 November 2026). Raw unseen pair accuracy is 14.02% for Base,
13.05% for SSL, and 13.74% for the conditioned model; the baseline is not at an
accuracy ceiling. The review identifies specific disease confusions and verifies
the reusable original Base development reference. Its
[development protocol](diagnostics/base_model_review_20261107/DEVELOPMENT_PROTOCOL_PROPOSAL.md)
was accepted for the development run on 8 November. Saved baseline references
make routine control retraining unnecessary.

The [SimSiam Base run](simsiam_base_v1/FINDINGS.md) is complete and verified
(7 November2026,11:35 pm Malaysia time;33,909 checks). With the same fixed
gamma6.2, its seen/unseen/H were66.10%/74.65%/70.12%, below the saved Base's
67.00%/76.52%/71.44%. Keep the original Base baseline; no further training is
queued. The following records preserve the run's execution history.

The completed experiment is [SimSiam self-supervised learning on ConvNeXt-Base](simsiam_base_v1/README.md),
which finished20 SSL epochs and saved/verified its encoder at7:10 pm Malaysia
time on7 November. It paused before supervised training as requested. The user's
subsequent “continue” resumed and completed the remaining15 supervised disease
epochs, without redoing SSL. Earlier artifacts and the paused encoder archive
remain preserved.
The numerical recipe is unchanged,
authorized on 7 November 2026: 20 fixed label-free plant-image SSL epochs from
the original ImageNet initialization, then 15 supervised disease epochs. Reuse
the original Base crop model, full 8,000-image training set, all 333 candidates
and both existing scoring settings. No new control, test-image SSL or calibration
search. A detached Windows supervisor, two-slot checked checkpoints and exact
cursor/RNG/augmentation recovery protect the run. Its GPU and fault-recovery
preflight checks passed; see the [fixed overnight protocol](simsiam_base_v1/PLAN.md).

The completed [crop-conditioned disease experiment on ConvNeXt-Base](crop_conditioned_base_v1/README.md)
remains preserved.
On 6 November 2026 the user clarified that the intended reference is the saved
Base baseline and asked to replace only the current experiment. Previous runs
remain preserved and are not being rerun. The Base experiment reuses its saved
disease weights and frozen crop model. Its [fixed protocol](crop_conditioned_base_v1/PLAN.md)
uses 15 additional epochs on all 8,000 training images, with no new validation
selection: the saved full-data Base weights already saw every training image.
Both gamma0 and the existing frozen gamma6.2 are reported, with no new calibration.
It completed and passed verification on 6 November 2026 at 11:35 pm Malaysia
time. It did not improve the saved Base baseline: with the same fixed gamma6.2,
seen/unseen/H were 67.40%/72.58%/69.90%, versus 67.00%/76.52%/71.44%.
See the [Base findings](crop_conditioned_base_v1/FINDINGS.md).

The earlier [current Tiny experiment](crop_conditioned_disease_v1/README.md)
was stopped after its first epoch. Its checkpoint and logs remain preserved
under `crop_conditioned_disease_v1/runs/full_seed42`; it was not evaluated on
the official tests. This was a baseline-reference error, not a Base result.

The [full learned compositional classifier](compositional_classifier_v1/README.md)
trains one joint image encoder with shared crop/disease embeddings and a learned
pair composer. It uses the original manifests and applies the accuracy-first
validation principle to primary pair scores. It compares against the saved
baseline without retraining a control. See its fixed protocol for this explicit
adaptation from the historical separate factor-head checkpoint rules.
It completed on 6 November 2026 and failed on unseen combinations (0.14%
Top-1 versus the baseline's 12.43%). See its
[findings](compositional_classifier_v1/FINDINGS.md).

## Default experiment policy

User preference confirmed on 5 November 2026: compare new experimental models
against the existing original independent CNN baseline. Reuse its saved models
and verified results; do not routinely retrain a baseline/control for every new
experiment. Any additional control is a separate experiment whose purpose and
compute cost should be made explicit.

The 6,746 fitting / 1,254 seen-validation manifests describe the earlier Tiny
series. They must not be treated as fresh validation for the saved full-data
Base checkpoint. The completed full-data Base experiments used predeclared fixed
budgets without new pseudo-unseen validation. The newly authorized development
run uses the exact original Base development partition documented above.
Keep the official 2,000 seen / 1,448
unseen tests and all 333 candidates. Preserve historical artifacts, report
material protocol differences, and reuse the appropriate crop checkpoint when
an experiment changes only disease training. Prior frozen protocols are unchanged.

The already-started feature-memory V2 run is allowed to finish both its control
and experimental arms. This exception does not make control retraining the
default for future experiments. This policy does not alter its frozen protocol.

The [original-split feature-memory experiment](cross_crop_memory_v2/README.md)
uses the exact baseline manifests and validation policy, full image coverage,
and a 1,024-entry fitting-feature memory. It completed and passed independent
verification on 5 November 2026; it did not improve the saved baseline.
See its [findings](cross_crop_memory_v2/FINDINGS.md).

The [cross-crop supervised contrastive pilot](cross_crop_supcon_v1/FINDINGS.md)
completed and passed independent verification on 5 November 2026.
The contrastive loss improved its sampler-matched control, but the full recipe
did not outperform ordinary classification and did not improve the reserved
composition audit. The original independent CNN remains the reference.

All experiments live separately from historical baseline and RL implementations.
