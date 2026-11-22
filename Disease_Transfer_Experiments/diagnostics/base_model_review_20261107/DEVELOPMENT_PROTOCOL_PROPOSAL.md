# Proposed Base development protocol

Prepared 8 November 2026. **Accepted by the user's subsequent "start it" on 8 November.** Applied to the separate [SimSiam Base development run](../../simsiam_base_development_v1/PLAN.md). The text below preserves the proposed procedure; the existing full-data Base, SSL, and conditioned models and historical protocols remain preserved.

The purpose is to decide whether a new disease-training recipe helps before paying for a full-data run. The original Base already has a suitable saved development split and reference checkpoints, so this proposal does not require retraining a control.

## Reuse the original Base development partition

The staged manifests are byte-identical copies from `Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_study_seed42`, not a new random split.

| Existing role | Images | Use during development |
|---|---:|---|
| `fit.csv` | 5,798 | Experimental disease training; label-free training too, if SSL is tested |
| `validation_seen.csv` | 862 | Validate on new images of fitting combinations |
| `validation_unseen.csv` | 533 | Validate on 20 combinations withheld from fitting |
| `selection_excluded.csv` | 807 | Excluded during development; these are extra images of the same held-out combinations, above the original validation cap |
| `all_train.csv` | 8,000 | Full-data refit after the recipe is fixed |

All 55 crop labels and 175 disease labels remain represented in the 5,798 fitting images. None of the 20 pseudo-unseen combinations occurs there. "Pseudo-unseen" simply means practicing on withheld combinations drawn from the original training set; it does not alter the official held-out combinations.

The official 2,000 seen and 1,448 unseen test images stay outside training and selection. In particular, using validation images without labels for SSL would still expose their images during representation learning; this proposal excludes them and the 807 extra held-out images from development SSL.

## Reuse saved references

Use the saved gamma-zero Base development trial:

`Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000`

- `crop/best_epoch_15.pt`: reusable frozen development crop model, trained on the fitting partition.
- `disease/best_epoch_15.pt`: saved disease reference when needed for initialization under an explicitly declared new experiment.
- `validation_epoch_15.npz`: reusable baseline development logits.

The [verification](development_reference_verified.json) reproduces these reference results exactly:

| Existing Base development metric, gamma 0 | Value |
|---|---:|
| Seen combination Top-1 | 89.675174% |
| Pseudo-unseen combination Top-1 | 4.502814% |
| Harmonic mean of those two accuracies | 8.575054% |

These are development results, not official test results. Do not compare a new development run against a baseline trained on all 8,000 images. The latter has already seen the proposed validation images. For the same reason, the completed full-data SSL and conditioned models cannot be retroactively treated as clean validation runs on this split.

## Fix the comparison and selection rules

1. Use ConvNeXt-Base, the same label IDs, and the appropriate saved crop reference. Train only the experimental disease pipeline unless an experiment explicitly requires another change.
2. Make **uncalibrated composition Top-1 harmonic mean** the primary development selection measure. It balances accuracy on seen and pseudo-unseen combinations. Apply gamma 0 with no score-offset sweep. Use the earliest checkpoint on an exact metric tie, matching the saved reference rule.
3. During development, reproduce the original objective: rank the 323 original training combinations, including all 20 pseudo-unseen ones. Mask the 10 official unseen combination columns. Do not restrict candidates according to an individual validation image's true split or crop.
4. Report seen and pseudo-unseen accuracy separately, disease Top-1/Top-5, per-combination results, and fixes versus regressions. A better aggregate should be checked for broad improvement rather than a gain concentrated in one combination.
5. Declare the method, initialization, augmentation, training budget, stopping rule, seed, and intended comparison before launching it. The historical Base development search allowed at most 30 supervised epochs with patience 8 and selected epoch 15; this does not establish an optimal budget for SSL or a new method. This proposal does not launch or select a new training recipe.
6. Use development data for any future recipe decisions. Do not tune losses, checkpoints, epochs, or gamma against the official tests or target a patch solely at the two official-test classes highlighted in the diagnostic.

For final reporting, raw gamma-zero results are primary. Preserve the existing **fixed gamma 6.2** comparison as a separately labeled secondary result for continuity. It is not newly optimized and is not guaranteed to suit a new model's score scale. Any future calibration study needs its own declared development procedure.

## Full-data experiment after a development decision

Once a recipe is chosen, start a fresh refit following its declared initialization and budget on all 8,000 original training images. If the method includes SSL, do the full-data SSL stage there as part of that fresh refit. Reuse the existing full-data Base crop model and original Base final results; do not retrain a routine control.

Report the final model over all 333 official candidates at gamma 0 and the unchanged secondary gamma 6.2. Record the differences between development and full-data training. The existing official tests have already informed this project, so treat comparisons there as exploratory. A newly sealed independent set would be needed for untouched confirmation of a method selected in response to these diagnostics.

## Decision record

The user earlier asked to keep the no-pseudo-unseen-validation procedure. At that time, the discussion incorrectly used the Tiny reference; the actual original Base did have the partition documented here. After reviewing this proposal and the diagnostic, the user instructed "start it." The separate development run therefore restores this original Base partition. Its method-specific recipe and stopping rule are recorded in its PLAN.md; the historical full-data experiments are unchanged.
