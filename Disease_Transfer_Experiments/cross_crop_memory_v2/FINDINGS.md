# Original-split feature memory: no demonstrated improvement

The feature-memory contrastive configuration did not improve the existing
baseline or establish an advantage over its matched control. Keep the original
independent CNN as the reference. This is one full training seed and one fixed
loss/queue setting; it does not rule out every contrastive-learning approach.

Training, evaluation and independent numerical verification completed on
5 November 2026 at 11:52 pm Malaysia time. The training figure was visually
reviewed on 6 November. All 13 implementation tests and the GPU preflight passed;
independent NumPy verification passed 42,924 checks.

## Official results

All rows use 333 seen+unseen candidates. Values are percentages.

| Model | Seen Top-1 | Unseen Top-1 | H | Unseen disease Top-1 |
|---|---:|---:|---:|---:|
| Saved original baseline | 71.05 | 12.43 | 21.16 | 12.29 |
| Fresh ordinary-classification control | 70.50 | 12.22 | 20.84 | 13.47 |
| Classification + cross-crop feature memory | 70.65 | 12.09 | 20.64 | 13.40 |

The experimental model gets 175/1,448 unseen compositions correct, versus
177 for the fresh control and 180 for the saved original baseline. Its seen
score improves by 0.15 percentage points over the fresh control, while unseen
accuracy falls by 0.14 pp and H falls by 0.19 pp.

The predeclared paired composition-bootstrap 95% interval for the unseen
difference is [-0.74, +0.35] pp; for H it is [-1.23, +0.43] pp. Both include
zero. These intervals condition on this one training seed and the observed
composition sample; they do not estimate initialization-to-initialization
variation. The result provides no evidence that this added objective improves
the matched training recipe.

The disease-factor score is higher than the historical baseline for both new
models, but the control is slightly above the contrastive model. That factor
improvement cannot be credited to the contrastive loss.

## What was held fixed

Both new disease models use the exact original 6,746-image fitting and
1,254-image seen-validation manifests. There is no pseudo-unseen validation.
The original selected crop checkpoint is unchanged. Every fitting image appears
exactly once per epoch, and both arms use identical exposure sequences and
starting weights. Checkpoint selection uses validation disease Top-1, with
ordinary cross-entropy breaking ties, and all selected hashes were frozen
before official test inference.

The original 30-epoch cap, eight-epoch patience, optimization settings and
preprocessing were retained. The fresh control stopped after 28 epochs and
selected epoch 20 (85.01% validation disease Top-1). The experimental arm ran
30 epochs and selected epoch 23 (84.85%). There were seven AMP-skipped optimizer
steps in the control and eight in the experimental arm; all selected weights
and final logits are finite. Recorded epoch times include loading and validation
and are not isolated compute-speed benchmarks.

The additional loss uses a 1,024-entry detached feature queue from fitting
images only, reset each epoch, with weight 0.1 and temperature 0.1. A matching
disease label on a different crop is a positive; healthy anchors and same-label,
same-crop comparisons are excluded. Queue entries can become stale. These
fixed choices were not adjusted using official test scores.

The new control's scores are close to the historical reference. The large
seen-accuracy drop in the earlier composition-holdout trial is absent here.
The historical run still differs in shuffled order and seed-stream execution;
the two new arms provide the direct matched comparison. Official tests were
familiar from previous studies, so the evidence remains exploratory.

## Artifacts and decision

- [Generated results](runs/pilot_seed42/results.md), [all metrics](runs/pilot_seed42/metrics.json), and [paired intervals](runs/pilot_seed42/paired_analysis.json)
- [Independent verification](runs/pilot_seed42/independent_verification.json)
- [Training curves](runs/pilot_seed42/training_curves.png)
- [Fixed protocol](PLAN.md) and [frozen selections](runs/pilot_seed42/selection_frozen_before_test.json)

The original baseline remains unchanged. Per the user's preference, future
experiments should reuse the existing baseline rather than routinely retraining
a control. No additional training is queued by this completed study.
