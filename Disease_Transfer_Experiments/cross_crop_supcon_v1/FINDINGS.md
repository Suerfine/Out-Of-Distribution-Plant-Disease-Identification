# Cross-crop contrastive pilot: useful loss signal, no baseline improvement

**The contrastive objective improved its sampler-matched control on the official
unseen set. The complete recipe did not outperform ordinary classification and
gave no improvement on the reserved composition audit. Keep the original
independent CNN as the reference.**

Completed and independently verified on **5 November 2026 at 8:25 pm Malaysia
time**. This was a bounded first trial with one full CNN training seed, three
independent disease networks and one shared independent crop network. All CNN
parameters were trainable. No calibration was used.

## Matched official comparison

All rows use the same crop checkpoint, fitting/validation roles, preprocessing
and all 333 composition candidates on 2,000 official-seen and 1,448
official-unseen images. Values are percentages.

| Disease training | Official seen Top-1 | Official unseen Top-1 | H | Unseen disease Top-1 | Reserved audit Top-1 |
|---|---:|---:|---:|---:|---:|
| Ordinary image-shuffled classification | 51.60 | 12.36 | 19.95 | 9.39 | 7.69 |
| Grouped-sampling classification | 44.80 | 9.88 | 16.18 | 7.39 | 3.08 |
| Same grouped sampling + cross-crop contrastive loss | 45.90 | 11.46 | 18.35 | 9.32 | 3.08 |

The predeclared primary comparison is the third row minus the second:

- **Unseen composition Top-1: +1.59 percentage points**, or 23 additional correct
  images. The paired whole-composition bootstrap 95% interval is **[+0.31,
  +3.68] pp**; the paired image interval is [+0.90, +2.35] pp.
- **H: +2.16 pp**, with whole-composition interval [+0.57, +3.35] pp.
- Unseen disease Top-1 increases from 7.39% to 9.32%, so the gain also appears
  in the disease factor. Both grouped arms remain below the ordinary control
  on official unseen composition accuracy and H in this run.

The sampling change itself reduced unseen Top-1 by 2.49 pp relative to ordinary
classification. Adding the contrastive objective recovered part of that observed
loss. The full recipe's difference from ordinary classification was -0.90 pp
unseen and -1.60 pp H; the corresponding composition-bootstrap intervals include
zero. These are conditional comparisons for one training seed.

## Reserved audit and error interpretation

The untouched audit contains 195 images from eight different compositions,
absent from every CNN's fitting and checkpoint-selection data. Both grouped
arms correctly classify **the same 6/195 images (3.08%)**. Ordinary classification
gets 15/195 (7.69%). The contrastive objective changes 24 audit predictions but
corrects no errors and breaks no correct decisions.

Its audit bootstrap delta is consequently [0, 0] on this observed sample. That
does not establish equivalence on other compositions or initialization seeds.
The reserved audit supplies no evidence that the official-set gain transfers
reliably to another set of held-out combinations.

Post-hoc inspection of the official unseen predictions finds 145 changed
decisions: 26 errors corrected and three correct decisions broken. Net gains
occur in bell-pepper/bacterial spot (+13), cotton/bacterial blight (+2),
sugarcane/rust (+4) and tomato/early blight (+4). Correct counts for the other
six compositions are unchanged. See [post-hoc counts](runs/pilot_seed42/posthoc_summary.json).
These diagnostics did not select or alter a model.

## Training, data and historical comparison

The fixed split has 6,413 fitting images, 1,197 seen-validation images, 195
development images from eight held-out combinations, and the separate 195-image
audit. Exact duplicate image content is kept apart across roles. Every crop and
disease factor remains represented in fitting. All models start from the cached
original ImageNet-22K weights, never a historical disease-fine-tuned checkpoint.

The grouped arms use identical initialization and per-epoch image sequences.
The added loss pulls together the same supplied disease label across different
crops, with weight 0.1 and temperature 0.1. Healthy labels are excluded as
positive anchors. Approximately 43% of sampled images have an eligible positive;
all sampled images still contribute classification loss.

All three disease arms early-stopped after eight epochs and selected epoch 1 by
seen/development composition H. The crop CNN stopped after 11 epochs and selected
epoch 5. The curves show improving fitting and seen-validation performance while
held-out-combination performance deteriorates. This setup did not prevent that
generalization failure. All four selected checkpoint hashes were frozen before
any audit or official prediction.

The historical original baseline scored **71.05% seen / 12.43% unseen / 21.16%
H**. Its fitting partition, checkpoint rule and 30-epoch cap differ from this
trial's held-composition split, H-based selection and 20-epoch cap. The comparison
is descriptive; the decline from the historical seen score cannot be attributed
to the added loss alone. The new matched controls provide the relevant comparison.
Of the 2,000 supplied official-seen images, 100 belong to the 16 compositions
deliberately withheld in this trial; [full metrics](runs/pilot_seed42/metrics.json)
also separate these from the other 1,900 images.

The study completed **35 CNN epochs**, totaling **1 h 34 min 53 s** of recorded
epoch time, including training and validation. Elapsed times are not isolated
GPU benchmarks. Dynamic AMP skipped one optimizer step per CNN, four in total;
all selected weights and final logits are finite. Epoch checkpoints retain
optimizer, scaler and RNG states for reproducibility.

## Verification and limits

All **nine implementation tests** and the real-CNN forward/backward preflight
passed. Independent verification passed **61,684 checks**, including NumPy
recomputation of primary metrics and rankings, every exported prediction row,
checkpoint selection, matching grouped exposures, disjoint data roles, finite
weights and frozen hashes. Historical source files, baseline models, predictions
and reports checked by hash remained unchanged. The training figure was visually
inspected.

There is one full training seed, eight audit compositions and ten official unseen
compositions. Bootstrap intervals do not measure variation between newly trained
CNNs. Official tests were familiar from earlier project studies. The contrastive
comparison set is only eight images per microbatch; gradient accumulation does
not enlarge it. Shared disease names need not describe identical symptoms or
biology across crops. This trial provides a limited signal for the added loss,
not a reliable improvement over the existing baseline or a rejection of every
contrastive-learning approach.

User-approved cleanup removed 3.63 GiB of regenerable arrays and their cache
completion markers from the completed weight-editing study. The
[cleanup receipt](cache_cleanup_receipt.json) and original completion metadata
are preserved in this experiment directory.

## Reviewable artifacts

- [Fixed pretraining protocol](PLAN.md) and [configuration](config.json)
- [Generated results](runs/pilot_seed42/results.md), [all metrics](runs/pilot_seed42/metrics.json), and [paired intervals](runs/pilot_seed42/paired_analysis.json)
- [Training curves](runs/pilot_seed42/training_curves.png)
- [Frozen selections and checkpoint hashes](runs/pilot_seed42/selection_frozen_before_audit_and_test.json)
- [Independent verification](runs/pilot_seed42/independent_verification.json)
- [Selected experimental contrastive checkpoint](runs/pilot_seed42/ce_supcon/best.pt)
- [Runner and resumption instructions](README.md)
