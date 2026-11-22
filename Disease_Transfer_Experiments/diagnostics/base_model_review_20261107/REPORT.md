# Base model diagnostic review

Completed 8 November 2026, Malaysia time. This is a post-experiment diagnostic of saved predictions, not a model-selection experiment. No training, inference, weight updates, split changes, or calibration search were run.

Subsequent action: the user accepted the proposed development procedure with "start it." The separate [SimSiam Base development run](../../simsiam_base_development_v1/PLAN.md) was launched on 8 November. The statements below describe the audit itself, before that authorization.

The baseline is not at an accuracy ceiling. Its uncalibrated accuracy on unseen crop-disease combinations is only **14.02%**. The recent Base experiments did not improve this: SSL fixed 14 baseline mistakes but broke 28 previously correct predictions. The clearest observed weakness is disease discrimination and transfer to new crop-disease combinations. This does not establish a single cause inside training.

## Like-for-like results

All three runs use ConvNeXt-Base and the same frozen full-data Base crop network. Pair prediction ranks all 333 allowed combinations. "Unseen" means a held-out crop-disease combination; its individual crop and disease labels occur in training. This is the existing compositional task, not a newly introduced domain-generalization task.

| Saved model | Raw seen pair Top-1 | Raw unseen pair Top-1 | Raw unseen disease Top-1 | Unseen pair Top-1, fixed gamma 6.2 |
|---|---:|---:|---:|---:|
| Original Base | 73.80% | 14.02% (203/1,448) | 14.50% | 76.52% |
| SimSiam + supervised disease training | 72.60% | 13.05% (189/1,448) | 11.26% | 74.65% |
| Crop-conditioned disease model | 74.15% | 13.74% (199/1,448) | 13.47%* | 72.58% |

*For the conditioned model, disease accuracy is computed from its disease distribution marginalized over predicted crops; it is not an independent disease head.

The raw and calibrated baseline use the same weights. Calibration subtracts 6.2 from the 323 training-seen candidate scores for every image; it does not receive that image's true split or label. Its large effect shows sensitivity to the preference for familiar combinations. It does not mean the raw disease classifier is already excellent. There was no new gamma search in this review.

## Where accuracy is lost

### Pepper bacterial spot: a clear raw disease regression

There are 220 official unseen pepper bacterial-spot images. The independent disease head is correct on **125 for Base versus 79 for SSL**; the true disease's median rank moves from 1 to 2. Raw composed-pair accuracy falls from **151 to 139** correct.

Among the 14 previously correct pair predictions lost by SSL, 7 become pepper frogeye leaf spot and 3 become pepper powdery mildew. The unchanged crop branch and these within-pepper errors point to disease decisions as a concrete place to investigate. They do not prove why the representation changed.

The 8,000-image training set contains 188 bacterial-spot images on other crops: tomato 92, peach 74, papaya 14, cauliflower 7, and plum 1. Pepper itself has only 63 training images: healthy 57, blossom-end rot 3, powdery mildew 2, and frogeye leaf spot 1. Pepper bacterial spot is intentionally absent from training. Thus neither recognizing pepper nor memorizing pepper's observed diseases solves this held-out combination.

### Grape black rot: both new methods favor grape esca

On 271 unseen grape black-rot images, pair predictions using the existing fixed gamma 6.2 are correct on **204 for Base, 162 for SSL, and 155 for the conditioned model**.

SSL breaks 53 previously correct predictions and fixes 11; **48 of those 53 regressions become grape esca**. The conditioned model breaks 55 and fixes 6; **51 of its 55 regressions become grape esca**. This is a shared observed error pattern, not proof that both methods fail for the same internal reason.

The raw disease head is already weak here: Base recognizes black rot on 0/271 images, versus SSL on 1/271. Base's median true-disease rank is 18; SSL's is 23. The calibrated result therefore should not be read as 204 cases of strong, standalone black-rot recognition.

There are only **36 black-rot training images** across apple 21, cauliflower 7, brassica 5, cabbage 2, and coffee 1. Grape has 167 training images, including 68 esca images but no black rot. This support imbalance is a plausible contributor worth testing on development data. It does not establish label noise, visual similarity, background shortcuts, or a causal explanation for the failures.

## What the checks establish

- Saved filenames, row order, composition labels, and candidate IDs align across the original Base and both new Base models.
- Saved crop logits are exactly identical for all three models on both official test partitions.
- Cached JPEG-byte hashes have no exact training/test overlap in either official partition. This is a check of the stored hashes, not a fresh scan for every possible near duplicate or recompression.
- The saved original Base gamma-zero development logits reproduce its recorded metrics to an absolute tolerance of 1e-12 using the original objective function. Copies of the five original development manifests match byte for byte.
- The saved development crop and disease checkpoints exist and are hashed. They allow the baseline reference and crop network to be reused without training a control.

These checks found no alignment or score-reproduction problem. They do not establish that every training choice was correct, that all labels are correct, or that no implementation bug could exist. The earlier evaluation-export API error was repaired after training; it is not evidence of a gradient-training failure. Lower SSL loss and noncollapsed feature statistics did not translate into better measured disease accuracy.

## Recommended next step

Use raw scores to judge learning progress and evaluate candidate recipes on development data before another full-data overnight run. A useful next hypothesis is whether a disease-training change can improve transfer across crops while retaining separation between diseases on the same crop. This review does not choose a new loss, SSL epoch count, or class-specific patch from official test results.

The concrete [development protocol proposal](DEVELOPMENT_PROTOCOL_PROPOSAL.md) reuses the original Base split, its saved gamma-zero baseline logits, and its saved development crop checkpoint. It requires no routine baseline retraining. It is **not applied**, because using pseudo-unseen development validation would revisit the user's earlier preference to keep the no-pseudo-validation procedure.

The official tests have already been inspected repeatedly. Their results remain useful descriptive comparisons, but they cannot now serve as untouched confirmation for choices inspired by this review. The evidence here comes from single saved runs; it does not measure training-seed variability.

## Reproduction and evidence

Run from the project workspace with `Plant_Disease/.venv/Scripts/python.exe`:

```powershell
& Plant_Disease/.venv/Scripts/python.exe Disease_Transfer_Experiments/diagnostics/base_model_review_20261107/analyze.py
& Plant_Disease/.venv/Scripts/python.exe Disease_Transfer_Experiments/diagnostics/base_model_review_20261107/verify_reference.py
```

- [Audit and input hashes](audit.json): full per-composition counts, support, disease ranks, and paired prediction counts.
- [All changed decisions](paired_decision_changes.json): individual regressions and fixes, including whether the crop branch was correct.
- [Illustrative regression file references](illustrative_regressions.json): first three sorted-filename SSL regressions for each of the two highlighted settings; these are examples, not a random sample.
- [Verified existing development reference](development_reference_verified.json): exact recomputed metrics and checkpoint/source hashes.
- [Machine-readable proposed protocol](development_protocol_proposal.json) and [byte-identical existing manifests](reference_manifests).
- [Earlier method/reference audit](../../POST_EXPERIMENT_AUDIT_20261107.md): the Tiny/Base reference correction and limitations of previous comparisons.
