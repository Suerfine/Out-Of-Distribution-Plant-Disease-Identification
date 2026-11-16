# Disease-only CLIP + GAN with a ConvNeXt seen expert

## Final pipeline

The system predicts a crop–disease pair. It uses ConvNeXt-Base crop and disease scores for seen pairs, and ConvNeXt crop scores plus disease-only CLIP scores for unseen pairs.

1. **Score the candidates.** For each pair `(crop, disease)`, add the ConvNeXt crop logit to the CLIP disease logit. CLIP supplies one logit per disease across 175 labels.
2. **Choose an expert.** Compute the gap between the highest seen-pair and unseen-pair scores from step 1. Route to seen when the gap is at least `gamma`. The official candidate list contains 323 seen and 10 unseen pairs.
3. **Return the top pair from the selected expert.** The seen expert ranks seen candidates by ConvNeXt crop logit plus ConvNeXt disease logit. The unseen expert ranks unseen candidates using the scores from step 1.

The gate makes the same branch decision as subtracting `gamma` from all seen scores in step 1 and taking the top pair. The seen handoff then lets ConvNeXt change the final pair within the seen branch. The routing offset leaves the ranking within each branch intact.

## Official test results

The metrics are image-level top-1 accuracy on full crop–disease pairs. The harmonic mean is `2 × seen × unseen / (seen + unseen)`.

| Routing offset | Seen (2,000 images) | Unseen (1,448 images) | Harmonic mean |
|---|---:|---:|---:|
| 0 | 72.90% (1,458) | 48.34% (700) | 58.13% |
| 3.0 | 66.80% (1,336) | 82.80% (1,199) | 73.95% |

The 3.0 offset was selected on the seed-42 pseudo-unseen development split by maximizing harmonic mean. The official model used all 8,000 training images. CLIP ViT-B/16 was adapted with eight LoRA epochs using crop, disease, and pair supervision with text anchoring. Its encoder was then frozen, and the disease head was fit for one epoch on real embeddings plus 300 synthetic vectors per unseen candidate pair. The GAN used the official task ontology to specify those 10 target pairs and real training features for its crop anchors. Official test images were used for evaluation.

[Development selection](clip_disease_only/development_results.json) · [Official results](clip_disease_only/official/results.json) · [Training selection](clip_disease_only/official/selection.json)

## What each change added

Each row uses its own comparison. The gains overlap and cannot be added together.

| Change | Rationale | Matched official result |
|---|---|---|
| Crop-anchor the CLIP feature GAN | Start from a healthy feature of the target crop when generating a disease feature. | Harmonic mean rose from **46.27% to 49.02%** at zero offset and from **65.71% to 67.55%** at the shared 0.03 offset. |
| Use ConvNeXt crop scores with pair-based CLIP + GAN | Use an independently trained crop predictor as evidence for the pair ranking. | At zero offset in the standalone comparison, seen accuracy went from **66.10% to 68.00%**, unseen from **38.95% to 40.68%**, and harmonic mean from **49.02% to 50.90%**. |
| Train a disease-only CLIP head and keep multi-task LoRA | Give CLIP a single prediction target and let ConvNeXt supply crop evidence. | Compared with the pair-based two-expert system at zero offset, harmonic mean went from **52.32% to 58.13%**, with **111 more unseen** and **8 fewer seen** pairs correct. With each system's development-selected offset, harmonic mean went from **73.17% to 73.95%**. Routing and unseen ranking both changed. |
| Hand seen predictions to ConvNeXt | Use the stronger seen disease classifier after routing. | At offset 3.0 with identical routing and unseen predictions, seen accuracy rose from **62.80% to 66.80%** and harmonic mean from **71.43% to 73.95%**. The handoff fixed 160 seen errors and introduced 80. At zero offset it added 104 net correct seen predictions. |
| Tune the routing offset | Send more borderline images to the unseen expert. | In the final pipeline, increasing the offset from 0 to 3.0 raised unseen accuracy from **48.34% to 82.80%** and lowered seen accuracy from **72.90% to 66.80%**, lifting harmonic mean from **58.13% to 73.95%**. |

[GAN at zero offset](official/results.json) · [GAN at shared offset](official/fixed_gamma_results.json) · [Crop-score comparison](convnext_crop_clip_disease/results.json) · [Pair-based two-expert baseline](convnext_crop_clip_disease/calibrated_two_expert_results.json) · [Fixed-gate seen handoff](clip_disease_only/official/seen_handoff_ablation.json)

The full score ensemble was a separate experiment. It reached **76.02% harmonic mean** at its development-selected offset of 4.9 and **31.02%** at zero offset. [Ensemble results](score_ensemble/results.json)

## Fixed and gate-dependent score blending

Both blends keep all candidate pairs eligible. They combine the log-probabilities of ConvNeXt crop + ConvNeXt disease and ConvNeXt crop + CLIP + GAN disease. The fixed blend uses one weight for every image. The gate-dependent blend varies the ConvNeXt weight with the seen–unseen score gap. Blend settings and offsets were selected on development split 42 and applied to the official test.

| Official configuration | Seen at selected offset | Unseen at selected offset | HM at selected offset | HM at zero offset |
|---|---:|---:|---:|---:|
| Earlier full-score ensemble | 66.15% | 89.36% | 76.02% | 31.02% |
| Disease-only CLIP hard routing | 66.80% | 82.80% | 73.95% | **58.13%** |
| Fixed blend with disease-only CLIP | 65.55% | 91.02% | **76.21%** | 53.73% |
| Gate-dependent blend with disease-only CLIP | 64.40% | **91.37%** | 75.55% | 57.38% |

The calibrated fixed blend gained **0.19 percentage points** in harmonic mean over the earlier calibrated ensemble. It correctly classified 24 additional unseen images and 12 fewer seen images. The zero-offset blend rows use weights selected for zero-offset performance.

[Development comparison](convnext_crop_clip_disease/soft_gate_development.json) · [Official blend comparison](clip_disease_only/official/soft_gate_comparison.json)

## Official comparison of final variants

Overall accuracy pools the 2,000 seen and 1,448 unseen test images. Harmonic mean weights the two split accuracies equally. All values below are percentages. In the 50/50 ConvNeXt rows, the original and oversampled disease heads share one backbone and their scores receive equal weight.

### Zero routing offset

| Variant | Seen | Unseen | Overall | HM |
|---|---:|---:|---:|---:|
| Earlier full-score ensemble | 74.10 | 19.61 | 51.22 | 31.02 |
| Hard routing, pair-based CLIP + GAN | 73.30 | 40.68 | 59.60 | 52.32 |
| Hard routing, disease-only CLIP + GAN | 72.90 | 48.34 | 62.59 | 58.13 |
| Hard routing, oversampled ConvNeXt head | 73.10 | 48.34 | 62.70 | 58.20 |
| Hard routing, 50/50 ConvNeXt heads | 73.25 | 48.34 | **62.79** | **58.25** |
| Fixed blend, original ConvNeXt head | 69.70 | 43.72 | 58.79 | 53.73 |
| Gate-dependent blend, original ConvNeXt head | 72.25 | 47.58 | 61.89 | 57.38 |
| Fixed blend, 50/50 ConvNeXt heads | 70.00 | 43.30 | 58.79 | 53.50 |
| Fixed-blend gate, 50/50 ConvNeXt heads and specialist answers | 73.40 | 43.30 | 60.76 | 54.47 |

### Calibrated routing or score offset

| Variant | Offset | Seen | Unseen | Overall | HM |
|---|---:|---:|---:|---:|---:|
| Earlier full-score ensemble | 4.90 | 66.15 | 89.36 | 75.90 | 76.02 |
| Hard routing, pair-based CLIP + GAN | 2.90 | 69.25 | 77.56 | 72.74 | 73.17 |
| Hard routing, disease-only CLIP + GAN | 3.00 | 66.80 | 82.80 | 73.52 | 73.95 |
| Hard routing, oversampled ConvNeXt head | 3.00 | 67.15 | 82.80 | 73.72 | 74.16 |
| Hard routing, 50/50 ConvNeXt heads | 3.00 | 67.15 | 82.80 | 73.72 | 74.16 |
| Fixed blend, original ConvNeXt head | 5.21 | 65.55 | 91.02 | **76.25** | **76.21** |
| Gate-dependent blend, original ConvNeXt head | 6.09 | 64.40 | 91.37 | 75.73 | 75.55 |
| Fixed blend, 50/50 ConvNeXt heads | 5.21 | 67.30 | 87.71 | 75.87 | 76.16 |
| Fixed-blend gate, 50/50 ConvNeXt heads and specialist answers | 5.21 | 67.35 | 87.02 | 75.61 | 75.93 |
| Fixed-blend gate with specialist answers, cutoff matched on development | 5.86 | 64.55 | 91.37 | 75.81 | 75.65 |

The 50/50 ConvNeXt head weights were preset comparison points. When tuning head weights on development with an unseen-accuracy floor, hard routing selected 75% oversampled, which reached 58.18 HM without an offset and 74.10 HM with calibration. The fixed blend selected the original ConvNeXt head. The small differences between ConvNeXt head variants on the official test come from a single test set.

[Oversampled seen head](seen_oversampling/official_results.json) · [Head blends and ensemble-gate comparisons](seen_oversampling/ensemble_routed_experts_results.json)

## Improving seen recognition

At offset 3.0, the hard-routing system sent 1,747 of 2,000 true seen images to ConvNeXt. It got 1,336 right. Of its 411 errors on correctly routed images, 338 had the right crop and wrong disease. These counts describe the hard-routing system.

A next development experiment is to audit the most frequent same-crop confusions for class coverage, capture conditions, and label consistency, then test targeted sampling of real images from the confused classes. Compare with ordinary fine-tuning under the same training budget and evaluate across held-out source images with the gate fixed. Report top-1 changes and per-class changes so improvements on one disease can be checked against regressions on others.

The full-leaf cassava specialist is a separate candidate for a small pilot. Validate it on fresh leaves and let it intervene only when both cassava diseases are shortlisted. Count corrected errors and newly introduced errors across the seen branch.

## Follow-up experiments

- **Disease-only LoRA.** Removing the crop and pair losses from encoder training raised disease accuracy on seen pairs from **66.55% to 68.10%** and reduced unseen disease accuracy from **45.03% to 36.26%**. The final pipeline retains the multi-task encoder. [Original disease accuracy](clip_disease_only/official/disease_accuracy.json) · [Disease-only LoRA](clip_disease_only/official_disease_only_lora/results.json)
- **Alternative unseen expert.** Under the earlier pair-based gate at offset 2.9, the three ConvNeXt + GAN heads averaged **73.15% harmonic mean**, close to **73.17%** for CLIP + GAN. [Fixed-gate comparison](convnext_crop_clip_disease/fixed_gate_convnext_gan_unseen_results.json)
- **Gate tuning.** The second-look agreement rule gave no development gain. On seed 42 at matched seen accuracy, the older hybrid equation with its fixed 0.02 ConvNeXt weight got **346/533** pseudo-unseen pairs right versus **337/533** for the crop-aware pair-based gate. Tuning the weight reduced that to **338/533**. [Agreement rule](convnext_crop_clip_disease/agreement_gate_development.json) · [Weight tuning](convnext_crop_clip_disease/older_gate_weight_development.json)
- **Seen expert changes.** A selective CLIP override went from **613/667 to 609/667** on correctly routed seen development images. Hard-negative fine-tuning tied ordinary fine-tuning at epoch 3, **595/652**, and hinge-only head training got **594/652** versus **595/652** for cross-entropy. [Selective override](convnext_crop_clip_disease/selective_clip_seen_rerank_development.json) · [Hard negatives](seen_hard_negative/results.json) · [Hinge-only head](seen_hinge_head/results.json)
- **Cassava specialist.** In the small reviewed subset routed to seen and predicted as bacterial blight or brown streak disease, the original expert got **14/19** right, the full-leaf specialist **17/19**, and the manually cropped symptom specialist **18/19**. The symptom regions were selected with labels visible. This is a development pilot and needs independent validation. [Cassava pilot](seen_lesion_crop/hard_cassava/results.json)

## Run or inspect the final configuration

From the workspace root with the existing Plant_Disease environment and a CUDA GPU:

```powershell
& .\Plant_Disease\.venv\Scripts\python.exe -u Disease_Transfer_Experiments/clip_anchored_gan_v1/clip_disease_only_official.py
```

The script prints the saved official results when they exist. For a fresh run, it uses the local CLIP weight file, dataset manifests, saved ConvNeXt factor logits, and development-selected offset. It saves the encoder, disease head, official predictions, and results under `clip_disease_only/official`.
