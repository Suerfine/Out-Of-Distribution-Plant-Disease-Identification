# Cross-crop supervised contrastive disease training: first bounded trial

Authorized by the user's "try it" on 5 November 2026. This protocol is written
before any new training or scoring. All historical implementations and artifacts
remain unchanged. Reference: Khosla et al., Supervised Contrastive Learning,
https://arxiv.org/abs/2004.11362. The cross-crop positive mask is our adaptation.

## Question and fixed comparison

Does explicitly bringing together features with the same supplied disease label
on different crops improve recognition of crop-disease combinations not observed
during training? Shared labels are dataset categories, not assertions of a common
biological pathogen. Crop shortcuts are a hypothesis, not an established cause.

Train one independent crop CNN, then three independent disease CNNs:

1. `ce_random`: ordinary image-shuffled classification.
2. `ce_balanced`: classification with the exact grouped sampler used in arm 3.
3. `ce_supcon`: the same grouped sampler plus 0.1 times cross-crop SupCon.

The primary contrast is arm 3 minus arm 2. Arm 2 minus arm 1 measures the sampler
effect. All arms are reported. No arm is promoted using official test scores.
All three disease models start with exactly the same ImageNet-22K backbone,
random classifier and random projection head. All CNN parameters are trainable.
Each disease model is independent of the crop CNN. Use one full training seed
for this initial trial; do not treat images or policy seeds as independent CNNs.

## Data roles and selection

Read the 8,000 official training rows. Deterministically select eight development
and eight audit compositions using only training metadata and split seed
20261105. Each has 8-80 images, is not healthy, and has a different disease label
from the other 15 holdouts. Retain at least two other source crops and eight
training-pool images for each held-out disease, and at least five training-pool
images for each crop. Audit and development compositions are entirely absent
from both CNNs' training and seen validation. Split the remaining training pool
by composition into fit/seen validation, grouping exact duplicate content.
Retain every crop/disease label in fitting. Fail rather than relaxing these rules
after scores are available. Hash and verify images and official split separation.

Use all 333 ontology candidates throughout, with the unchanged score
`log_softmax(crop)[c] + log_softmax(disease)[d]`; no calibration. Official metadata
defines the ontology, but official image predictions are deferred until all
checkpoint selections are frozen. Official tests are already familiar from
previous project experiments; their results remain exploratory.

Select the shared crop checkpoint by the average of seen-validation and
development crop Top-1, then lower average ordinary CE, then earlier epoch.
Select each disease checkpoint by the harmonic mean of seen-validation and
development composition Top-1 using that same selected crop CNN, then lower
mean disease CE, then earlier epoch. Audit images are never used for training,
checkpoint selection, hyperparameters or stopping. Freeze all four checkpoints
and their hashes before audit and official predictions. The selected historical
baseline is a descriptive reference because its training partition differs.

## Training and sampler

224-pixel inputs, original mild baseline augmentations, batch 8, accumulation 4,
AdamW backbone LR 5e-5 / classifier and projector LR 5e-4, WD 0.05, label
smoothing 0.1, three warm-up epochs then cosine to 1%, gradient norm clipping 1,
FP16 AMP, channels-last, deterministic cuDNN, two data workers. Maximum 20 epochs
per model, minimum eight, early stopping after six non-improving epochs.
This is a bounded pilot, not the historical 30-epoch recipe or a multi-seed study.

The grouped sampler picks four distinct diseases per microbatch with probability
weighted by fitting-image counts. Within each disease it samples two different
crops uniformly if available, then one image per crop uniformly. Single-crop
diseases contribute two images and ordinary CE. Sampling uses replacement across
batches; the final batch preserves the same number of image presentations as
the ordinary arm. Both grouped arms use identical per-epoch indices and seeded
augmentations. Record exposure counts, unique images and eligible-anchor counts.
Gradient accumulation does not enlarge the contrastive comparison set beyond
the eight-image microbatch.

Apply a 768->256->128 ReLU projection head and normalize outputs. A positive has
the same disease label and a different crop. Different diseases are negatives;
same-disease/same-crop examples and self-pairs are excluded from the denominator.
Healthy labels are excluded as positive anchors. An anchor without an eligible
positive contributes zero contrastive loss; it still contributes CE and can be
a negative. Use stable float32 logsumexp at temperature 0.1. Average over eligible
anchors. No memory bank, extra image views, pretrained disease checkpoint, or
test-label-driven tuning. Fixed loss weight/temperature, no adaptive search.

## Verification and reporting

Test the positive/denominator mask, no-positive behavior, feature gradients,
permutation invariance, sampler pairing, matched exposures, and duplicate-safe
data roles. Preflight a real pretrained CNN forward/backward and AMP update on
fitting data. Preserve atomic epoch checkpoints, optimizer/scaler/RNG state,
source/config hashes and an exclusive OS lock for safe resumption.

Report composition Top-1/Top-5 and macro Top-1, crop/disease metrics, seen-unseen
harmonic means, per-composition counts and the untouched audit. Primary evidence
is the contrastive arm versus the sampler-matched CE arm. Report paired image
and whole-composition bootstrap intervals, explicitly conditional on one full
training seed. Verify final metrics independently from raw logits with NumPy,
checkpoint selection from every epoch, all frozen hashes and original artifacts.
Inspect training plots and report negative results, failures and limitations.
