# Fixed protocol: crop-conditioned disease recognition V1

Authorized 6 November 2026. One experimental disease model; compare with the
saved original independent baseline. Do not train a new control or crop model.
This document and all Python/config sources are hashed before any fitting update.

## Question and model

Can shared crop-dependent feature adjustments help recognize diseases on unseen
crop-disease combinations? This is a plant-specific conditional-attribute
adaptation, not a reproduction of Learning Conditional Attributes (CVPR 2023).

Start the entire ConvNeXt-Tiny disease network and its shared 175-way classifier
from the original saved disease checkpoint (epoch 16), not ImageNet initialization.
All disease-network parameters remain trainable. Reuse the original crop model
(epoch 17), frozen, outside the optimizer. It supplies image-based crop logits.

A 55-entry crop embedding table (dimension 64) and shared linear map produce
768 scale and 768 bias adjustments. Apply 0.1*tanh to bound each adjustment.
For image features h, the classifier sees h + h*gamma(c) + beta(c). Implement
the equivalent residual classifier calculation so zero initialization exactly
reproduces baseline logits. The shared map starts at zero; crop embeddings start
normal(0,.02). There is one disease classifier shared by every crop, with no
independent classifier for each pair and no unseen-pair parameter table.

## Data and fitting

Exact original manifests: 6,746 fitting and 1,254 familiar-composition validation
images, with filename remapping and verified label/order preservation. No
pseudo-unseen validation. Official tests: 2,000 seen and 1,448 unseen images;
all 333 ontology pairs are candidates. Candidate identities are known metadata;
test images and labels do not enter gradients or checkpoint selection.

Train ordinary disease cross-entropy using the fitting image's crop label to
select its conditioning vector. Smooth labels by .1 over all 175 diseases.
No pair CE, contrastive loss, memory, calibration or test-membership signal.
Present every fitting image once each epoch, naturally shuffled, no replacement.

Seed 42 / model seed 43. Original augmentations and normalization, image size
224, batch 8 with accumulation 4, workers 2, AdamW, weight decay .05. Backbone
LR 5e-5, shared disease classifier and conditioner LR 5e-4; 3 warmup epochs then
cosine to 1%. Train at most 30 additional epochs, patience 8, norm clipping 1,
NCHW and FP16 AMP initial scale 65536. These are fresh optimizer/scheduler states.

## Prediction and selection

Compute the disease CNN once per image and score all 55 crop conditions. For
each crop, normalize logits over all 175 diseases separately. Score candidate
(c,d) by log P(c|image) + log P(d|image,c), using the unchanged crop model.
Do not use the true crop label at validation or test inference. Do not mask
seen or unseen candidates. The crop-dependent disease normalizer must be kept.

Use the original accuracy-first disease selection principle on the original
seen validation images. Its disease prediction is the marginal
P(d|image) = sum_c P(c|image)*P(d|image,c), summed over all 55 crops, including
those without a candidate pair for a particular disease. Select highest marginal
disease Top-1, then lower unsmoothed marginal disease CE; earlier exact ties win.
This explicitly defines the conditional model's analogue of the original
unconditional disease metric, without supplying ground-truth validation crops.

Include the identity-initialized saved model as epoch 0 in checkpoint selection.
It can remain selected if fitting never improves the prescribed validation rule.
Epoch 0 does not count as a fitting epoch or incur a baseline training run.
Freeze the checkpoint hash before official inference. Report all official metrics
even if worse; no test-selected score adjustment, hyperparameter or rescue run.

## Verification and reporting

Before fitting, test identity initialization, correct candidate-specific
normalization, all-crop marginalization, absence of inference-label inputs,
conditioning gradients/shared effects on unobserved pairs, selection rule and
full-image exposure. A discarded real-GPU probe must update both the disease
backbone and conditioner with finite gradients. Verify historical hashes,
original manifest membership, checkpoint selection including epoch 0, full
exposure logs and independent NumPy reconstruction of conditional probabilities,
all 333 rankings, disease marginals, exported predictions and metrics.

Write status plus 15-minute progress snapshots internally, atomic resumable
checkpoints, automatic official evaluation, paired image/composition bootstrap
intervals, a report and a training plot. Progress logs do not send chat messages.
No extra training is queued. Visual plot review is explicitly pending until seen.

Interpret as one warm-started conditional fine-tuning recipe. The saved baseline
has a different training history; this comparison cannot isolate the effect of
conditioning from additional fine-tuning. One seed and previously inspected
official tests limit claims; bootstrap intervals do not measure training variance.
