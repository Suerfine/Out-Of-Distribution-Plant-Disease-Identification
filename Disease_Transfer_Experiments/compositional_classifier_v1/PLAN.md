# Full learned compositional-classifier experiment

Authorized 6 November 2026. Fixed before model training or model scores.
Train ONE complete model end-to-end; reuse the saved original baseline and its
predictions, without retraining a baseline/control. This is a plant-specific
learned composition architecture, not an exact CGE or CompCos reproduction.
Relevant principles: https://arxiv.org/abs/2102.01987 and
https://arxiv.org/abs/2101.12609. No claim of reproducing published benchmarks.

## Data and model

Use the exact original 6,746 fitting / 1,254 seen-validation manifests, matched
by filenames and labels and remapped to workspace image paths. No pseudo-unseen
validation, refit, extra held-out split or test-driven parameter selection.
Official splits remain 2,000 seen and 1,448 unseen images. The supplied ontology
contains 333 candidate pairs, and both official splits always use all of them.

A full ImageNet-22K pretrained ConvNeXt-Tiny image encoder is trainable. Learn
55 crop and 175 disease embedding vectors (dimension 256). A SHARED MLP composes
the normalized crop/disease vectors: concatenate 512 -> hidden 512 -> ReLU ->
256, then normalize. It constructs every pair vector from its components;
there is no independently learned row/classifier vector for each of 333 pairs.
An image projection 768 -> 512 -> ReLU -> 256 scores all pair vectors by cosine
similarity with fixed scale 20. Separate linear visual crop/disease projections
score the same primitive embeddings and supply factor supervision.

Loss is pair CE + 0.5 crop CE + 0.5 disease CE, all with label smoothing .1.
Pair CE normalizes over the 323 fitting compositions ONLY. The 10 official
unseen pairs are not negative training classes. Their component embeddings and
the composer are learned from other pairs. All 333 pair scores are used for
validation metrics and official prediction, without test-split membership,
feasibility masks, probability-product fallback, calibration or bias tuning.
The auxiliary factor predictions are diagnostics, not a product used to rank
compositions. Tests must verify evaluation actually uses the learned pair score.

This joint model replaces the two independent factor networks for its own
predictions. The historical crop model is not used to supply oracle crop labels
or predictions to the experimental model. The saved independent baseline is
the external reference and supplies paired prediction comparisons only.

## Training and checkpoint policy

One initialization seed, model seed 43 (configured base seed 42 plus one), same
cached original pretrained backbone. Train all encoder, projection, embedding
and composer parameters. 224 inputs and the original mild augmentations;
batch 8 with accumulation 4; every fitting image once per epoch; deterministic
per-epoch shuffle and worker augmentation seeds. The exact shuffled order is a
new stream, so this is not bitwise reproduction of a historical CNN run.
AdamW backbone LR 5e-5 / new parameters LR 5e-4, WD .05, original three-epoch
warmup and cosine to 1%, clipping 1, FP16 AMP with original initial scale 65,536,
NCHW. Maximum 30 epochs, patience eight, no arbitrary shorter pilot phase.
Apply the original accuracy-first / lower-loss tie-break principle to the new
primary output: select by seen-validation PAIR Top-1, then lower ordinary pair
CE, then earlier epoch. This is an explicit adaptation from the historical
separate factor-head selection rules, fixed before any training or model scores.
The validation images remain exactly the original 1,254; auxiliary factor
scores do not select checkpoints. Never select a test winner or alternate score.

## Verification and execution

Before training: test novel pair construction from shared factors, composer's
dependence on BOTH factor identities, gradient flow to both factor tables and
composer, exclusion of unseen negatives, rejection of held-out training labels,
actual joint-score evaluation, exact original manifest membership, full exposure,
and complete-checkpoint report recovery. A real fitting-image GPU probe must
verify a finite applied optimizer step changes the CNN and compositional head;
allow normal dynamic-AMP scale recovery and discard all probe updates.

Freeze source/config/data hashes before training, checkpoint hash before official
inference. Save atomic last/best weights, optimizer, scaler, RNG, exposure arrays,
and histories for epoch-boundary resumption under an exclusive OS lock. Recover
missing histories/summaries when resuming a completed checkpoint. Preserve all
historical sources, models, predictions and reports. Check baseline prediction
filenames/targets and correct counts against its frozen metrics before using it.

Report all 333-candidate composition and auxiliary factor Top-1/Top-5/macro
scores, per-composition counts, H, and paired image/whole-composition bootstrap
intervals against saved original baseline predictions. Independently verify with
NumPy from raw logits and exported rankings. Include training curves and a
generated explanation of results. No seed variance estimate from one seed;
official tests are familiar from previous experiments and remain exploratory.

Retain the user's background-run preference. The runner produces results and
numerical verification automatically and saves approximately 15-minute progress
logs as part of normal training; no scheduled assistant/chat automation is
created. Visual plot review remains pending until inspected after completion.
