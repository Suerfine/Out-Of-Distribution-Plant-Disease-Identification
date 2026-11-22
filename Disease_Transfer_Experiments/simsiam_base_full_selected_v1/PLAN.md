# Full-data SSL recipe selected on development

Authorized by the user's "can we do full SSL now" on 8 November 2026, Malaysia time. This is a separate full-data refit and official-test evaluation. Preserve the existing Base, previous full-data SSL model, and development run.

## Fixed recipe

Use **ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k`, 1024 features), 20 epochs of the existing SimSiam plant-image SSL recipe, followed by **9 supervised disease epochs**. Nine is the earliest maximum of uncalibrated validation composition Top-1 H in the completed development run. It was selected on the original Base split of 5,798 fitting, 862 seen-validation, and 533 pseudo-unseen-validation images, with another 807 images excluded during development. The official tests are not used to choose this new run's budget or checkpoint.

For this full-data stage, both SSL and supervised training use **all 8,000 original training images**, including the development validation/excluded images. The 2,000 official seen and 1,448 official unseen test images remain outside all training. Rank all 333 candidate combinations for every test image.

## Reuse the identical completed full-data SSL stage

The earlier `simsiam_base_v1/runs/overnight_seed42/ssl_encoder.pt` already contains the full-data 20-epoch SSL encoder needed by this recipe. Before reuse, verify its checksum against the original supervised-transfer receipt; its original fingerprint, tensor identity, all 20 image-exposure banks, source verification, full 8,000-image membership/order and image hashes, pinned pretrained weights, model/view/loss/optimizer source, and SSL configuration. Match the SSL recipe to the development run too.

Copy the encoder and SSL histories/exposures byte for byte into this isolated run, retaining the original source fingerprint and recording provenance in `ssl_reuse.json`. Repeat **zero SSL updates**. This saves the cost of recomputing an identical stage. The existing 15-epoch disease model is not used as a warm start.

Initialize a fresh 175-class disease head with seed 42 and copy only the verified pre-supervised SSL encoder. Start a fresh AdamW optimizer and LR schedule; train the full encoder and head for exactly 9 epochs on all 8,000 images. The supervised head initialization must match the original seed-42 head hash. Do not continue from the old 15-epoch model or truncate its reported history.

## Training details

Reuse the original Base image cache (shorter side 256 JPEG), 224 bicubic crops, ImageNet normalization, supervised crop scale 0.5-1.0, horizontal/vertical flips 0.5, and color jitter 0.2. AdamW backbone LR 5e-5, head LR 5e-4, weight decay 0.05 except 1D parameters/biases, betas 0.9/0.999, eps 1e-8; warmup 3 and original cosine horizon **30**, minimum LR ratio 0.01. Keep the 30-epoch schedule when stopping at epoch 9, matching development. Effective batch 32 through microbatch 4 accumulation, with 2 or 1 only as an OOM fallback. Dropout 0.2, DropPath 0.1, label smoothing 0.1, gradient clip 1, FP16 AMP initial scale 65536, seed 42, two workers. Expose every fitting image exactly once each epoch.

The reused SSL recipe is unchanged: two independently augmented views per image, 224 crop scale 0.7-1.0, flips/jitter, SimSiam stop-gradient negative cosine loss, projector 1024/1024/2048 and predictor 2048/512/2048, logical head batch 64, SGD momentum 0.9 and weight decay 0.0001, encoder LR 0.001 and projector/predictor LR 0.0125, warmup 2 and cosine over 20 epochs, constant predictor LR, and zero encoder DropPath for exact gradient caching. No crop/disease labels or test images enter SSL.

## Frozen test evaluation

After supervised epoch 9, save and checksum the final disease model and freeze selection before computing its test predictions. Reuse the unchanged full-data Base crop model and its verified saved logits. Rank the product of independent crop/disease probabilities over all 333 combinations without per-image true-crop or true-split masks.

**Gamma-zero accuracy and H are primary.** Report the historical fixed gamma 6.2 separately, with no sweep or new tuning. The original Base final results are the reference; do not retrain a control. The tests have already been inspected in previous experiments, so these are exploratory comparisons, not untouched confirmation. This is one seed, and epoch budgets differ between the saved Base and this selected SSL recipe.

## Reliability and checks

Before training, test SSL-setting matching, the development-selected budget, stable ranking, recovery storage, the observed Windows transient file-replacement failure, gradient caching, and supervisor crash/hang recovery. Run a discarded GPU probe using the actual reused encoder and fresh supervised head, verifying finite backbone updates and exact next-update replay of model/optimizer/scaler/RNG after checkpoint restoration. Check the original SSL GPU probe as part of provenance.

Use a detached hidden Windows supervisor, idle-sleep prevention, process-tree isolation, checksummed two-slot checkpoints at five-minute intervals and epoch boundaries, exact cursor/RNG/augmentation resume, and local 15-minute progress snapshots. Retry transient Windows replacement denials before failing a status/checkpoint write. Use the supported keyword `dim` for stable Torch sorting. Preserve old experiment sources; these reliability fixes apply only to this new run.

Keep at least 6 GiB free, use a 15-minute no-progress watchdog, allow up to 12 recoverable restarts, and request a checkpointed pause at a 12-hour wall-time limit. Scientific/integrity failures stop blind retries. Hardware failure or power loss cannot be guaranteed recoverable.

After test inference, independently check all source/data/reference/reuse hashes, full image exposure, initialization and SSL transfer, the fixed nine-epoch disease checkpoint, score construction, raw and fixed-gamma metrics, prediction exports, and saved Base reference counts. Write metrics/reports and stop. No subsequent experiment is automatically queued.
