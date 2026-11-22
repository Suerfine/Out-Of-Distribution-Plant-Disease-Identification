# SimSiam Base development run

Authorized by the user's "start it" on 8 November 2026, Malaysia time. This applies the prepared original Base development protocol. Historical full-data experiments and references are preserved.

## Fixed scope

Reproduce the existing SimSiam adaptation recipe on the exact original Base development fitting partition. This is one development experiment, with no baseline/control retraining, official-test inference, calibration sweep, automatic full-data refit, or hyperparameter search.

Use ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`, 1024 features) from the same pinned ImageNet-pretrained weights. Run 20 fixed SSL epochs, then supervised disease training for at most 30 epochs with patience 8. Select the highest uncalibrated validation composition Top-1 harmonic mean, using the earliest exact tie. Only supervised epochs participate in checkpoint selection. These budgets retain the previous SSL recipe and original Base development supervised budget; neither is claimed optimal.

## Immutable image roles

Reuse `independent_cnn_baseline_study_seed42`: fitting 5,798; seen validation 862; pseudo-unseen validation 533 across 20 withheld combinations; additional 807 held-out-combination images excluded during development. Every fitting image appears exactly once per epoch in both stages, including partial final batches (38 SSL images, 6 supervised images). All 55 crops and 175 diseases occur in fitting. No validation, excluded, or official-test image is used for SSL or supervised updates.

The development crop model is the saved `independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000/crop/best_epoch_15.pt`. Reuse its saved validation crop logits in original manifest order. Compare with the same trial's `validation_epoch_15.npz`; reproduce reference S=89.675174%, U=4.502814%, H=8.575054%. No full-8,000-image crop model or disease warm start is used for this development comparison.

## SSL recipe

The same 3-layer projector (1024, 1024, 2048), BatchNorm, and 2048-to-512-to-2048 predictor as the completed Base SSL experiment. Symmetric negative cosine similarity with stop-gradient targets. No labels, teacher, feature memory, or queue. Disable encoder DropPath/dropout for exact gradient caching. Logical batch 64; encoder microbatch 4 (only fall back to 2 or 1 for CUDA OOM). Head BatchNorm sees the entire logical view, including the final 38-image batch.

SGD momentum 0.9, weight decay 0.0001; backbone LR 0.001, projector and predictor LR 0.0125. Two warmup epochs then cosine decay for encoder/projector; predictor LR constant. AMP FP16 with initial scale 65536; clip gradients to 1.0. Views: 224 bicubic random crop scale 0.7-1.0, horizontal and vertical flip 0.5, color jitter 0.2, original ImageNet normalization. The two views are independently sampled transformations of the same fitting image.

## Supervised recipe and selection

Discard SSL projector/predictor. Initialize a new 175-class disease head using seed 42, copy the SSL encoder, and train the whole disease network. Original Base AdamW groups: backbone 5e-5, head 5e-4, weight decay 0.05 excluding 1D parameters/biases, betas 0.9/0.999, eps 1e-8. Effective batch 32 through microbatch accumulation; warmup 3, cosine horizon 30, minimum LR ratio 0.01; label smoothing 0.1; dropout 0.2; DropPath 0.1; gradient clip 1.0. Original supervised 224 crop scale 0.5-1.0 and flip/jitter transforms; same shorter-side-256 JPEG cache and normalization.

After each complete supervised epoch, infer disease logits on the 1,395 validation images and combine them with the saved development crop logits. Use the original gamma-zero objective and tie rule. Score 323 development candidates (303 fitting plus 20 pseudo-unseen); mask the 10 official unseen columns for every validation image. Do not supply the true crop or per-image split as an inference mask. Stop after 8 consecutive nonimproving supervised epochs or at epoch 30. Save each improving model and its validation bank, then choose the earliest maximum. Report disease and per-combination metrics plus paired changes from the saved development baseline.

## Execution and recovery

Before fitting, run gradient, split/selection, and real storage/supervisor tests, plus a discarded GPU probe that verifies encoder updates and exact checkpoint next-update replay. Save checksummed transactional two-slot training checkpoints at least every five minutes and at epoch boundaries, including optimizer/scaler/RNG/cursor/history/selection state. Deterministic image order and per-image augmentation permit resumption. Unique epoch-specific best checkpoints prevent a crash from overwriting an older selected model.

A detached hidden Windows supervisor prevents idle sleep while alive, uses a process-tree job, restarts recoverable crashes, falls back to smaller encoder microbatches on OOM, and refuses blind retries for scientific/integrity failures. Keep a 6 GiB free-disk reserve, watchdog at 15 minutes without progress, up to 12 restarts, and a 24-hour wall-time cap that requests a checkpointed pause. This cannot guarantee recovery from power loss or every hardware/OS failure. Progress snapshots are written every 15 minutes locally; no app automation or chat polling is required.

When development training finishes, verify image exposure, source/reference hashes, selection, model checksum, and saved metrics; write `results.md` and stop. Do not automatically start full-data training or access official test images.
