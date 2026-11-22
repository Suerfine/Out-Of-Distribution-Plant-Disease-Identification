# Two ConvNeXt-Base image-detail experiments

Authorized by the user's "try those. i wouldnt try a different CNN yet" on 8 November 2026, Malaysia time. Run **resolution384**, then **global_local**, sequentially on the same GPU. Preserve the original Base and every earlier experiment. No different CNN, transformer, BioCLIP download, new SSL stage, routine baseline/control training, official-test evaluation or automatic full-data refit is included. These are two new development recipes with one seed each.

## Shared reference and development roles

Use the exact original Base fitting 5,798 / seen-validation 862 / pseudo-unseen-validation 533 manifests; keep the additional 807 held-out images excluded. All 55 crop and 175 disease labels remain in fitting. Only fitting images update either model. Reuse the saved development crop model/logits and baseline metrics: seen 89.675174%, pseudo-unseen 4.502814%, raw H 8.575054%. No crop model or 224 baseline is retrained.

Both arms independently start from the same pinned ImageNet-pretrained **ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k`, 1024 feature width, 87,566,464 backbone parameters), with the same fresh seed-42 175-class disease head. Do not warm start from the previous full-data or SSL models. The pretrained checksum remains `4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381`. Initialize and train each new disease pipeline once, without rerunning an old control.

## Original-image-derived inputs

Build one shared 512-short-side JPEG-quality-95 cache directly from the **original image paths** for the 7,193 fitting/validation images. Verify original file hashes against the previously audited inventory and retain source/destination hashes and original/cached dimensions. Never enlarge the existing 256-pixel cache to make this cache. Some originals are small and must be upsampled; that does not create new source detail. The fitting set contains 3,298 originals with shorter side at least 384, including 2,956 with shorter side at least 448; 1,993 have shorter side at most 256.

Training samples one crop window with the original random area scale 0.5-1.0 and aspect ratio 3/4-4/3, then shared flips (0.5 each) and color jitter (0.2 brightness/contrast/saturation, hue 0). Resize that augmented real crop directly to 384x384 for the global input. For the local arm, resize the same augmented crop to a 448x448 canvas and take its four nonoverlapping 224x224 quadrants. Both arms receive byte-identical normalized global tensors for the same image/epoch seed. The local branch makes no additional random crop or photometric decisions.

Validation center-crops a 448x448 canvas from the 512-short-side cache; its normalized field of view 448/512 equals the former 224/256. Resize that canvas to global384 and, for the local arm, split it into four local224 views. Use the original ImageNet mean/std and bicubic interpolation. Store deterministic per-image augmentation seeds and epoch shuffles for exact cursor resume. Each fitting image appears once per primary epoch, including the final six-image logical batch.

This higher-resolution recipe changes the image cache and resize sampling as well as CNN input dimensions; it is not a mathematical isolation of every preprocessing effect. The second arm uses exactly the same global view and adds local processing. Additional local views increase computation and give a complete-recipe comparison rather than a compute-matched ablation.

## Arm A: resolution384

The original independent Base disease network processes only the 384 global input. The entire backbone and head remain trainable. Keep the standard dropout-0.2/linear disease head and existing independent crop/disease probability-product scorer.

## Arm B: global_local

Use ONE shared Base encoder for the global384 view and all four local224 views. Gated MLP attention computes a softmax over four patch features (1024 -> hidden128 for tanh/sigmoid gates, hidden128 -> scalar). Pool the local features using those weights. A dropout-0.2/linear local disease head produces a residual added to the original global disease logits. Initialize local linear weights and bias to zero so the initial prediction exactly reproduces global-only prediction at equal backbone/head weights. Added attention/local modules use a separate CPU seed 1745 that preserves the original initialization RNG stream.

Apply one 175-way disease CE to the aggregate image prediction. Do NOT label every patch independently: a local patch may contain healthy tissue or background. There are no lesion boxes/masks or oracle crop/split labels. Attention weights are learned contributions, not certified lesion segmentations. This is a CNN adaptation of [attention-based multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html), not a transformer or a reproduction of that paper's benchmarks.

## Shared optimization and selection

Maximum **30 supervised epochs per arm**, patience **8**. Select the maximum uncalibrated validation composition Top-1 harmonic mean; retain the earliest exact tie. Validate on real images after every epoch using the original gamma-zero objective and 323 development candidates, with the ten official unseen columns excluded for every image. No per-image true crop/split mask or score-offset sweep. Save all improving checkpoints and aligned validation logits.

Keep original AdamW settings: backbone LR 5e-5, disease head and added attention/local parameters LR 5e-4; weight decay 0.05 except 1D parameters/biases, betas 0.9/0.999, eps 1e-8. Warmup 3 and cosine horizon 30 to a 0.01 minimum ratio; label smoothing 0.1, DropPath 0.1, dropout 0.2, gradient clip 1. FP16 AMP initial scale 65536, seed 42, two workers and prefetch factor 1. Logical image batch 32 through **microbatch 1** accumulation; encode local patches in chunks of 1. Enable supported ConvNeXt block activation checkpointing in BOTH arms to fit the 8 GiB GPU, preserving stochastic RNG replay. The same image contributes one aggregate CE regardless of its number of views.

## Reliability, sequential queue and reporting

Before launch, test shared view identity, image/cache source and geometry, deterministic data resume, attention normalization/permutation invariance, zero-residual identity, actual dependence on local pixels, aggregate-loss gradients, optimizer groups, candidate masking, checkpoint selection, storage failures and supervisor crash/hang recovery. Test that the queue runs A then B, resumes verified completed arms, and pauses without starting B when A is paused.

Perform actual-GPU preflights for BOTH arms before starting A. Each discarded probe must apply two finite updates, change the backbone/head and (for B) local/attention parameters, observe a nonzero attention gradient, and reproduce the exact next update after checksummed model/optimizer/scaler/RNG restoration. Save per-arm preflight receipts. Keep the two initialization/model/data scopes distinct with arm-specific fingerprints.

A hidden detached supervisor runs the two workers sequentially, with an exclusive study lock and killable process trees. Prevent idle sleep while each worker is active; checkpoint at least every five minutes and at epoch boundaries, keep two checksummed recovery slots, preserve exact image cursor/history/selection/optimizer/scaler/RNG, and retry transient Windows file-replacement denials. Write per-arm and shared local progress snapshots approximately every 15 minutes. No scheduled chat automation is created.

Keep at least 6 GiB free; watchdog at 15 minutes without progress, up to 12 recoverable restarts, and a 24-hour wall-time cap per arm that requests a checked pause. Scientific/integrity failures stop blind retries. Microbatch 1 is the minimum approved size; a remaining OOM requires attention. Power or hardware failures cannot be guaranteed recoverable.

On completion of each arm, verify source/original/cache/reference hashes, full exposure/order, selected checkpoint and logit identities, finite weights, reproduced selected metrics and earliest-maximum selection. On successful completion of A, automatically start the already-authorized B. After both finish, write a joint development comparison and stop. Original official tests remain outside this study; no full-data refit is automatically queued.
