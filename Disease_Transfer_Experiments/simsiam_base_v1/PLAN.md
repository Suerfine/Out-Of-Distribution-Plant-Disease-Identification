# Fixed overnight protocol: SimSiam adaptation → supervised ConvNeXt-Base

Authorized 7 November 2026. One new disease pipeline, no baseline/control or
crop retraining. Preserve all historical runs. The intended baseline is the
saved ConvNeXt-Base under the group repository, not the separate Tiny study.

## Scientific question and fixed budgets

Test whether label-free adaptation to plant images improves recognition of
novel crop-disease combinations after supervised disease fine-tuning. Start
from the SAME cached ImageNet-22K→ImageNet-1K pretrained ConvNeXt-Base weights
used by the saved baseline. Do not initialize from the plant-label-trained
disease checkpoint. Thus the plant-adaptation stage uses no crop/disease labels;
the external ImageNet initialization is itself supervised, as in the baseline.

Run20 fixed SSL epochs, then15 fixed supervised disease epochs. Both stages use
all8,000 official training images exactly once each epoch. No validation split,
pseudo-unseen partition, test-image SSL, early stopping or test-based checkpoint
selection. The final epoch of each stage is used. Existing calibrated and raw
Base reports remain the reference; no control retraining or extra SSL sweep.

## SSL method

SimSiam-style symmetric negative cosine loss between each view's prediction
and the other view's detached projection. No labels, negative pairs, queue or
momentum teacher. Shared ConvNeXt-Base encoder width1024 (>80M parameters),
fully trainable. Canonical three-layer projector1024→1024→1024→2048 with BN
and ReLU after hidden layers and non-affine BN at output; predictor2048→512→2048
with hidden BN/ReLU. No final predictor BN. This is a plant/ConvNeXt adaptation
of the published SimSiam architecture, not its ResNet/ImageNet benchmark.

The heads see a genuine64-image batch PER VIEW. Encode/replay four images at a
time using gradient caching, then backpropagate cached feature gradients through
the encoder. The encoder contains LayerNorm, no BatchNorm, and SSL DropPath/
dropout are zero, so replay is deterministic. Unit tests must match cached to
direct full-batch-head gradients. The GPU probe must confirm exact feature
replay,64-image BN batches and exact next-update recovery after a disk round trip.
Gradient accumulation is not treated as a substitute for BN batch size.

SSL SGD momentum.9, decay1e-4; initial backbone LR.001, projector/predictor
LR.0125. Two warmup epochs and cosine decay for encoder/projector; predictor
LR stays constant. These conservative backbone settings adapt pretrained
features rather than reproduce the original from-scratch100-epoch recipe.
CUDA FP16 AMP scale65536, clipping1, NCHW, seed42.

Two independently augmented views use the original256-pixel JPEG cache,
224 crops with scale.7–1, horizontal/vertical flips.5, color jitter.2 and original
ImageNet normalization. No grayscale, solarization or aggressive erasing.
The SSL dataset returns only images and image indices, never task labels.

After each SSL epoch, monitor256 fixed, unlabelled FITTING images. Record mean
standard deviation of normalized encoder/projector features and encoder
effective rank. Fail explicitly if either std<.001 or rank<4; this avoids
continuing a scientifically invalid collapsed run. These are not validation
accuracy measurements and do not select checkpoints or tune settings.

## Supervised stage and evaluation

Discard both SSL heads. Initialize a fresh175-way disease classifier with the
original Base seed42 and transfer the entire SSL encoder strictly. Fully
fine-tune encoder+head. Reuse original Base crop model/logits unchanged.

Retain Base256 JPEG preprocessing;224 crop; scale.5–1, h/v flips.5, jitter.2;
dropout.2, DropPath.1; AdamW backbone LR5e-5, head LR5e-4, WD.05 excluding
bias/one-dimensional normalization tensors, betas(.9,.999), eps1e-8, smoothing.1,
clipping1. Effective batch32 processed in physical batches4; original3 warmup
and30-epoch cosine horizon, running only its first15 epochs. Fresh optimizer/
scaler states, workers2, AMP scale65536. All8,000 training images each epoch.

Freeze the final disease checksum before any new official disease inference.
Use the unchanged independent factor-product scorer over all333 ontology pairs,
FP32 log-softmax/addition and lower-pair-ID exact tie policy. No true crop label
or per-image seen/unseen membership is supplied. Report gamma0 and the SAME
existing gamma6.2, with no calibration search. Compare matching scoring policies
against the saved Base: raw73.80/14.02/H23.56; calibrated67.00/76.52/H71.44.
Official seen/unseen counts remain2,000/1,448.

## Overnight reliability

Use an independent hidden supervisor launched outside the tool's Windows job.
Launch native Python with the venv's site-packages to avoid PID-changing Windows
venv shims. Use per-worker Windows job objects so a failed worker's DataLoader
descendants are also cleaned up. Supervisor prevents automatic idle sleep only
while alive; no permanent power settings or app automations are created.

Checkpoint at optimizer boundaries every300 seconds and every epoch. Keep two
alternating slots, explicit-stream writes, flush/fsync, ZIP CRC verification and
SHA256, with a transactional pointer. Never overwrite the active good slot.
Fall back to the previous valid slot if corruption is detected. Preserve model,
optimizer, scaler, RNG, epoch, group cursor, partial metrics and histories.
Image augmentations are a deterministic function of stage/epoch/image index;
prefetch is reproducible after resuming within an epoch.

Retry storage writes three times. Reserve6GiB free; supervisor waits for space
without deleting user data. Restart exited or stalled workers from saved state,
up to12 restarts, with backoff; watchdog requires meaningful progress within
15 minutes. CUDA OOM may reduce only physical encoder microbatch4→2→1,
preserving actual64/32 objective batches. Record every reduction/restart.
Integrity or collapse failures require attention instead of blind retries.

Twelve-hour supervisor wall-time boundary requests a checkpoint/pause; no
unbounded automatic extra training. STOP.request also pauses after a safe group.
Permanent shutdown, power loss or explicit system sleep can still interrupt
processes; the launcher can resume the saved state. The laptop should remain
powered and open for the overnight run.

Write15-minute progress snapshots, live worker/supervisor state, per-attempt
logs and automatic final evaluation, paired image/composition bootstrap,
report and independent numerical verification. No chat notifications are
promised. Source/config/data/Base hashes and owned source snapshots are frozen
before fitting. Independent verification checks both stage coverage, real SSL
encoder updates, exact encoder transfer, original head seed, final Base updates,
original artifact preservation and independently reconstructed rankings/metrics.
Plot review is explicitly pending until inspected.

One seed and familiar official tests limit conclusions. A result measures the
complete SSL→supervised recipe, including extra training and adapted settings;
it does not isolate each design choice or establish seed-to-seed significance.
