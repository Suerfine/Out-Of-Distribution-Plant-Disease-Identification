# Compositional feature augmentation on ConvNeXt-Base

Completed 8 November 2026 at 5:02 pm Malaysia time on the first worker attempt;
7,747 verification checks passed. Training stopped after 13 epochs and selected
epoch 5. Raw seen/pseudo-unseen/H were 87.47%/2.44%/4.75%, below Base's
89.68%/4.50%/8.58% and the saved SSL development reference. See the
[findings](FINDINGS.md). The run stopped after development reporting; no further
training is queued by it.

Authorized 8 November 2026. The [fixed protocol](PLAN.md) trains one disease pipeline using real fitting images and synthesized crop-disease features. It reuses the saved fitting-only SSL encoder, original Base development partition, crop logits, baseline results and SSL development reference.

The Base backbone remains trainable. New factor encoders and a composer build synthetic 1024-dimensional features from fitting-image donors; the same disease head classifies real and synthetic features. At inference, the model uses the real image with the ordinary independent disease head, and reuses the existing crop scorer.

Budget: at most 30 supervised epochs, patience 8; selection is raw validation H on 862 seen and 533 pseudo-unseen images. Every primary epoch covers the 5,798 fitting images once, with additional logged donor exposures. The run stops after development reporting.

Artifacts: `runs/development_seed42/status.json`, `supervisor_status.json`, `progress_15min.json`, histories, exact exposure banks and checked recovery checkpoints. Completion writes `results.md`, `results.json` and `verification.json`. A `STOP.request` asks for a checkpointed pause at the next training-step boundary.

No original model or historical experiment is replaced. This is a new method with no promised improvement; real held-out-image performance determines its usefulness.
