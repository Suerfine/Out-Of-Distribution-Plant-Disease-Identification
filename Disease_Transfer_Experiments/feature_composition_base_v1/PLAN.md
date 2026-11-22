# ConvNeXt-Base feature-composition development experiment

Authorized by the user's "lets do that" on 8 November 2026, Malaysia time, following the recommendation to train on real and synthesized crop-disease features. This is one new development experiment. Preserve all previous models, protocols and results. No routine baseline/control retraining or new SSL stage is required. The run stops after development reporting; a full-data refit/test run is a separate decision.

## Purpose and method

Test whether explicitly recombining crop identity and disease appearance improves recognition of real withheld combinations. This is a plant-specific feature-composition adaptation, not a reproduction of the dense feature model in [Huynh and Elhamifar, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html). Synthetic features are learned representations, not additional real plant photographs. Their realism and transfer value remain hypotheses tested on real validation images.

Start with the **saved fitting-only 20-epoch SSL ConvNeXt-Base encoder** from `simsiam_base_development_v1/runs/development_seed42/ssl_encoder.pt`. Verify its checksum, fingerprint, source roles, pretrained reference, transfer receipt, source verification and all 20 fitting-exposure banks. It saw only the 5,798 fitting images. Do not initialize from any full-8,000-image encoder or disease checkpoint. Seed 42 initializes the same original independent 175-class disease head; the whole Base backbone/head is trainable. The added modules use a separate seed 1745 without altering the original initialization stream.

The 1024-dimensional Base feature h passes through two trainable factor encoders, each LayerNorm(1024), Linear(1024,256), GELU, Linear(256,256), producing crop code c and disease code d. Auxiliary linear classifiers predict 55 crops and 175 diseases from the respective codes. A shared composer Linear(512,1024), GELU, Linear(1024,1024) maps concatenated codes back to a 1024-dimensional feature. The production disease head is the original dropout-0.2/linear head and classifies both real h and synthesized features.

At inference, use only the real image, its trained Base backbone and the shared disease head. Auxiliary factor classifiers and the composer are training modules. Reuse the unchanged saved development crop logits and the original independent crop/disease probability-product scorer. No crop label, per-image split label, synthetic target, generator output or extra score offset is supplied at inference.

## Data and synthetic targets

Keep the exact original Base manifests: fitting 5,798; seen validation 862; pseudo-unseen validation 533 over 20 withheld combinations; 807 additional images of those combinations excluded during development. All 55 crops and 175 disease labels remain in fitting. The 2,000/1,448 official tests are not loaded for this run's training or evaluation. Source inventory metadata contains the known class ontology, which does not grant training access to its images.

Known class identities may be synthesized without any real image of that combination. During development, consider only the original **323 training-ontology candidates**; exclude the ten official unseen classes from both synthesis and validation candidate ranking. Novel synthesis targets are the 20 known candidate pairs absent from real fitting images. This uses the supplied candidate class identities, not the appearances or per-image labels of validation/test images. Real validation and excluded images never supply donors. This is a generative zero-shot assumption and is explicitly part of the method.

Each logical step has 32 primary real images (last step 6), plus 16 synthetic features: **8 novel targets and 8 familiar targets**. Draw targets in balanced shuffled cycles. All 20 novel targets and 161 eligible familiar targets have fitting-only donors. For each target (crop c, disease d):

- Its crop donor has crop c and a different disease.
- Its disease donor has disease d on a different crop.
- Encode the actual donor images with the current trainable Base backbone. Combine the crop code from the first donor with the disease code from the second, then pass the resulting feature through the SAME disease head used on real images.

Deduplicate image indices within a logical step. Reuse a primary view when its image is also a donor; give additional donor views deterministic seeds varying by epoch/step. Primary shuffled order and augmentation seeds follow the earlier SSL-supervised development stream. Every fitting image appears exactly once as a primary each epoch; donor roles introduce additional, nonuniform real-image exposures. Log both separately and retain every donor/target schedule. In a complete epoch there are 182 steps and 2,912 synthetic features, including 1,456 novel targets. This complete recipe also changes target balance and total exposure; it cannot isolate feature synthesis from these changes without an additional matched control, which is not included.

## Fixed objective

Let CEreal classify the primary real features through the original disease head. Let CEfactor be the sum of the primary crop-code and disease-code classification losses. Reconstruct primary h through G(c,d), with h detached as its MSE target. For a synthesized feature G(c_i,d_j), CEsynthetic is the production disease CE plus 0.5 times each auxiliary crop and disease CE. Cycle loss is the sum of mean squared differences between L2-normalized re-encoded synthetic crop/disease codes and their detached donor codes.

`L = CEreal + 0.25 CEfactor + 0.1 reconstruction_MSE + r * (0.5 CEsynthetic + 0.1 cycle_MSE)`

`r = min(1, epoch / 3)`. Real and synthetic production-head CE use label smoothing 0.1; auxiliary CEs use unsmoothed labels. Targets and reconstruction/cycle calculations use float32. No loss weights are tuned against the official tests. This recipe is fixed before training and does not guarantee any accuracy gain.

## Training and selection

Use ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`, 87,566,464 backbone parameters), original 256 JPEG cache, 224 bicubic training crops scale 0.5-1.0, flips 0.5, color jitter 0.2 and ImageNet normalization. Keep dropout 0.2 and DropPath 0.1. AdamW: backbone LR 5e-5; original disease head and all new modules LR 5e-4; weight decay 0.05 excluding 1D parameters/biases; betas 0.9/0.999, eps 1e-8; warmup 3, cosine horizon 30, minimum LR ratio 0.01; FP16 AMP initial scale 65536 and gradient clip 1. Seed 42, two workers, encoder microbatch 4 with 2/1 OOM fallback; primary logical batch remains 32.

Cache detached logical-batch encoder features to compute joint real/donor/composition losses, then replay encoder microbatches with their exact original CPU/CUDA RNG states and backpropagate feature gradients. This keeps the full Base backbone trainable within memory limits. Replay preserves DropPath and restores the post-head RNG state. Encoder BatchNorm is forbidden because replay would update its buffers twice. Test gradients against direct stochastic autograd, including donor gradients, and verify exact GPU replay before fitting.

Maximum 30 supervised composition epochs, patience 8. Validate after every epoch on real validation images. Select the highest **gamma-zero composition Top-1 H**, earliest exact tie, using the original objective and all 323 development candidates, including the 20 held-out combinations. Restore training RNG after validation. Save every improving checkpoint and its aligned validation bank. Compare against the saved original Base development reference (seen 89.675174%, pseudo-unseen 4.502814%, H 8.575054%) and the saved SSL development reference (89.791183% / 3.752345% / 7.203652%). Neither reference is retrained. Report disease/per-combination metrics and paired decision changes as well as H.

## Recovery and verification

Before launching, test real/donor/target membership, novel-class coverage, exact schedule/augmentation resume, nonzero synthetic gradients to the shared disease head/composer/factor encoders and donor features, stochastic cached-gradient equivalence, inference without target inputs, optimizer groups, candidate masking, selection ties/patience, and storage/supervisor fault recovery. A discarded actual-GPU probe must update the backbone, production head, composer and both factor encoders and reproduce the exact next update after a checked optimizer/scaler/RNG checkpoint round trip.

Run under a detached hidden Windows supervisor with process-tree isolation and idle-sleep prevention. Keep two checksummed recovery slots, save at five-minute intervals and epoch boundaries, preserve step cursor/history/selection/optimizer/scaler/RNG, retry transient Windows replacement denials, and use stable ranking with the supported keyword dimension. Write local progress snapshots every 15 minutes. Keep at least 6 GiB free; watchdog at 15 minutes without progress; up to 12 recoverable restarts; request a checkpointed pause at a 12-hour wall-time limit. Stop blind retries for scientific/integrity failures. Power/hardware failure cannot be guaranteed recoverable.

After training, verify all source/input/reference and selected model/logit hashes, independently regenerate every primary/donor/target exposure schedule, confirm fitting-only donors, recompute selected validation metrics and earliest-maximum selection, and check finite selected weights. Write development results and stop. No calibration search, official-test evaluation or automatic full-data training follows this development run.
