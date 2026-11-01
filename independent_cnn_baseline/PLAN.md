# Independent timm baseline overnight study

Historical full-sweep plan. On 28 October 2026 the user chose to prepare the
smaller [R0 refit-only plan](REFIT_ONLY_PLAN.md) and keep training paused.
The original configs and training artifacts remain preserved.

Approved on 28 October 2026. This study preserves the existing group experiments
and the earlier standalone ConvNeXt-Tiny runs.

Three backbones: `resnetv2_50x1_bit.goog_in21k_ft_in1k`,
`convnext_base.fb_in22k_ft_in1k`, `tf_efficientnetv2_l.in21k_ft_in1k`.
Each recipe fits independent crop and disease networks. All use one saved
seed-42 seen/pseudo-unseen validation partition; official tests are excluded
from every selection decision.

The ordered recipes are R0 (LR 5e-5/5e-4), R1 (2e-5/2e-4),
R2 (1e-4/1e-3), then R3 (selected LR, WD 0.01 versus 0.05),
R4/R5 (selected LR/WD, separate head dropout 0/0.4 versus 0.2).
Complete each round across all three backbones before the next round.
Select by validation composition Top-1 harmonic mean; retain the earlier
candidate on exact ties. Select epoch and gamma within each recipe by that
same metric, retaining the earliest epoch and smallest gamma on exact ties.

Common training: 224 pixels, AdamW, effective batch 32, starting physical
batch 4, 30 maximum epochs, 3 warmup epochs, cosine decay to 1%, label
smoothing 0.1, gradient clipping 1, head dropout 0.2 initially, DropPath 0
for BiT / 0.1 for ConvNeXt and EfficientNet, CUDA FP16 AMP. Freeze
EfficientNet BatchNorm running statistics, but train affine parameters.
Exclude bias and normalization parameters from weight decay. Use each
checkpoint's recorded normalization and interpolation. Fixed augmentation:
crop scale 0.5-1, horizontal/vertical flips 0.5, brightness/contrast/saturation
jitter 0.2, hue 0; no Mixup, CutMix, erasing or TTA.

Use paired per-epoch validation, with only one network and its optimizer on
the GPU at a time. Early stopping patience is eight paired epochs without
a validation HM improvement. Checkpoints include model/optimizer/scaler/RNG
state and the data-loader generator. Partial epochs repeat after interruption.
The fixed LR-versus-epoch trajectory is preserved in final refitting.

The complete study has 18 recipe pairs (36 selection trainings), followed
by six fresh selected-model refits on all 8,000 training images. Freeze all
three final choices before refitting and official test inference. Report
both gamma-zero and validation-calibrated 333-candidate results through
the existing shared evaluator and report format.

One invocation has a ten-hour training window. Finish and checkpoint the
current epoch at its deadline, record pending work, then exit cleanly.
Resume the same study explicitly with `--resume`; never launch duplicate
controllers. An OS lock and atomic status/checkpoint files protect recovery.
Do not infer a winner from an unfinished queue. CUDA is mandatory.

Before training, verify data integrity, cache pinned pretrained weights,
perform actual CUDA forward/backward/optimizer checks, and record throughput
and memory. Memory-compatible physical batches may be selected only during
preflight (effective batch stays 32), then frozen for every trial of that
backbone. Store complete recovery states for unfinished jobs and compact
verified weights/histories for completed selection jobs. Original artifacts
are never pruned. Run outputs, caches and data remain gitignored.

App inference must reproduce cached downscale/JPEG preprocessing and use
the same model-specific transforms, architecture options and precision.
Completion requires finite metrics/logits, correct prediction counts,
checkpoint hashes and shared-app inference agreement. Monitoring is quiet
during normal progress and reports completion, actionable failures or input
needed; a deadline stop is reported as incomplete, with a resume command.
