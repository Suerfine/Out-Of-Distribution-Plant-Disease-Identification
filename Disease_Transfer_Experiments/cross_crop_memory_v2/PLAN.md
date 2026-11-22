# Original-split cross-crop feature-memory experiment

Authorized on 5 November 2026. Fixed before any training or scoring.

Question: does cross-crop supervised contrastive loss improve disease transfer
while preserving the original baseline's fitting/validation protocol and image
coverage? This is an adaptation of supervised contrastive learning and
cross-batch memory, not a reproduction of image-retrieval papers.
https://arxiv.org/abs/2004.11362
https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html

## Fixed controls

Reuse the exact original fit.csv (6,746 images) and validation.csv (1,254 images)
by filename and label, remapping obsolete C:/Plant_Disease paths to existing
workspace images. Preserve manifest membership/order and source files. There is
no pseudo-unseen validation, composition audit, new split, or validation refit.
Reuse the original selected crop checkpoint unchanged. Official seen/unseen
tests retain 2,000/1,448 images and all 333 candidates for both splits.

Train two independent full disease ConvNeXt-Tiny networks, both starting from
the exact original cached ImageNet-22K weights and the same seed-43 classifier
and projection head: ce_random (CE only) and ce_memory (CE + cross-crop loss).
Both execute the same queue/feature calculation; its weight is zero in the
control. Every fitting image appears exactly once per epoch, with identical
per-epoch order and seeded augmentations in both arms. The order is newly
deterministic, not a claim of bitwise reproduction of the historical run.

Use the original 224 input/augmentations, batch 8, accumulation 4, AdamW
backbone/head LR 5e-5/5e-4, WD .05, label smoothing .1, three warm-up epochs,
cosine schedule to 1%, gradient clipping 1, FP16 AMP, NCHW, and default AMP
initial scale 65,536. Both arms have the same 768->256->128 projection head;
the head is unused for inference. Maximum 30 epochs, patience eight. Select
disease checkpoints solely by seen-validation disease Top-1, then lower
ordinary CE, then earlier epoch, exactly the original selection policy.

## Cross-batch loss

Keep the most recent 1,024 detached, normalized projection vectors, together
with their disease/crop labels. Use fitting data only; reset the queue at every
epoch boundary, including resumed epochs. Compare current anchors against the
current batch and queued vectors. Same disease/different crop is positive;
different disease is negative; same disease/same crop, including self and
repeated appearances of the same image, is neutral. Healthy labels are excluded
as positive anchors. No-positive anchors still contribute CE. Old vectors
receive no gradients. Loss weight .1 and temperature .1 are fixed before runs.
Queue features become stale as weights change; the finite FIFO and epoch reset
limit their age but do not establish invariance or guarantee an improvement.

This uses more comparisons without changing sampling or adding image passes.
The queue is not enlarged by gradient accumulation. Optimizer-step skips from
dynamic AMP are recorded, not silently counted as applied updates.

## Execution and evidence

One full training seed for this first controlled trial. All fitting/validation
and source hashes are saved before training; all disease checkpoint hashes are
frozen before official inference. Never select an epoch, loss weight or method
using official scores. The tests are familiar from earlier studies; results
remain exploratory. Report both arms and the historical reference, without
promoting a test-selected winner. Primary comparison is ce_memory minus
ce_random on composition seen/unseen Top-1 and H, with factor/macro/Top-5
metrics and paired image/composition bootstrap intervals. Intervals condition
on this one training seed and do not estimate initialization variance.

Validate NumPy loss calculations, gradients, FIFO/detach behavior, neutral masks,
real-CNN forward/backward, exact manifest membership, complete shuffled coverage,
identical image exposure sequences, model selection and saved raw-logit metrics.
Preserve best/last model, optimizer, scaler and RNG state after each epoch.
The GPU probe permits dynamic-AMP overflow skips and scale reduction until a
finite optimizer update is verified; all probe updates are discarded. The
production starting scale remains the original default. Completed-model resume
also reconstructs history/summary files if interrupted between checkpoint and
report writes. The initial failed one-shot probe is archived separately before
training; it did not produce model selections or test predictions.
Resume automatically under an exclusive OS lock. A failure records a traceback.

Run under a hidden persistent background process. The training program writes
progress_15min.json and appends progress_15min.jsonl at 900-second intervals
on its normal status events, plus on completion/failure. These are intrinsic
training logs, not scheduled Codex turns or chat notifications. Generate final
predictions, reports, plots and independent numerical verification automatically.
Plots require a human/assistant visual review after completion; do not assert
that visual review has happened automatically. No recurring Codex monitoring
automation is created, as the app instructions prohibit single-run monitoring
automations. Historical models, implementations and results remain unchanged.
