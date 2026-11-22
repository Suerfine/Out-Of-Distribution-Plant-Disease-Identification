# Fixed protocol: current crop-conditioned experiment on ConvNeXt-Base

Authorized 6 November 2026: replace only the current Tiny experiment with Base.
Preserve all previous runs; no rerunning historical experiments or new controls.

## Source and model

Use the saved full-data baseline at
`Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_seed42`.
The actual architecture is `convnext_base.fb_in22k_ft_in1k`, feature width 1024.
Load the entire saved disease state strictly, including its dropout/linear head.
All disease backbone, classifier and conditioning parameters are trainable.
The original Base crop model stays unchanged; reuse its verified saved official
crop logits with checkpoint, input-reference and filename alignment checks.

As in the proposed current experiment, a shared 64-dimensional crop embedding
and shared zero-initialized affine map produce feature scale/bias adjustments,
each bounded by 0.1*tanh. The zero adjustment exactly reproduces the saved
disease network. Apply the baseline's head dropout once, then modulation and
the shared disease linear classifier. No independent pair classifier is added.
This is a plant-specific conditional-attribute adaptation, not a paper reproduction.

## Training budget and data

The saved Base weights already trained on all 8,000 official training images.
The Tiny experiment's 1,254-image validation subset therefore cannot serve as
fresh validation for this warm start. Do not silently reuse it for model selection.
Instead, specify **15 additional full-data epochs** before training, matching
the original Base full-data refit length. Use the final epoch without selection,
early stopping, pseudo-unseen validation or an official-test-driven rescue.
There is no epoch-zero candidate or test-selected choice between checkpoints.

Keep all original official training images and the unchanged 2,000 seen / 1,448
unseen tests. Every fitting image appears once each epoch, no replacement.
Verify exact original Base all_train.csv membership/order, source labels and
content hashes. Score all 333 ontology pairs on both test splits.

Use the original Base 256-pixel JPEG-95 resized cache, transforms and normalization:
224 crop, training crop scale .5-1, horizontal/vertical flip .5, color jitter .2.
Keep dropout .2, DropPath .1, batch4/accumulation8 (effective32), workers2,
AdamW backbone LR5e-5, classifier/conditioner LR5e-4, decay .05 except biases
and one-dimensional normalization parameters, betas(.9,.999), eps1e-8,
label smoothing .1 and clipping1. Seed42, new optimizer and scheduler states.
Retain the Base three-epoch warmup and original 30-epoch cosine horizon, running
only its first15 steps for this fixed continuation. CUDA FP16 AMP scale65536;
NCHW. Training CE conditions on the fitting crop label and classifies175 diseases.
Do not supply validation/test crop labels to inference.

## Prediction and fair reference reporting

Compute image features once, score all55 crop conditions, and normalize disease
logits separately over175 diseases for every crop. Candidate(c,d) score is
log P(c|image)+log P(d|image,c). Disease marginal sums over all55 predicted
crop probabilities; no true crop or test-split membership is used per image.

Match the saved Base scorer's FP32 log-softmax and FP32 pair addition. Export
raw logits and both rounded log-probability tables, then store pair scores as
float64 so existing calibration subtraction preserves precision. Exact score
ties prefer the lower composition ID, as in the shared Base evaluator.

Report BOTH gamma0 and the baseline's existing fixed gamma6.2. Gamma6.2
subtracts from323 seen-pair scores using known ontology membership, without
knowledge of whether any particular input is seen or unseen. No new gamma search
or recalibration is performed. Compare each scoring policy to the saved Base
under the SAME policy: raw73.80/14.02/H23.56 and existing calibrated
67.00/76.52/H71.44 (seen/unseen/H percentages). Preserve both historical reports.

Freeze the final checkpoint hash after exactly15 additional epochs, before any
new official disease inference. Report the final model even if it is worse.

## Verification and operation

Before full training, check identity initialization, conditional normalizers,
all-crop marginalization, no inference-label inputs, shared gradients, correct
Base optimizer groups, full exposure and resumable final-checkpoint recovery.
A discarded real-GPU step must update the Base backbone, shared disease head
and conditioner with finite gradients. Confirm width1024 and >80M backbone
parameters to guard against accidental Tiny usage.

Independent final checks cover frozen source/input/Base hashes, all cache and
raw-image hashes, exact 8,000-image training membership,15 complete epochs,
final-checkpoint identity, full backbone/conditioner changes and exposure logs.
Verify probabilities against NumPy, reconstruct FP32 pair addition exactly from
exported log probabilities, recompute all rankings, predictions, metrics, H and
paired baseline differences for both frozen scoring policies. Bootstrap paired
image and whole-composition intervals, without claiming new-seed uncertainty.

Run in a hidden background process. Save atomic resumable checkpoints and
15-minute progress logs; automatically evaluate, verify and report. Logs do not
send chat notifications. Review the generated plot visually when available;
record visual review as pending until inspected. No further training is queued.

This comparison tests the complete warm-started conditional fine-tuning recipe;
it cannot isolate conditioning from extra training without an additional control.
Only one seed and previously inspected official tests are available. Results of
the earlier Tiny experiments do not establish outcomes for Base.
