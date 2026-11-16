# Original ConvNeXt-Tiny: full-data refit

**Completed and verified: 30 October 2026, 01:27 MYT.** Both frozen refits
and official tests finished. [Final results and checkpoints](../runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42/results.md).

Authorized by the user on 29 October 2026 to measure how the original Tiny
baseline changes after restoring all 8,000 training images. This is a fresh
refit and official re-evaluation, with no new tuning or calibration search.

- Model: `convnext_tiny.fb_in22k`, original ImageNet-22K cached checkpoint,
  revision `342f2588bb2ea79d793d0712c9be16ec6ae0149a`.
- Refit crop for 17 epochs with seed 42; disease for 16 with seed 43.
  These are the original validation-selected epochs, fixed before this run.
- Fresh pretrained initialization for both networks; no warm-start from the
  previous fitted models. All 8,000 official training images are fitted.
- Preserve the original 30-epoch LR schedule and 3-epoch warmup, AdamW
  backbone/head LR 5e-5/5e-4, WD .05 applied to all parameters, original
  native timm classifier, dropout/DropPath zero, smoothing .1, clipping 1,
  224 input, physical batch 8 / accumulation 4, workers 2, CUDA FP16.
- Reuse the pinned original engine's transforms and accumulation/step logic;
  images use the original RGB decoding path without the newer JPEG resize cache.
- Finish both refits before official inference. Gamma stays zero. No validation
  loop, early stopping, model selection or further recipe trials during refitting.
- Official tests: 2,000 seen and 1,448 unseen images, all 333 candidates.
  Use the corrected shared evaluator with lower-ID exact-tie handling.
- Report composition Top-1/Top-5/H, raw independent factor metrics, separately
  labeled ten-candidate unseen-only diagnostics and the 323x333 seen confusion
  matrix. Save raw weights/logits/scores/predictions and verify app score parity.
- Preserve all original Tiny/tuning outputs and the newer three-backbone study.
  Compare to original Tiny and uncalibrated Base; remaining differences in
  initialization, augmentation and validation protocol still limit architecture claims.

[Fixed manifest](configs/tiny_full_refit_seed42.json) pins 19 original source
artifacts, legacy implementation hashes and the pretrained checkpoint hash.
New module: `independent_cnn_baseline.tiny_refit`. It uses the existing sibling
`Plant_Disease/baseline` helpers; the sibling project and its CUDA venv are required.

Original launch command (this approved run is complete):

```powershell
.\independent_cnn_baseline\run_tiny_refit.ps1 -Detach
```

The ten-hour invocation cap starts after auditing. It checkpoints before stopping;
no additional training window is started automatically. After positively verifying
controller exit, routine recovery keeps the existing deadline:

```powershell
.\independent_cnn_baseline\run_tiny_refit.ps1 -Detach -Resume -DeadlineUtc '<saved UTC deadline>'
```

A user-authorized new window uses `-Resume -Hours 10`. A user pause is requested
with `python -B -m independent_cnn_baseline.tiny_refit --request-stop`. Preserve
`STOP.request`; an explicit user resume can use `-Resume -ClearStop -Hours 10`.
The runner saves model, optimizer, LR scheduler, scaler, loader and RNG state at
every completed epoch, and keeps completed targets on resume.

Outputs: `runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42`.
`status.json` and `training.log` show progress; final files include `results.md`,
`study_summary.json`, `metrics.json`, `crop_net.pt`, `disease_net.pt`, and
`completion_verified.json`.
