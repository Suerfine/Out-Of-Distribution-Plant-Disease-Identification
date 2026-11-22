# Overnight SSL experiment on ConvNeXt-Base

**Completed and verified:** all20 SSL and15 supervised epochs finished, and
33,909 numerical checks passed on7 November2026 at11:35 pm Malaysia time.
The SSL recipe did not improve the saved Base baseline: with the same gamma6.2,
seen/unseen/H were66.10%/74.65%/70.12%, versus67.00%/76.52%/71.44%.
See [findings](FINDINGS.md). No further training is queued.

Evaluation encountered a positional-dimension `Tensor.argsort` API mismatch.
The logged, tested adapter in `../operations/evaluation_recovery.py` finished
evaluation without changing frozen training sources or model weights. Use that
evaluation-only entry point if reproducing the existing model's reports; do not
restart the completed training pipeline. The recovery and checksum receipt is
under the run's `evaluation_recovery_verified.json`.

**Current status, 7 November 2026:** the20 SSL epochs finished and the saved
encoder was verified at7:10 pm Malaysia time. The run paused before supervised
training as requested. The user subsequently said “continue”; the original15
supervised disease epochs are now authorized to resume from that encoder.
The original numerical recipe and source snapshot remain preserved. The exact
encoder archive from the pause is also retained as `ssl_encoder_saved_at_pause.pt`.

The remaining SSL work runs through the separately logged operational guard
`../operations/ssl_stage_control.py`. At the boundary it verifies the saved
encoder against the complete SSL checkpoint, records
`runs/overnight_seed42/ssl_stage_pause_verified.json`, and exits before the
supervised model can be constructed. The supervisor then exits and releases
its temporary sleep-prevention request. Two boundary tests passed.

The stage-pause request is archived as consumed. See
`RESUME_SUPERVISED.request.json`, the verified SSL pause receipt and
`ssl_only_launch.json` in the run for the execution history. Resume the remaining
supervised stage using the normal full-pipeline launcher; do not redo SSL.

20 epochs of label-free SimSiam adaptation from the baseline's original
ImageNet-pretrained weights, then15 supervised disease epochs. The original
crop model stays frozen. No new control, pseudo-unseen validation, test-image
SSL or calibration search. See [fixed protocol](PLAN.md).

`launch.py` starts the detached hidden supervisor and returns. It requests
temporary sleep prevention, restarts failed/stalled workers, and resumes from
checksummed two-slot checkpoints. Progress snapshots are written every15 minutes;
they are files, not chat notifications. Final evaluation and verification are
automatic. `runs/overnight_seed42/supervisor_status.json` shows controller health.

The preflight verifies GPU memory, actual64-image heads through gradient caching,
noncollapsed fitting features, both trainable backbones, and exact next-update
recovery after a real GPU checkpoint round trip. Fault tests exercise worker
crashes, hangs, disk-write failures and corrupt-checkpoint fallback.

Keep the computer powered and lid open. The supervisor prevents idle sleep;
explicit shutdown/power loss still interrupts processes, and a later invocation
of the launcher resumes saved work. No permanent system power settings change.
