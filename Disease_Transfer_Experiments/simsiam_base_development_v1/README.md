# SimSiam development on ConvNeXt-Base

Completed 8 November 2026 at 5:32 am Malaysia time; final verification passed
131 checks. The selected epoch-9 model achieved seen/pseudo-unseen/H of
89.79%/3.75%/7.20%, versus the saved Base's 89.68%/4.50%/8.58%, all without
calibration. Training completed 20 SSL and 17 supervised epochs and stopped by
the declared patience rule. Keep the Base reference. See the [findings](FINDINGS.md).

This isolated run applies the [fixed development protocol](PLAN.md) authorized on 8 November 2026. It trains SSL and the supervised disease network on the original Base's 5,798 fitting images, reuses the saved development crop logits and baseline, and selects checkpoints using uncalibrated validation results.

Budget: 20 SSL epochs, then at most 30 supervised epochs with patience 8. It stops after development reporting. No full-data refit or official-test evaluation is automatically queued.

Run artifacts are in `runs/development_seed42`: `status.json`, `supervisor_status.json`, `progress_15min.json`, histories, checked recovery checkpoints, and (on completion) `results.md`. `STOP.request` asks the worker to save and pause at the next training-step boundary. Remove that request before an explicitly requested resume.

The historical [full-data SSL run](../simsiam_base_v1/README.md) is preserved. It cannot be reused as a clean validation checkpoint because its SSL and supervised stages saw all 8,000 original training images.
