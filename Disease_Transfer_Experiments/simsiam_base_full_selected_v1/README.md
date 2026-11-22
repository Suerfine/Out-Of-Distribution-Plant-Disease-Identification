# Full-data SimSiam Base with the selected supervised budget

Completed 8 November 2026 at 1:36 pm Malaysia time on the first worker attempt;
33,937 verification checks passed. Raw official seen/unseen/H were
73.00%/14.30%/23.91%, versus Base's 73.80%/14.02%/23.56%. This is a small raw H
gain of 0.35 percentage points; its composition-bootstrap interval includes
zero. The unchanged fixed-gamma secondary H is 71.14%, versus Base's 71.44%.
See the [findings](FINDINGS.md). No further training is queued by this run.

Authorized 8 November 2026. The [fixed protocol](PLAN.md) uses the verified existing **20-epoch SSL encoder trained on all 8,000 images**, then trains a fresh disease model for **9 supervised epochs**, as selected by the completed development run. The crop model and original Base comparison are reused.

Only the already completed identical SSL stage is reused; the supervised stage starts from the pre-supervised encoder, a fresh seed-42 head, and a fresh optimizer. Official test inference follows the fixed training budget. Raw gamma-zero results are primary, with unchanged fixed gamma 6.2 reported separately.

Artifacts are in `runs/full_selected_seed42`, including `status.json`, `supervisor_status.json`, `ssl_reuse.json`, checked recovery checkpoints, and `progress_15min.json`. On completion: `results.md`, `metrics.json`, test predictions and `independent_verification.json`. A `STOP.request` asks for a checkpointed pause at the next training-step boundary.

The [development run](../simsiam_base_development_v1/FINDINGS.md) and [earlier full-data run](../simsiam_base_v1/FINDINGS.md) remain preserved.
