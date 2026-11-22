# ConvNeXt-Base: higher resolution and local patch attention

Status checked 9 November 2026 around 9:11 am Malaysia time: **resolution384 is
complete** (15 epochs, selected epoch 7; raw seen/unseen/H
89.33%/3.75%/7.20%, below the saved Base reference). See its
[findings](RESOLUTION384_FINDINGS.md). **global_local is running epoch 3** after
two completed epochs; its best completed validation H is 5.80%. Those patch
results are preliminary. Both arms have used one worker attempt without a crash.

Startup confirmed 8 November 2026 around 8:57 pm Malaysia time. The
**resolution384** arm is training (epoch 1); **global_local** is queued to start
automatically afterward. All 18 setup tests and BOTH actual-GPU update/exact
checkpoint-recovery probes passed. Both use the same ConvNeXt-Base backbone.

Authorized 8 November 2026. The [fixed protocol](PLAN.md) runs two development experiments sequentially:

1. **resolution384:** train the original Base disease network at 384 using an original-image-derived larger cache.
2. **global_local:** use the identical 384 global view plus four real 224 patches through the same Base encoder, with learned gated attention and a zero-initialized local residual classifier.

Both start from the original pinned ImageNet Base weights. The crop model, saved 224 baseline and original 5,798/862/533 development partition are reused. Each arm has a 30-epoch cap and patience 8; raw validation H selects checkpoints. The study stops after development reporting.

Shared artifacts are in `runs/development_seed42`: `status.json` / `supervisor_status.json` show the current arm and nested worker status; `progress_15min.json` stores local progress snapshots. Each arm has its own status/history/checkpoints/results under `resolution384` or `global_local`. `results.md` at the shared root is written after both finish.

Create a shared-root `STOP.request` to request a checkpointed pause of the active worker and prevent the next arm from starting. All historical models and reports remain preserved. No different CNN or transformer is part of this study.
