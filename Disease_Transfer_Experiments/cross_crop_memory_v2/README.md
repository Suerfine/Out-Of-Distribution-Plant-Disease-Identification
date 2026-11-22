# Cross-crop feature memory on the original baseline split

**Completed and independently verified.** The feature-memory model scored
70.65% seen / 12.09% unseen / 20.64% H, below the saved original baseline.
Its small difference from the matched control was within the bootstrap interval.
See [findings](FINDINGS.md) and [generated results](runs/pilot_seed42/results.md).
Training plots were visually checked on 6 November 2026.

See [fixed protocol](PLAN.md). Two disease models use the exact original
6,746/1,254 manifests, original crop checkpoint and original validation policy.
Every fitting image appears once per epoch. No pseudo-unseen validation.

Run/resume from this directory using the existing virtual environment:

```powershell
../../Plant_Disease/.venv/Scripts/python.exe -m unittest -v test_study.py
../../Plant_Disease/.venv/Scripts/python.exe -u train.py --preflight-only
./start.ps1
```

The background runner automatically trains both arms, freezes checkpoints,
evaluates official tests, produces plots/results and independently verifies
the raw metrics. An OS lock prevents duplicate controllers. Epoch checkpoints
are resumable. Historical artifacts are read only.

Live status: [status.json](runs/pilot_seed42/status.json).
Approximately 15-minute program snapshots:
[progress_15min.json](runs/pilot_seed42/progress_15min.json) and
[history](runs/pilot_seed42/progress_15min.jsonl). These do not send chat
notifications. Final results appear in runs/pilot_seed42/results.md and
independent_verification.json. Plot visual review remains pending until inspected.
