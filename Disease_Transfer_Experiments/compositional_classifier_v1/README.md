# Full compositional-classifier experiment

**Completed and independently verified.** The model scored 70.55% seen /
0.14% unseen / 0.28% H. This is a severe unseen-combination failure; retain the
saved original baseline. See [findings](FINDINGS.md) for the interpretation and
the clearly separated post-hoc ranking diagnostics. Plot review is complete.

[Fixed scientific protocol](PLAN.md). One fully trainable ConvNeXt image encoder,
shared crop/disease embeddings and learned pair composition. No retrained control,
no pseudo-unseen validation. Exact original 6,746/1,254 manifests and original
accuracy-first validation policy applied to its primary pair output, with pair
CE breaking ties. This adapts the old separate factor-head selection rule.
All 333 candidates are present at inference.

The architecture replaces the independent crop/disease probability product with
learned joint compatibility. It is inspired by compositional-zero-shot-learning
principles; it does not reproduce CGE's graph or its published benchmark setup.

Run/resume with the existing environment:

```powershell
../../Plant_Disease/.venv/Scripts/python.exe -m unittest -v test_study.py
../../Plant_Disease/.venv/Scripts/python.exe -u train.py --preflight-only
./start.ps1
```

The hidden background runner completes training, freezes the validation-selected
checkpoint, evaluates official tests, compares saved original baseline predictions,
generates reports/plots and independently verifies the numerical results. Epoch
checkpoints retain optimizer/scaler/RNG states and resume under an OS lock.

Live file: [status](runs/full_seed42/status.json). Approximately 15-minute program
snapshots: [latest](runs/full_seed42/progress_15min.json) and
[history](runs/full_seed42/progress_15min.jsonl). These are local logs, not chat
notifications or recurring Codex automation. Results appear in
`runs/full_seed42/results.md`; numerical verification in
`runs/full_seed42/independent_verification.json`. Training plot visual inspection
remains pending until reviewed after completion.
