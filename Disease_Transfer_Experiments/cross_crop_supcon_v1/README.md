# Cross-crop supervised contrastive disease training

**Completed and independently verified on 5 November 2026 at 8:25 pm Malaysia
time.** The added loss improved its sampler-matched control by 1.59 pp unseen
Top-1, but the full recipe did not outperform ordinary classification and did
not improve the reserved audit. Keep the original independent CNN as reference.
See [verified findings](FINDINGS.md) and [complete results](runs/pilot_seed42/results.md).

Isolated, authorized three-arm experiment. See [PLAN.md](PLAN.md) for the fixed
scientific protocol and [config.json](config.json) for its bounded training budget.

Use the existing Plant_Disease virtual environment. From this directory:

```powershell
../../Plant_Disease/.venv/Scripts/python.exe -m unittest -v test_study.py
../../Plant_Disease/.venv/Scripts/python.exe -u train.py --preflight-only
./run.ps1
```

The runner automatically resumes atomic epoch checkpoints. An exclusive Windows
file lock prevents duplicate controllers. All three disease arms and the shared
crop model train from the cached original ImageNet-22K weights; no historical
fine-tuned model is used. Sources and data roles are frozen before training.
All model selections are frozen before audit and official inference.

Outputs: `runs/pilot_seed42/`, including raw logits, predictions, matched exposure
counts, histories, resumable checkpoints, figures, metrics, paired uncertainty,
an independent verification and `results.md`. No historical artifacts are edited.
