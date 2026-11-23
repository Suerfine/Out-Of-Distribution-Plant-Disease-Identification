# Out-Of-Distribution Plant Disease Identification

A COS30082 Applied Machine Learning (Sem 2, 2026) group project on **zero-shot / out-of-distribution plant disease identification**. Each image is a crop–disease pair (55 crops, 175 diseases, 323 seen and 10 unseen pairs). Models train on seen pairs only and must also recognise the unseen ones. Dataset: DPD-Small (8,000 train, 2,000 test seen, 1,448 test unseen). The spec is in `docs/Out of Distributon Plant Disease Identification v2.pdf`.

## What the report covers

- **Baselines:** Independent CNN, Multi-task (MTL) CNN
- **Methods:** CLIP, LoRA, CSP, CoOp, GAN, GATE (gate)
- **Not used:** calibration (the gamma offset). All reported numbers are at gamma = 0.

Everything else from earlier experiments (SupCon, disentangled CNN, lesion copy-paste, SimSiam, LeafGAN, cross-crop, etc.) was removed from `main` and lives on the `archive/full-experiments` branch (tag `pre-cleanup`).

## Current best (official test, gamma = 0, top-1 %)

| System | Seen | Unseen | Harmonic mean |
|---|---:|---:|---:|
| Independent CNN (ConvNeXt-B) | 73.8 | 14.0 | 23.6 |
| Disease-only CLIP (multi-task LoRA) + crop-anchored GAN + ConvNeXt seen handoff + zero-threshold gate | 72.9 | 48.3 | **58.1** |

Single seed (42) and one official test set. With 10 unseen pairs, differences under about 1 HM are noise. CoOp and CSP are not yet part of the 58.1 pipeline; the combined model is in progress. See `docs/ablation_status.md` for the ablation status and open gaps, and `Disease_Transfer_Experiments/clip_anchored_gan_v1/README.md` for the pipeline details.

## Layout

| Path | Contents |
|---|---|
| `dpd/` | Shared code: data layout, evaluation protocol (`evaluate.py`), CLIP utilities, reports, inference |
| `train/` | Training scripts: CNN baselines, multi-task CNN, CLIP zero-shot / prompt learning / LoRA, feature generation |
| `config/baseline/`, `config/multitask/`, `config/vlm/` | One JSON per experiment; `_note` / `_result` keys record intent and outcome |
| `independent_cnn_baseline/` | Independent CNN baseline notes, configs and tests |
| `Disease_Transfer_Experiments/clip_anchored_gan_v1/` | Final pipeline: CLIP LoRA + GAN + ConvNeXt + gate |
| `Disease_Transfer_Experiments/cnn_detail_base_v1/` | ConvNeXt logits and splits used by the final pipeline |
| `streamlit_app.py`, `app_pages/` | Inference UI (prediction plus matching training images) and results dashboard |
| `docs/` | `all_results.csv` (all logged runs), `ablation_status.md`, VLM notes |
| `runs/` | Outputs and caches (not tracked) |

## Setup

Install the CUDA build of torch first, then the requirements:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

The dataset is not in git. Place it at `./DPD_Small` or set `DPD_ROOT`.

## Run

```powershell
.venv\Scripts\python -m train.train_cnn --config config/baseline/aniq_baseline_independent_cnn.json
.venv\Scripts\python -m train.train_multi_task_cnn --config config/multitask/<name>.json
.venv\Scripts\python -m train.zeroshot --config config/vlm/<name>_config.json
.venv\Scripts\python -m train.train_lora --config config/vlm/<name>_config.json
.venv\Scripts\streamlit run streamlit_app.py
```

Scripts are run as modules from the repo root. Add a new config for each experiment instead of editing an old one. Choose epochs and thresholds on a validation split carved from train, never on the test sets. Log finished runs in `docs/all_results.csv` without changing its columns. See `AGENTS.md` for the full conventions.
