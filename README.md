# Out-Of-Distribution Plant Disease Identification

<<<<<<< HEAD
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
=======
A COS30082 Applied Machine Learning (Sem 2, 2026) group project focused on training machine learning models to identify **seen and unseen crop–disease classes** from the **DPD-Small** plant disease dataset (generalised zero-shot / out-of-distribution identification).

The dataset comprises **55 crops** and **175 diseases** across **333 crop–disease compositions** (**323 seen** during training, **10 unseen** held out for out-of-distribution evaluation). Models are trained only on seen compositions (8,000 images) and evaluated on both seen (2,000 images) and unseen (1,448 images) test splits under a shared 333-class generalised zero-shot evaluation protocol (`dpd/evaluate.py`).

## Approaches

- **Independent CNN Baselines** ([`independent_cnn_baseline/`](independent_cnn_baseline/), [`config/baseline/`](config/baseline/), [`train/train_cnn.py`](train/train_cnn.py)): Trains two separate CNN classifiers—one for crop and one for disease (ConvNeXt-Base, EfficientNetV2-L, BiT/ResNet-50)—and composes their log-probabilities across all 333 crop–disease pairs.
- **Multi-Task CNN Baselines** ([`config/multitask/`](config/multitask/), [`train/train_multi_task_cnn.py`](train/train_multi_task_cnn.py)): Uses a single shared backbone (ConvNeXt-Base, EfficientNetV2-L, ResNetV2-50/101) with joint crop and disease classification heads.
- **Pretext Self-Supervised Multi-Task CNN** ([`config/pretext/`](config/pretext/), [`train/train_pretext_multitask.py`](train/train_pretext_multitask.py)): Applies self-supervised pretext pretraining (Barlow Twins, SimSiam) before multi-task supervised fine-tuning.
- **Disentangled Multi-Task CNN** ([`config/disentangle/`](config/disentangle/), [`train/train_disentangle.py`](train/train_disentangle.py)): Splits representations into dedicated crop and disease branches with adversarial gradient-reversal ("spy") heads and feature decorrelation to reduce spurious crop–disease entanglement.
- **Crop-Normalised Disease Representation** ([`config/cropnorm/`](config/cropnorm/), [`train/train_crop_norm.py`](train/train_crop_norm.py)): Normalises disease features conditioned on crop identity to improve cross-crop disease generalisation.
- **Vision-Language Models (CLIP / SigLIP)** ([`config/vlm/`](config/vlm/), [`train/zeroshot.py`](train/zeroshot.py), [`train/train.py`](train/train.py), [`train/train_lora.py`](train/train_lora.py), [`train/train_feature_gen.py`](train/train_feature_gen.py), [`knowledge/`](knowledge/)): Explores zero-shot CLIP/SigLIP with domain-specific symptom prompts, prompt learning (CoOp / CSP), LoRA adaptation with text-anchor regularisation, feature generation, and VLM–CNN ensembles.
- **Disease-Transfer & Generative Feature Augmentation** ([`Disease_Transfer_Experiments/`](Disease_Transfer_Experiments/)): Investigates crop-anchored Feature GANs, CLIP-anchored GANs, SimSiam self-supervised disease transfer, crop-conditioned disease classifiers, and lesion copy-paste augmentation.
- **Contrastive / Compositional CNNs & Cross-Family Ensembles** ([`config/contrastive/`](config/contrastive/), [`train/train_supcon.py`](train/train_supcon.py), [`train/supcon_eval/`](train/supcon_eval/)): Combines high-resolution (336 px) ConvNeXt-Base models trained with supervised contrastive / same-crop mixing recipes, crop-weighted pair scoring, and multi-seed ensembles with CLIP-LoRA (`v10.3`) pair and disease-only evidence.

## Summary of Results

The table below compares our best-performing **Independent CNN baseline**, **Multi-Task CNN baseline**, and **Final Best Model** on the official DPD-Small test splits (all evaluated against the full 333-class space). Both **calibrated** ($\gamma > 0$, selected on validation splits) and **uncalibrated** ($\gamma = 0$) results are shown.

| Category | Experiment | Calibration ($\gamma$) | HM Top-1 | HM Top-5 | Seen Top-1 | Unseen Top-1 | Seen Top-5 | Unseen Top-5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best Independent Baseline** | `independent_cnn_baseline_convnext_base_seed42` | $\gamma = 6.2$ | 71.44% | 93.26% | 67.00% | 76.52% | 88.35% | 98.76% |
| | *(uncalibrated)* | $\gamma = 0.0$ | 23.56% | 62.50% | 73.80% | 14.02% | 91.55% | 47.44% |
| **Best Multi-Task Baseline** | `Nat_multitask_convnext_base` | $\gamma = 6.3$ | 65.82% | 93.25% | 60.25% | 72.51% | 88.50% | 98.55% |
| | *(uncalibrated)* | $\gamma = 0.0$ | 6.34% | 40.05% | 73.15% | 3.31% | 92.55% | 25.55% |
| **Final Best Model** | `supcon_mix_two_seeds_clip_disease` | $\gamma = 8.0$ | **77.04%** | **95.42%** | 66.45% | **91.64%** | 91.70% | **99.45%** |
| | *(uncalibrated)* | $\gamma = 0.0$ | 27.04% | 79.63% | 74.75% | 16.51% | 93.65% | 69.27% |

> **Note:** `supcon_ensemble4_res336_v103_two_seeds_mix` achieves a virtually tied HM Top-1 of **77.00%** and our highest HM Top-5 of **95.73%** ($\gamma = 7.5$). Complete metrics for every experiment run across all approaches are recorded in [`docs/all_results.csv`](docs/all_results.csv).

## Project Structure

```text
Out-Of-Distribution-Plant-Disease-Identification/
├── app_pages/                     # Streamlit multipage frontend views (inference, dashboard, dataset explorer)
├── config/                        # JSON experiment configs organised by model family
│   ├── baseline/                  # Required independent and multi-task CNN baseline configs
│   ├── contrastive/               # Supervised contrastive, same-crop mixing, and ensemble configs
│   ├── cropnorm/                  # Crop-normalised CNN configs
│   ├── disentangle/               # Disentangled multi-task CNN configs
│   ├── multitask/                 # Tunable multi-task CNN configs
│   ├── pretext/                   # Self-supervised pretext + multi-task configs
│   └── vlm/                       # CLIP/SigLIP zero-shot, prompt learning, LoRA, and ensemble configs
├── Disease_Transfer_Experiments/  # Standalone disease-transfer, Feature GAN, CLIP-GAN, and SimSiam studies
├── docs/                          # Project specification, technical notes, and master results log
│   └── all_results.csv            # Consolidated evaluation metrics across all experiments
├── dpd/                           # Shared core library (data loading, splits, models, evaluation, reporting, UI)
├── independent_cnn_baseline/      # Standalone independent CNN baseline training & Optuna tuning package
├── knowledge/                     # Structured disease descriptions and symptom prompts for VLM approaches
├── notebooks/                     # Google Colab notebooks for cloud GPU training
├── tests/                         # Unit tests (e.g., evaluation protocol verification)
├── train/                         # Modular training, evaluation, and ensembling entrypoints
│   └── supcon_eval/               # Validation-driven ensembling and cross-family mixing scripts
├── DPD_Small/                     # Local dataset directory (gitignored; override path via DPD_ROOT)
├── runs/                          # Experiment outputs, checkpoints, scores, and caches (gitignored)
├── streamlit_app.py               # Entrypoint for the Streamlit web application
└── requirements.txt               # Python dependencies
```

### Training Scripts (`train/`)

Core training and evaluation scripts live in [`train/`](train/) and are executed as Python modules from the project root so shared imports from [`dpd/`](dpd/) resolve cleanly:

```bash
python -m train.<script> --config <path/to/config.json>
```

Each script validates that the config's `method` field matches its expected model family and uses the shared evaluation (`dpd/evaluate.py`) and reporting (`dpd/report.py`) pipeline to write `metrics.json`, `confusion_test_seen.csv`, `predictions_<split>.csv`, and `scores.npz`. Key entrypoints include:

- `train.train_cnn` — baseline independent and multi-task CNNs
- `train.train_multi_task_cnn` — tunable multi-task CNNs with backbone selection, checkpointing, and `--resume`
- `train.train_pretext_multitask` — self-supervised pretext + multi-task CNNs
- `train.train_disentangle` — disentangled dual-branch CNNs
- `train.train_crop_norm` — crop-normalised disease models
- `train.train_supcon` & `train/supcon_eval/` — supervised contrastive / same-crop mixing models and cross-family ensembles
- `train.zeroshot`, `train.train`, `train.train_lora`, `train.train_feature_gen`, `train.ensemble` — CLIP/SigLIP VLM methods

Self-contained experimental suites with their own runners and tuning scripts are organised under [`independent_cnn_baseline/`](independent_cnn_baseline/) and [`Disease_Transfer_Experiments/`](Disease_Transfer_Experiments/).

### Config-Driven System (`config/`)

Experiments are controlled declaratively via JSON configuration files under [`config/`](config/) rather than hard-coded hyperparameters:

- Configs are grouped by model family (`baseline/`, `multitask/`, `pretext/`, `disentangle/`, `cropnorm/`, `contrastive/`, `vlm/`).
- New experiments are created by adding a new config file rather than mutating previous runs.
- Keys prefixed with `_` (such as `_note` and `_result`) document the motivation and outcome of each experiment.
- Every run writes its outputs and a snapshot of its config to the matching `runs/<family>/<name>/` directory, making runs reproducible and directly readable by the evaluation dashboard and inference app.

### Google Colab Notebooks (`notebooks/`)

For GPU-intensive training on Google Colab, ready-to-run notebooks are provided in [`notebooks/`](notebooks/):

- [`train_multi_task_cnn_colab.ipynb`](notebooks/train_multi_task_cnn_colab.ipynb) — trains multi-task CNN configs on Colab GPUs with Drive checkpointing.
- [`train_pretext_multitask_colab.ipynb`](notebooks/train_pretext_multitask_colab.ipynb) — runs self-supervised pretext pretraining (Barlow Twins / SimSiam) followed by multi-task fine-tuning.
- [`train_clip_anchored_gan_colab.ipynb`](notebooks/train_clip_anchored_gan_colab.ipynb) — executes the CLIP-anchored GAN disease-transfer experiments.

### Demo Frontend (`streamlit_app.py` & `app_pages/`)

An interactive multipage Streamlit web application is provided for live inference, model comparison, and dataset exploration:

```bash
streamlit run streamlit_app.py
```

- [`streamlit_app.py`](streamlit_app.py) — configures multipage navigation (`st.navigation`) and verifies dataset availability.
- [`app_pages/inference.py`](app_pages/inference.py) — default **Identify disease** page: upload a plant image, run any finished model from `runs/`, view the predicted crop–disease composition, and retrieve the most relevant matching training images via embedding similarity.
- [`app_pages/dashboard.py`](app_pages/dashboard.py) — **Model dashboard** comparing metrics and confusion matrices across saved runs.
- [`app_pages/overview.py`](app_pages/overview.py), [`app_pages/gallery.py`](app_pages/gallery.py), [`app_pages/unseen.py`](app_pages/unseen.py) — dataset explorer pages for inspecting class distributions, sample images, and the 10 unseen crop–disease compositions.
- [`dpd/inference.py`](dpd/inference.py) & [`dpd/ui.py`](dpd/ui.py) — shared model loader, image standardisation, retrieval bank, and Streamlit caching helpers.
>>>>>>> c3f23b1465b20c737ae8554f74757faad90ff213
